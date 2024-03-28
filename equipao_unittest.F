!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2024 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!

PROGRAM equipao_unittest

   USE cp_files,                        ONLY: discover_file
   USE kinds,                           ONLY: default_path_length,&
                                              dp,&
                                              int_8,&
                                              sp
   USE mathlib,                         ONLY: inv_3x3
   ! USE physcon,                         ONLY: angstrom,&
   !                                           evolt
   USE torch_api,                       ONLY: &
        torch_cuda_is_available, torch_dict_create, torch_dict_get, torch_dict_insert, &
        torch_dict_release, torch_dict_type, torch_model_eval, torch_model_load, &
        torch_model_read_metadata, torch_model_release, torch_model_type
#include "./base/base_uses.f90"

   IMPLICIT NONE

   CHARACTER(LEN=default_path_length) :: filename_o, filename_h, cutoff_str
   REAL(dp) :: cutoff

   ! Inputs.
   INTEGER, PARAMETER  :: natoms = 6
   ! INTEGER :: iatom, nedges
   REAL(sp), DIMENSION(3, natoms) :: pos
   REAL(dp), DIMENSION(3, 3):: cell, hinv
   REAL(sp), DIMENSION(2,natoms):: atom_types

   ! Torch objects.
   TYPE(torch_model_type) :: model_o
   TYPE(torch_dict_type) :: inputs, outputs

   ! Outputs.
   REAL(sp), DIMENSION(:, :, :), POINTER :: pao_vectors
   REAL(sp), DIMENSION(:), POINTER :: gradient
   NULLIFY (pao_vectors, gradient)

   ! Water Dimer.
   pos = RESHAPE(REAL([ &
                      2.5270_dp, 4.1865_dp, 4.3500_dp, &
                      2.0350_dp, 4.7115_dp, 3.7100_dp, &
                      2.1940_dp, 3.2885_dp, 4.2560_dp, &
                      5.4400_dp, 4.2345_dp, 3.6350_dp, &
                      5.9650_dp, 4.5745_dp, 4.3650_dp, &
                      4.5220_dp, 4.2535_dp, 3.9470_dp],kind=sp),shape=[3, natoms])

   cell(1, :) = [8.0_dp, 0.0_dp, 0.0_dp]
   cell(2, :) = [0.0_dp, 8.0_dp, 0.0_dp]
   cell(3, :) = [0.0_dp, 0.0_dp, 8.0_dp]

   hinv = inv_3x3(cell)

   atom_types(1,2) = 1.0 ! Hydrogen
   atom_types(1,3) = 1.0 ! Hydrogen
   atom_types(1,5) = 1.0 ! Hydrogen
   atom_types(1,6) = 1.0 ! Hydrogen
   atom_types(2,1) = 1.0 ! Oxygen
   atom_types(2,4) = 1.0 ! Oxygen

   atom_types(2,2) = 0.0 ! Hydrogen
   atom_types(2,3) = 0.0 ! Hydrogen
   atom_types(2,5) = 0.0 ! Hydrogen
   atom_types(2,6) = 0.0 ! Hydrogen
   atom_types(1,1) = 0.0 ! Oxygen
   atom_types(1,4) = 0.0 ! Oxygen           

   WRITE (*, *) "CUDA is available: ", torch_cuda_is_available()

   filename_o = discover_file('/home/ubuntu/software/cp2k/data/EquiPAO/pao_equi_model_O.pth')
   filename_h = discover_file('/home/ubuntu/software/cp2k/data/EquiPAO/pao_equi_model_H.pth')
   WRITE (*, *) "Loading EquiPAO model from: "//TRIM(filename_o)
   CALL torch_model_load(model_o, filename_o)
   ! CALL torch_model_load(model_h, filename_h)
   cutoff_str = "4.0"
   READ (cutoff_str, *) cutoff
   WRITE (*, *) "Cutoff: ", cutoff
   WRITE(*, *) "Central atom: ", pos(:,1)
   WRITE(*, *) "Neighbours: ", pos(:,2:)

   ! CALL neighbor_search(nedges)
   ! ALLOCATE (edge_index(nedges, 2))
   ! ALLOCATE (edge_cell_shift(3, nedges))
   ! CALL neighbor_search(nedges, edge_index, edge_cell_shift)
   ! WRITE (*, *) "Found", nedges, "neighbor edges between", natoms, "atoms."

   CALL torch_dict_create(inputs)
   CALL torch_dict_insert(inputs, "x", pos(:, 1))
   CALL torch_dict_insert(inputs, "pos", pos(:, 2:))
   CALL torch_dict_insert(inputs, "z", atom_types(:,2:))

   CALL torch_dict_create(outputs)
   CALL torch_model_eval(model_o, inputs, outputs)

   CALL torch_dict_get(outputs, "pao_vectors", pao_vectors)
   CALL torch_dict_get(outputs, "gradient", gradient)

   WRITE (*, *) "PAO Vector : ", pao_vectors
   WRITE (*, *) "Gradient: ", gradient


   CALL torch_dict_release(inputs)
   CALL torch_dict_release(outputs)
   CALL torch_model_release(model_o)
   DEALLOCATE (pao_vectors, gradient)

   WRITE (*, *) "Ran EQUI-PAO test!"

CONTAINS

! **************************************************************************************************
!> \brief Naive neighbor search - beware it scales O(N**2).
!> \param nedges ...
!> \param edge_index ...
!> \param edge_cell_shift ...
! **************************************************************************************************
   SUBROUTINE neighbor_search(nedges, edge_index, edge_cell_shift)
      INTEGER, INTENT(OUT)                               :: nedges
      INTEGER(kind=int_8), DIMENSION(:, :), &
         INTENT(OUT), OPTIONAL                           :: edge_index
      REAL(sp), DIMENSION(:, :), INTENT(OUT), OPTIONAL   :: edge_cell_shift

      INTEGER:: iatom, jatom
      REAL(dp), DIMENSION(3) :: s1, s2, s12, cell_shift, dx

      nedges = 0
      DO iatom = 1, natoms
         DO jatom = 1, natoms
            IF (iatom == jatom) CYCLE
            s1 = MATMUL(hinv, pos(:, iatom))
            s2 = MATMUL(hinv, pos(:, jatom))
            s12 = s1 - s2
            cell_shift = ANINT(s12)
            dx = MATMUL(cell, s12 - cell_shift)
            IF (DOT_PRODUCT(dx, dx) <= cutoff**2) THEN
               nedges = nedges + 1
               IF (PRESENT(edge_index)) THEN
                  edge_index(nedges, :) = [iatom - 1, jatom - 1]
               END IF
               IF (PRESENT(edge_cell_shift)) THEN
                  edge_cell_shift(:, nedges) = REAL(cell_shift, kind=sp)
               END IF
            END IF
         END DO
      END DO
   END SUBROUTINE neighbor_search

END PROGRAM equipao_unittest
