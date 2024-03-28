import re
import warnings
import torch
import numpy as np
import numpy.typing as npt
import random as rd

from pathlib import Path
from dataclasses import dataclass
from typing import Union, Optional, Dict, List, Tuple

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.math import soft_one_hot_linspace
from e3nn.util.jit import script, trace, compile_mode, compile

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

t = torch.tensor

KindName = str
NDArray = npt.NDArray[np.float64]

prim_basis_specs = {
    "O": "2x0e + 2x1o + 1x2e", # DZVP-MOLOPT-GTH for Oxygen: two s-shells, two p-shells, one d-shell
    "H": "2x0e + 1x1o" # DZVP-MOLOPT-GTH for Hydrogen: two s-shells, one p-shell
}

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']


atomic_numbers = {symbol: Z for Z, symbol in enumerate(chemical_symbols)}
atomic_symbols = {Z: symbol for Z, symbol in enumerate(chemical_symbols)}

# ======================================================================================
@dataclass
class AtomicKind:
    atomic_number: int
    nparams: int = -1
    prim_basis_size: int = -1
    prim_basis_name: str = ""
    pao_basis_size: int = -1


# ======================================================================================
@dataclass
class PaoSample:
    rel_coords: NDArray
    xblock: NDArray


# ======================================================================================
@dataclass
class PAO_Object:
    def __init__(
        self,
        kind: AtomicKind,
        atomkind: torch.Tensor,
        center: torch.Tensor,
        coords: torch.Tensor, 
        xblock: torch.Tensor
    ):
        self.kind = kind
        self.atomkind = atomkind
        self.center = center
        self.coords = coords
        self.xblock = xblock
        U, S, Vh = torch.linalg.svd(xblock, full_matrices=False)
        self.label = Vh


# Torch Module for PAO learning
# ======================================================================================
@compile_mode('trace')
class PAO_model(torch.nn.Module):
    def __init__(self,
                 max_radius: int,
                 num_layers: int,
                 num_neighbours: int,
                 pao_basis_size: int,
                 prim_basis_spec: o3.Irreps,
                 prim_basis_size: int,
                 irreps_input: o3.Irreps,
                 irreps_sh: o3.Irreps,
                 irreps_output: o3.Irreps):
        super().__init__()
        change_of_coord = t([
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.]
        ]) 
        self.dim = prim_basis_size
        self.max_radius = max_radius
        self.num_distances = 10
        self.num_neighbours = num_neighbours
        self.pao_basis_size = pao_basis_size
        self.prim_basis_spec = prim_basis_spec
        self.prim_basis_size = prim_basis_size
        self.irreps_input = irreps_input
        self.irreps_sh = irreps_sh
        self.irreps_output = irreps_output
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        irreps_mid = o3.Irreps([(5, (l, (-1)**l)) for l in range(prim_basis_spec.lmax+1)])
        self.irreps_mid = irreps_mid
        
        #self.lin1 = o3.Linear(irreps_sh, irreps_sh)
        self.tp =  o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_mid, shared_weights=False)
        #self.lin2 = o3.Linear(irreps_mid, irreps_mid)
        self.tp_out = o3.FullyConnectedTensorProduct(irreps_mid, irreps_mid, irreps_output, shared_weights=True)
        
        self.num_layers = num_layers
        self.fc = FullyConnectedNet([self.num_distances, num_layers, self.tp.weight_numel], torch.nn.functional.silu)
        self.coord_change = self.tp_out.irreps_out.D_from_matrix(change_of_coord).to(self.device)

        idx_in = []
        aux_H_idx_in = 2*t(prim_basis_spec.ls)+1 
        self.wigner_dict = {}

        for idx, mu_i in enumerate(prim_basis_spec.ls):
            # Spherical Harmonic Factor 2
            for jdx, mu_j in enumerate(prim_basis_spec.ls[idx:]):
                # Contraction contribution of Spherical Harmonic with L={|L1-L2|...L1+L2}
                for mu_ij in range(abs(mu_i-mu_j),mu_i+mu_j+1):
                    # Check parity (even*even=even, odd*odd=even, even*odd=odd) and generate wigner matrix only for match
                    if mu_i%2==mu_j%2 and mu_ij%2==0 or mu_i%2!=mu_j%2 and mu_ij%2==1:
                        idx_in.append(2*mu_ij+1)
                        wigner_m = o3.wigner_3j(mu_i, mu_j, mu_ij)
                        wig_zero_factor = wigner_m[(2*mu_i+1)//2,(2*mu_j+1)//2,(2*mu_ij+1)//2]
                        wigner_m = wig_zero_factor*wigner_m
                        self.wigner_dict[f"{mu_i}{mu_j}{mu_ij}"] = wigner_m.clone().to(self.device)

        idx_in = t(idx_in)
        self.idx_out = torch.cumsum(idx_in, dim=0, dtype=int)              # rh-index into pred-vector
        self.aux_H_idx_out = torch.cumsum(aux_H_idx_in, dim=0, dtype=int)    # rh-index into auxiliary Hamiltonian
        self.idx_in = self.idx_out-idx_in                                    # lh-index of pred-vector
        self.aux_H_idx_in = self.aux_H_idx_out-aux_H_idx_in                  # lh-index into auxiliary Hamiltonian

    def build_aux_H(self, pred):
        aux_H = torch.zeros((self.prim_basis_spec.dim,self.prim_basis_spec.dim))        # auxiliary Hamiltonian
        jdx_shift = 0

        # Spherical Harmonic Factor 1
        for idx, mu_i in enumerate(self.prim_basis_spec.ls):
            # Spherical Harmonic Factor 2
            for jdx, mu_j in enumerate(self.prim_basis_spec.ls[idx:]):
                # Contraction contribution of Spherical Harmonic with L={|L1-L2|...L1+L2}
                for mu_ij in range(abs(mu_i-mu_j),mu_i+mu_j+1):
                    # Check parity (even*even=even, odd*odd=even, even*odd=odd) and calculate coefficients only for match
                    if mu_i%2==mu_j%2 and mu_ij%2==0 or mu_i%2!=mu_j%2 and mu_ij%2==1:
                        # Contraction per shell
                        contraction_coefficients = torch.matmul(self.wigner_dict[f"{mu_i}{mu_j}{mu_ij}"], pred[self.idx_in[jdx_shift]:self.idx_out[jdx_shift]])
                        # Add shell contribution to resp. block in the auxiliary Hamiltionan
                        aux_H[self.aux_H_idx_in[idx]:self.aux_H_idx_out[idx],self.aux_H_idx_in[jdx+idx]:self.aux_H_idx_out[jdx+idx]] = \
                        aux_H[self.aux_H_idx_in[idx]:self.aux_H_idx_out[idx],self.aux_H_idx_in[jdx+idx]:self.aux_H_idx_out[jdx+idx]]   \
                        + contraction_coefficients 
                        if idx!=idx+jdx: # also add transform of non-diagonal blocks
                            aux_H[self.aux_H_idx_in[jdx+idx]:self.aux_H_idx_out[jdx+idx],self.aux_H_idx_in[idx]:self.aux_H_idx_out[idx]]\
                            = aux_H[self.aux_H_idx_in[jdx+idx]:self.aux_H_idx_out[jdx+idx],self.aux_H_idx_in[idx]:self.aux_H_idx_out[idx]]\
                            + contraction_coefficients.T 
                        jdx_shift += 1
        return aux_H

    def build_aux_H_batch(self, batch, pred):
        aux_H = torch.zeros((batch, self.prim_basis_spec.dim,self.prim_basis_spec.dim), device=self.device)        # auxiliary Hamiltonian
        jdx_shift = 0

        # Spherical Harmonic Factor 1
        for idx, mu_i in enumerate(self.prim_basis_spec.ls):
            # Spherical Harmonic Factor 2
            for jdx, mu_j in enumerate(self.prim_basis_spec.ls[idx:]):
                # Contraction contribution of Spherical Harmonic with L={|L1-L2|...L1+L2}
                for mu_ij in range(abs(mu_i-mu_j),mu_i+mu_j+1):
                    # Check parity (even*even=even, odd*odd=even, even*odd=odd) and calculate coefficients only for match
                    if mu_i%2==mu_j%2 and mu_ij%2==0 or mu_i%2!=mu_j%2 and mu_ij%2==1:
                        # Contraction per shell
                        wigner_m = self.wigner_dict[f"{mu_i}{mu_j}{mu_ij}"]
                        pred_block = pred[:,self.idx_in[jdx_shift]:self.idx_out[jdx_shift]]

                        # If possible, convert this to torch.einsum() notation
                        contraction_coefficients = \
                            torch.transpose(
                                torch.transpose(
                                torch.matmul(wigner_m, torch.transpose(pred_block, dim0=0, dim1=1)),
                                dim0=0, dim1=2),
                            dim0=1, dim1=2)                        
                    
                        # Add shell contribution to resp. block in the auxiliary Hamiltionan
                        aux_H[:,self.aux_H_idx_in[idx]:self.aux_H_idx_out[idx],self.aux_H_idx_in[jdx+idx]:self.aux_H_idx_out[jdx+idx]] = \
                        aux_H[:,self.aux_H_idx_in[idx]:self.aux_H_idx_out[idx],self.aux_H_idx_in[jdx+idx]:self.aux_H_idx_out[jdx+idx]]   \
                        + contraction_coefficients 
                        if idx!=idx+jdx: # also add transform of non-diagonal blocks
                            aux_H[:,self.aux_H_idx_in[jdx+idx]:self.aux_H_idx_out[jdx+idx],self.aux_H_idx_in[idx]:self.aux_H_idx_out[idx]]\
                            = aux_H[:,self.aux_H_idx_in[jdx+idx]:self.aux_H_idx_out[jdx+idx],self.aux_H_idx_in[idx]:self.aux_H_idx_out[idx]]\
                            + torch.transpose(contraction_coefficients, dim0=1, dim1=2) 
                        jdx_shift += 1
        return aux_H

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        edge_vec = data["pos"]
        batch_size = edge_vec.shape[0]
        num_neighbors = edge_vec.shape[1]
        n_edge_vec = edge_vec.shape[0]*edge_vec.shape[1]
        edge_vec = torch.sub(edge_vec, data["x"].unsqueeze(dim=1))
        edge_vec = edge_vec.reshape(n_edge_vec, 3)
        f_in = data["z"].reshape(n_edge_vec, data["z"].shape[-1])
        x = o3.spherical_harmonics(l=self.irreps_sh, x=edge_vec, normalize=True, normalization='component')
        emb = soft_one_hot_linspace(x=edge_vec.norm(dim=1), start=0.0, end=self.max_radius, number=self.num_distances, 
                                    basis='cosine', cutoff=True).mul(self.num_distances**0.5)
        #x = self.lin1(sh)
        aux_H = self.tp(f_in, x, self.fc(emb))
        aux_H = aux_H.reshape((batch_size,aux_H.shape[0]//batch_size,aux_H.shape[1])).sum(dim=1).div(num_neighbors**0.5)
        #aux_H = self.lin2(aux_H)
        aux_H = self.tp_out(aux_H, aux_H)
        aux_H = torch.matmul(aux_H, self.coord_change)
        aux_H = self.build_aux_H_batch(batch_size, aux_H)

        # Eigendecomposition of auxiliary hamiltonian to extract PAO basis vectors from eigenvectors
        L, Q = torch.linalg.eigh(aux_H)
        pao_vectors = torch.transpose(Q[:,:,:self.pao_basis_size], dim0=1, dim1=2)

        data["pao_vectors"] = pao_vectors
        return data

    def _make_tracing_inputs(self, n: int):
        """This is needed to generate pseudo-input for the tracing of the module with TorchScript.
        """
        return [
            {
                'forward': (
                    {
                        "x": torch.rand((3,3), device=self.device),
                        "pos": torch.rand((3,5,3), device=self.device),
                        "y": torch.rand((3,4,5), device=self.device),
                        "z": torch.rand((3,5,2), device=self.device)
                    },
                )
            }
            for _ in range(n)
        ]

# Wrapper for Torch Module for PAO Inference with gradient output
# Heavily based on how output of forces is implemented in Nequip
# ======================================================================================
@compile_mode('script')
class GradOutput(torch.nn.Module):
    def __init__(
        self,
        func: PAO_model,
        of: str,
        wrt: Union[str, List[str]],
        out_field: Optional[List[str]] = None,
        sign: float = 1.0,
    ):
        super().__init__(
        )
        sign = float(sign)
        assert sign in (1.0, -1.0)
        self.sign = sign
        self._negate = sign == -1.0
        self.of = of
        self.skip = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # TO DO: maybe better to force using list?
        if isinstance(wrt, str):
            wrt = [wrt]
        if isinstance(out_field, str):
            out_field = [out_field]
        self.wrt = wrt
        self.func = func
        if out_field is None:
            self.out_field = [f"d({of})/d({e})" for e in self.wrt]
        else:
            assert len(out_field) == len(
                self.wrt
            ), "Out field names must be given for all w.r.t tensors"
            self.out_field = out_field

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        if self.skip:
            return self.func(data)

        # set req grad
        wrt_tensors = []
        old_requires_grad: List[bool] = []
        grad_outputs: List[Optional[torch.Tensor]] = []
        for k in self.wrt:
            old_requires_grad.append(data[k].requires_grad)
            data[k].requires_grad_(True)
            wrt_tensors.append(data[k])

        # run func
        data = self.func(data)

        for k in self.wrt:
            grad_outputs.append(torch.ones_like(data[self.of], device=self.device))
            
        # Get grads
        grads = torch.autograd.grad(
            inputs=wrt_tensors, outputs=[data[self.of],], grad_outputs=grad_outputs, retain_graph=True)
        
        # return
        for out, grad in zip(self.out_field, grads):
            if grad is None:
                # From the docs: "If an output doesn’t require_grad, then the gradient can be None"
                raise RuntimeError("Something is wrong, gradient couldn't be computed")

            if self._negate:
                grad = torch.neg(grad)
            data[out] = grad.detach()

        # unset requires_grad_
        for req_grad, k in zip(old_requires_grad, self.wrt):
            data[k].requires_grad_(req_grad)

        return data       
        
   
# ======================================================================================
def parse_pao_file(
    path: Path,
) -> Tuple[Dict[KindName, AtomicKind], List[KindName], NDArray, List[NDArray]]:
    ikind2name = {}  # maps kind index to kind name
    atom2kind: List[KindName] = []  # maps atom index to kind name
    kinds: Dict[KindName, AtomicKind] = {}
    coords_list = []
    xblocks = []

    for line in path.read_text().strip().split("\n"):
        parts = line.split()
        if parts[0] == "Parametrization":
            assert parts[1] == "EQUIVARIANT"

        elif parts[0] == "Kind":
            ikind = int(parts[1])
            ikind2name[ikind] = parts[2]
            kinds[ikind2name[ikind]] = AtomicKind(atomic_number=int(parts[3]))

        elif parts[0] == "NParams":
            ikind = int(parts[1])
            kinds[ikind2name[ikind]].nparams = int(parts[2])

        elif parts[0] == "PrimBasis":
            ikind = int(parts[1])
            kinds[ikind2name[ikind]].prim_basis_size = int(parts[2])
            kinds[ikind2name[ikind]].prim_basis_name = parts[3]

        elif parts[0] == "PaoBasis":
            ikind = int(parts[1])
            kinds[ikind2name[ikind]].pao_basis_size = int(parts[2])

        elif parts[0] == "Atom":
            atom2kind.append(parts[2])
            coords_list.append(parts[3:])

        elif parts[0] == "Xblock":
            xblocks.append(np.array(parts[2:], float))

    # Convert coordinates to torch tensor.
    coords = np.array(coords_list, float)

    # Reshape xblocks.
    for iatom, kind_name in enumerate(atom2kind):
        n = kinds[kind_name].prim_basis_size
        m = kinds[kind_name].pao_basis_size
        xblocks[iatom] = xblocks[iatom].reshape(m, n)

    return kinds, atom2kind, coords, xblocks


# ======================================================================================
def parse_pao_file_torch(path: Path):
    kinds, atom2kind, coords, xblocks = parse_pao_file(path)
    return kinds, atom2kind, t(coords, dtype=torch.float32), [t(x, dtype=torch.float32) for x in xblocks]


# ======================================================================================
def append_samples(
    samples: Dict[KindName, List[PaoSample]],
    kinds: Dict[KindName, AtomicKind],
    atom2kind: List[KindName],
    coords: NDArray,
    xblocks: List[NDArray],
) -> None:
    for iatom, kind_name in enumerate(atom2kind):
        rel_coords = coords - coords[iatom, :]
        sample = PaoSample(rel_coords=rel_coords, xblock=xblocks[iatom])
        if kind_name not in samples:
            samples[kind_name] = []
        samples[kind_name].append(sample)


# ======================================================================================
def write_pao_file(
    path: Path,
    kinds: Dict[KindName, AtomicKind],
    atom2kind: List[KindName],
    coords: NDArray,
    xblocks: List[NDArray],
) -> None:

    natoms = coords.shape[0]
    assert coords.shape[1] == 3
    assert len(xblocks) == natoms

    output = []
    output.append("Version 4")
    output.append("Parametrization EQUIVARIANT")
    output.append(f"Nkinds {len(kinds)}")
    for ikind, (kind_name, kind) in enumerate(kinds.items()):
        i = ikind + 1
        output.append(f"Kind {i} {kind_name} {kind.atomic_number}")
        output.append(f"NParams {i} {kind.nparams}")
        output.append(f"PrimBasis {i} {kind.prim_basis_size} {kind.prim_basis_name}")
        output.append(f"PaoBasis {i} {kind.pao_basis_size}")
        output.append(f"NPaoPotentials {i} 0")
    output.append("Cell 8.0 0.0 0.0   0.0 8.0 0.0   0.0 0.0 8.0")
    output.append(f"Natoms {natoms}")

    for iatom in range(natoms):
        c = coords[iatom, :]
        output.append(f"Atom {iatom + 1} {atom2kind[iatom]} {c[0]} {c[1]} {c[2]}")

    for iatom in range(natoms):
        kind = kinds[atom2kind[iatom]]
        assert len(xblocks[iatom].shape) == 2
        assert xblocks[iatom].shape[0] == kind.pao_basis_size
        assert xblocks[iatom].shape[1] == kind.prim_basis_size
        x = xblocks[iatom].flatten()
        y = " ".join(["%f" % i for i in x])
        output.append(f"Xblock {iatom + 1} {y}")

    output.append("THE_END")
    path.write_text("\n".join(output))


# ======================================================================================
def read_cp2k_energy(path: Path) -> float:
    try:
        content = path.read_text()
        m = re.search(r"ENERGY\|(.*)", content)
        assert m
        return float(m.group(1).split()[-1])
    except:
        print(f"error with: {path}")
    return float("NaN")


# ======================================================================================
# TO DO: Generate Irreps representation of the primitive basis set
def convert_basis_name_to_irreps(
    ) -> None:
    pass


# ======================================================================================
def generate_f_in(atomkinds: Dict[KindName, AtomicKind], atomkind: List[KindName]) -> torch.Tensor:
    r"""One-hot encoding of the atomtype
    """  
    f_in = t([[k==j for j in atomkinds] for k in atomkind], dtype=torch.float32)
    return f_in


# ======================================================================================
def pao_objects_from_file(file_path: Path) -> List[PAO_Object]:
    r"""PAO objects from CP2K PAO file
    """
    pao_objects = []
    kinds, atom2kind, coords, xblocks = parse_pao_file_torch(file_path)
    for idx, atom in enumerate(atom2kind):
        idxs = list(range(len(atom2kind)))
        idxs.pop(idx)
        f_in = generate_f_in(kinds, atom2kind)
        pao_objects.append(PAO_Object(kinds[atom], f_in[idxs], coords[idx], coords[idxs], xblocks[idx]))
    return pao_objects


# ======================================================================================
def irreps_output_from_prim_basis(prim_basis_specs: o3.Irreps) -> o3.Irreps:
    r"""Irreducible representations required to build the auxiliary Hamiltonian based on the composition of the primitive basis set"""
    all_ir = o3.Irreps()
    for idx, l1 in enumerate(prim_basis_specs.ls):
        irrep1=o3.Irrep(l1, (-1)**l1)
        for l2 in prim_basis_specs.ls[idx:]:
            irrep2=o3.Irrep(l2, (-1)**l2)
            mul = irrep1*irrep2
            for ir in mul:
                if ir.l%2==0 and ir.p==1 or ir.l%2==1 and ir.p==-1:
                    all_ir += ir
    return all_ir.simplify()


# ======================================================================================
def irreps_input_from_pao_object(pao_object: PAO_Object) -> o3.Irreps:
    r"""Irreps of the one-hot atomtype encoding of the form Nx0e where N is the number of unique atomtypes in the system.
    """
    irreps_input = o3.Irreps(f"{pao_object.atomkind.shape[1]}x0e")
    return irreps_input


# ======================================================================================
def loss_function_ortho_projector_batch(pred, label):
    r"""Loss function for batch learning based on projection matrices of two sets of vectors.
    """
    pred_projector = torch.bmm(torch.transpose(pred, dim0=1, dim1=2), pred)
    label_projector = torch.bmm(torch.transpose(label, dim0=1, dim1=2), label)
    residual = pred_projector - label_projector
    return residual.pow(2).mean()

def loss_function_ortho_projector(pred, label):
    r"""Loss function based on projection matrices of two sets of vectors.
    """
    pred_projector = pred.T @ pred
    label_projector = label.T @ label
    proj_residual = pred_projector - label_projector
    return proj_residual.pow(2).mean()


# ======================================================================================
def pao_objects_from_paths(paths: List[Path]) -> Dict[KindName, List[PAO_Object]]:
    r"""Create PAO objects from PAO files at provided paths.
    """
    pao_objects = {}
    for path in paths:
        temp_pao_objects = pao_objects_from_file(path)
        for pao_object in temp_pao_objects:
            kind = atomic_symbols[pao_object.kind.atomic_number]
            if kind not in pao_objects:
                pao_objects[kind] = [pao_object]
            else:
                pao_objects[kind].append(pao_object)
    return pao_objects


# ======================================================================================
def generate_train_test_datasets(pao_objects: Dict[KindName, List[PAO_Object]], test_size=0.2):
    r"""Create training and testing datasets from PAO objects.
    """
    datasets = {}
    for atomtype in pao_objects:
        dataset = [Data(
            x=pao_object.center,
            pos=pao_object.coords,
            y=pao_object.label,
            z=pao_object.atomkind) for pao_object in pao_objects[atomtype]]
        train_data, test_data = train_test_split(dataset, test_size=test_size)
        datasets[atomtype] = {
            "train": train_data.copy(),
            "test": test_data.copy()
            }
    return datasets


# ======================================================================================
def generate_train_test_data(pao_objects, test_size=0.2):
    r"""Create training and testing data dictionaries from PAO objects.
    """
    datadicts = {}
    for atomtype in pao_objects:
        train_data, test_data = train_test_split(pao_objects[atomtype], test_size=test_size)
        datadicts[atomtype] = {
            "train": train_data,
            "test": test_data
        }
    return datadicts


# ======================================================================================
def generate_batched_dict(pao_objects, batch_size):
    n_samples = len(pao_objects)
    idxs = list(range(n_samples))
    rd.shuffle(idxs)
    batches = n_samples//batch_size
    batched_datadicts = []
    for batch in range(batches):
        batch_idxs = idxs[batch*batch_size:(batch+1)*batch_size]
        batch_dict = {
                "x": torch.stack([pao_objects[idx].center for idx in batch_idxs]),
                "pos": torch.stack([pao_objects[idx].coords for idx in batch_idxs]),
                "y": torch.stack([pao_objects[idx].label for idx in batch_idxs]),
                "z": torch.stack([pao_objects[idx].atomkind for idx in batch_idxs])
        }
        batched_datadicts.append(batch_dict)
    return batched_datadicts


# ======================================================================================
def generate_train_dataloader(datasets: Dict[KindName, List[PAO_Object]], batch_size=32):
    r"""Create dataloaders for training and validation from datasets.
    """
    dataloaders = {}
    vdataloaders = {}
    for atomtype in datasets:
        dataloaders[atomtype] = DataLoader(datasets[atomtype]["train"], batch_size=batch_size, shuffle=True)
        vdataloaders[atomtype] = DataLoader(datasets[atomtype]["test"], batch_size=batch_size, shuffle=True)
    return dataloaders, vdataloaders


# ======================================================================================
def init_pao_models(pao_objects: Dict[KindName, List[PAO_Object]], cutoff, num_neighbors, num_layers=32, require_grad=True):
    r"""Initialize PAO models for each atomtype.
    """
    models = {}
    for atomtype in pao_objects:
        irreps_input = irreps_input_from_pao_object(pao_objects[atomtype][0])
        pao_basis_size = pao_objects[atomtype][0].kind.pao_basis_size
        prim_basis_spec = o3.Irreps(prim_basis_specs[atomtype])
        prim_basis_size = prim_basis_spec.dim
        irreps_sh = o3.Irreps.spherical_harmonics(lmax=prim_basis_spec.lmax)
        irreps_output = irreps_output_from_prim_basis(prim_basis_spec)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            models[atomtype] = PAO_model(
            max_radius=cutoff,
            num_layers=num_layers,
            num_neighbours=num_neighbors,
            pao_basis_size=pao_basis_size,
            prim_basis_spec=prim_basis_spec,
            prim_basis_size=prim_basis_size, 
            irreps_input=irreps_input,
            irreps_sh=irreps_sh,
            irreps_output=irreps_output,
            )
            if require_grad:
                gradModel = GradOutput(func=models[atomtype], of="pao_vectors", wrt=["x"], out_field=["gradient"], sign=-1)
                models[atomtype] = gradModel
            
    return models


# ======================================================================================
def train_model_epoch(model, optimizer, pao_objects, batch_size, batch_loss_average=10):
    r"""Train PAO model for one epoch.
    """
    running_loss = 0.
    last_loss = 0.

    batched_dicts = generate_batched_dict(pao_objects, batch_size)
    for i, data in enumerate(batched_dicts):
        for key, value in data.items():
            data[key] = data[key].to(model.device)
        data = model(data)
        optimizer.zero_grad()
        loss = loss_function_ortho_projector_batch(data["pao_vectors"], data["y"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % batch_loss_average == batch_loss_average-1:
            last_loss = running_loss / batch_loss_average
            running_loss = 0.
    
    return last_loss


# ======================================================================================
def validate_model(model, pao_objects, batch_size):
    r"""Validate PAO model.
    """
    running_vloss = 0.

    batched_dicts = generate_batched_dict(pao_objects, batch_size)
    for i, vdata in enumerate(batched_dicts):
        for key, value in vdata.items():
            vdata[key] = vdata[key].to(model.device)
        vdata = model(vdata)
        vloss = loss_function_ortho_projector_batch(vdata["pao_vectors"], vdata["y"])
        running_vloss += vloss.item()

    avg_vloss = running_vloss/(i+1)
    return avg_vloss


# ======================================================================================
def train_model(model, optimizer, num_epochs, train_data, test_data, batch_size, batch_loss_average=10):
    r"""Full training of PAO model.
    """
    train_loss = torch.zeros(num_epochs)
    validation_loss = torch.zeros(num_epochs)
    model.to(model.device)

    for epoch in range(num_epochs):
        # === Train ===
        model.train(True)
        train_loss[epoch] = train_model_epoch(model, optimizer, train_data, batch_size, batch_loss_average)
        # === Validate ===
        model.eval()
        validation_loss[epoch] = validate_model(model, test_data, batch_size)

        print(f"training epoch {epoch:3d} | loss {train_loss[epoch]:.8e} | validation loss {validation_loss[epoch]:.8e}")
    return train_loss, validation_loss


# ======================================================================================
def save_model(model, model_name):
    r"""Compile and save PAO model.
    """
    script_model = compile(model, trace_options={"strict": False})
    script_model.save(f"{model_name}.pth")
    return


# ======================================================================================
def plot_loss(train_loss, validation_loss, atomtype):
    r"""Plot and save training and validation loss of the training procedure.
    """
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(range(len(train_loss)), train_loss, label="training")
    ax.plot(range(len(validation_loss)), validation_loss, label="validation")
    ax.set_yscale("log")
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax.legend()
    plt.savefig(f"loss_plot_{atomtype}.png")
    return


# ======================================================================================
def main():
    # TO DO: Input for training settings via seperate input file
    paths = []
    atomtypes = ["O", "H"]
    print(f"Running Equi PAO training for atoms {atomtypes}.")
    test_ratio = 0.2
    batch_size = 16
    for path in sorted(Path().glob("training_data/*/*-1_0.pao")):
        paths.append(path)

    pao_objects = pao_objects_from_paths(paths)
    print("Generating PAO objects from provided paths.")
    datasets = generate_train_test_data(pao_objects, 0.2)
    print(f"Data split into training and validation data with test ratio of {test_ratio:.2e}")

    cutoff = 4.0
    num_neighbors= 5
    learning_rate = 1e-3
    train_loss = {}
    validation_loss = {}
    epochs = {"O": 1000,
              "H": 500}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}.")

    models = init_pao_models(pao_objects, cutoff, num_neighbors, num_layers=32)
    for atomtype in atomtypes:
        print(f"Training Equi PAO Model for atom {atomtype} for {epochs} epochs.")
        batch_loss_average = len(datasets[atomtype]["train"])//batch_size
        optim = torch.optim.Adam(models[atomtype].parameters(), lr=learning_rate)
        train_loss[atomtype], validation_loss[atomtype] = train_model(
            models[atomtype],
            optim,
            epochs[atomtype],
            datasets[atomtype]["train"],
            datasets[atomtype]["test"],
            batch_size,
            batch_loss_average
            )
        print(f"Finished training Equi PAO Model for atom {atomtype}.")
        plot_loss(train_loss[atomtype], validation_loss[atomtype], atomtype)
        save_model(models[atomtype], f"equi_pao_model_{atomtype}")
        print(f"Saved Equi PAO Model for atom {atomtype}.")
        

if __name__ == "__main__":
    main()


# EOF
