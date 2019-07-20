PROGRAM print_clebsch_gordan_real

IMPLICIT NONE


   INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(14, 200)
   REAL(KIND=dp), DIMENSION(:, :, :), ALLOCATABLE :: cg_table
   INTEGER :: lmax = -1
   REAL(KIND=dp) :: osq2, sq2

   ! Factorial function fac
   ! Inverse Factorial function ifac
   ! Double factorial function dfac
   ! Gamma functions
   ! gamma(n) = gamma0(n) = (n - 1)!
   ! gamma(n + 1/2) = gamma1(n) = SQRT(pi)/2^n (2n - 1)!!

   INTEGER, PARAMETER :: maxfac = 30
   REAL(KIND=dp), PARAMETER, DIMENSION(0:maxfac) :: fac = (/ &
                                      0.10000000000000000000E+01_dp, 0.10000000000000000000E+01_dp, 0.20000000000000000000E+01_dp, &
                                      0.60000000000000000000E+01_dp, 0.24000000000000000000E+02_dp, 0.12000000000000000000E+03_dp, &
                                      0.72000000000000000000E+03_dp, 0.50400000000000000000E+04_dp, 0.40320000000000000000E+05_dp, &
                                      0.36288000000000000000E+06_dp, 0.36288000000000000000E+07_dp, 0.39916800000000000000E+08_dp, &
                                      0.47900160000000000000E+09_dp, 0.62270208000000000000E+10_dp, 0.87178291200000000000E+11_dp, &
                                      0.13076743680000000000E+13_dp, 0.20922789888000000000E+14_dp, 0.35568742809600000000E+15_dp, &
                                      0.64023737057280000000E+16_dp, 0.12164510040883200000E+18_dp, 0.24329020081766400000E+19_dp, &
                                      0.51090942171709440000E+20_dp, 0.11240007277776076800E+22_dp, 0.25852016738884976640E+23_dp, &
                                      0.62044840173323943936E+24_dp, 0.15511210043330985984E+26_dp, 0.40329146112660563558E+27_dp, &
                                      0.10888869450418352161E+29_dp, 0.30488834461171386050E+30_dp, 0.88417619937397019545E+31_dp, &
                                                    0.26525285981219105864E+33_dp/)
   REAL(KIND=dp), PARAMETER, DIMENSION(0:maxfac) :: ifac = (/ &
                                      0.10000000000000000000E+01_dp, 0.10000000000000000000E+01_dp, 0.50000000000000000000E+00_dp, &
                                      0.16666666666666666667E+00_dp, 0.41666666666666666667E-01_dp, 0.83333333333333333333E-02_dp, &
                                      0.13888888888888888889E-02_dp, 0.19841269841269841270E-03_dp, 0.24801587301587301587E-04_dp, &
                                      0.27557319223985890653E-05_dp, 0.27557319223985890653E-06_dp, 0.25052108385441718775E-07_dp, &
                                      0.20876756987868098979E-08_dp, 0.16059043836821614599E-09_dp, 0.11470745597729724714E-10_dp, &
                                      0.76471637318198164759E-12_dp, 0.47794773323873852974E-13_dp, 0.28114572543455207632E-14_dp, &
                                      0.15619206968586226462E-15_dp, 0.82206352466243297170E-17_dp, 0.41103176233121648585E-18_dp, &
                                      0.19572941063391261231E-19_dp, 0.88967913924505732867E-21_dp, 0.38681701706306840377E-22_dp, &
                                      0.16117375710961183490E-23_dp, 0.64469502843844733962E-25_dp, 0.24795962632247974601E-26_dp, &
                                      0.91836898637955461484E-28_dp, 0.32798892370698379102E-29_dp, 0.11309962886447716932E-30_dp, &
                                                    0.37699876288159056439E-32_dp/)
   REAL(KIND=dp), PARAMETER, DIMENSION(-1:2*maxfac+1) :: dfac = (/ &
                                      0.10000000000000000000E+01_dp, 0.10000000000000000000E+01_dp, 0.10000000000000000000E+01_dp, &
                                      0.20000000000000000000E+01_dp, 0.30000000000000000000E+01_dp, 0.80000000000000000000E+01_dp, &
                                      0.15000000000000000000E+02_dp, 0.48000000000000000000E+02_dp, 0.10500000000000000000E+03_dp, &
                                      0.38400000000000000000E+03_dp, 0.94500000000000000000E+03_dp, 0.38400000000000000000E+04_dp, &
                                      0.10395000000000000000E+05_dp, 0.46080000000000000000E+05_dp, 0.13513500000000000000E+06_dp, &
                                      0.64512000000000000000E+06_dp, 0.20270250000000000000E+07_dp, 0.10321920000000000000E+08_dp, &
                                      0.34459425000000000000E+08_dp, 0.18579456000000000000E+09_dp, 0.65472907500000000000E+09_dp, &
                                      0.37158912000000000000E+10_dp, 0.13749310575000000000E+11_dp, 0.81749606400000000000E+11_dp, &
                                      0.31623414322500000000E+12_dp, 0.19619905536000000000E+13_dp, 0.79058535806250000000E+13_dp, &
                                      0.51011754393600000000E+14_dp, 0.21345804667687500000E+15_dp, 0.14283291230208000000E+16_dp, &
                                      0.61902833536293750000E+16_dp, 0.42849873690624000000E+17_dp, 0.19189878396251062500E+18_dp, &
                                      0.13711959580999680000E+19_dp, 0.63326598707628506250E+19_dp, 0.46620662575398912000E+20_dp, &
                                      0.22164309547669977187E+21_dp, 0.16783438527143608320E+22_dp, 0.82007945326378915594E+22_dp, &
                                      0.63777066403145711616E+23_dp, 0.31983098677287777082E+24_dp, 0.25510826561258284646E+25_dp, &
                                      0.13113070457687988603E+26_dp, 0.10714547155728479551E+27_dp, 0.56386202968058350995E+27_dp, &
                                      0.47144007485205310027E+28_dp, 0.25373791335626257948E+29_dp, 0.21686243443194442612E+30_dp, &
                                      0.11925681927744341235E+31_dp, 0.10409396852733332454E+32_dp, 0.58435841445947272053E+32_dp, &
                                      0.52046984263666662269E+33_dp, 0.29802279137433108747E+34_dp, 0.27064431817106664380E+35_dp, &
                                      0.15795207942839547636E+36_dp, 0.14614793181237598765E+37_dp, 0.86873643685617511998E+37_dp, &
                                      0.81842841814930553085E+38_dp, 0.49517976900801981839E+39_dp, 0.47468848252659720789E+40_dp, &
                                       0.29215606371473169285E+41_dp, 0.28481308951595832474E+42_dp, 0.17821519886598633264E+43_dp/)
   REAL(KIND=dp), PARAMETER, DIMENSION(0:maxfac) :: gamma0 = (/ &
                                      0.00000000000000000000E+00_dp, 0.10000000000000000000E+01_dp, 0.10000000000000000000E+01_dp, &
                                      0.20000000000000000000E+01_dp, 0.60000000000000000000E+01_dp, 0.24000000000000000000E+02_dp, &
                                      0.12000000000000000000E+03_dp, 0.72000000000000000000E+03_dp, 0.50400000000000000000E+04_dp, &
                                      0.40320000000000000000E+05_dp, 0.36288000000000000000E+06_dp, 0.36288000000000000000E+07_dp, &
                                      0.39916800000000000000E+08_dp, 0.47900160000000000000E+09_dp, 0.62270208000000000000E+10_dp, &
                                      0.87178291200000000000E+11_dp, 0.13076743680000000000E+13_dp, 0.20922789888000000000E+14_dp, &
                                      0.35568742809600000000E+15_dp, 0.64023737057280000000E+16_dp, 0.12164510040883200000E+18_dp, &
                                      0.24329020081766400000E+19_dp, 0.51090942171709440000E+20_dp, 0.11240007277776076800E+22_dp, &
                                      0.25852016738884976640E+23_dp, 0.62044840173323943936E+24_dp, 0.15511210043330985984E+26_dp, &
                                      0.40329146112660563558E+27_dp, 0.10888869450418352161E+29_dp, 0.30488834461171386050E+30_dp, &
                                                    0.88417619937397019545E+31_dp/)
   REAL(KIND=dp), PARAMETER, DIMENSION(0:maxfac) :: gamma1 = (/ &
                                      0.17724538509055160273E+01_dp, 0.88622692545275801365E+00_dp, 0.13293403881791370205E+01_dp, &
                                      0.33233509704478425512E+01_dp, 0.11631728396567448929E+02_dp, 0.52342777784553520181E+02_dp, &
                                      0.28788527781504436100E+03_dp, 0.18712543057977883465E+04_dp, 0.14034407293483412599E+05_dp, &
                                      0.11929246199460900709E+06_dp, 0.11332783889487855673E+07_dp, 0.11899423083962248457E+08_dp, &
                                      0.13684336546556585726E+09_dp, 0.17105420683195732157E+10_dp, 0.23092317922314238412E+11_dp, &
                                      0.33483860987355645697E+12_dp, 0.51899984530401250831E+13_dp, 0.85634974475162063871E+14_dp, &
                                      0.14986120533153361177E+16_dp, 0.27724322986333718178E+17_dp, 0.54062429823350750447E+18_dp, &
                                      0.11082798113786903842E+20_dp, 0.23828015944641843260E+21_dp, 0.53613035875444147334E+22_dp, &
                                      0.12599063430729374624E+24_dp, 0.30867705405286967828E+25_dp, 0.78712648783481767961E+26_dp, &
                                      0.20858851927622668510E+28_dp, 0.57361842800962338401E+29_dp, 0.16348125198274266444E+31_dp, &
                                                    0.48226969334909086011E+32_dp/)

   ! Constants related to Pi

   REAL(KIND=dp), PARAMETER :: pi = 3.14159265358979323846264338_dp ! Pi


CALL clebsch_gordon_init(5)
CALL print_clebsch_gordon(l1=1, l2=1, l3=2)
CALL clebsch_gordon_deallocate()

CONTAINS

! **************************************************************************************************
! **************************************************************************************************
   SUBROUTINE print_clebsch_gordon(l1, l2, l3)
      INTEGER, INTENT(IN)                                :: l1, l2, l3
      REAL(KIND=dp)                                      :: rga(10, 2), cga(10)
      INTEGER                                            :: m1, m2, m3, l3_idx

      !https://github.com/cp2k/cp2k/blob/master/src/common/cg_test.F

      l3_idx = l3/2+1

      DO m1 = -l1, l1
      DO m2 = -l2, l2
         !CALL clebsch_gordon_real(l1, m1, l2, m2, rga)
         CALL clebsch_gordon_complex(l1, m1, l2, m2, cga)
         DO m3 = -l3, l3
           !WRITE (*,*) "m1", m1, "m2", m2, "m3", m3, "rga", rga(l3_idx,:)
           WRITE (*,*) "m1", m1, "m2", m2, "m3", m3, "cga", cga(l3_idx)
         END DO
      END DO
      END DO
   END SUBROUTINE print_clebsch_gordon

   ! **************************************************************************************************
!> \brief ...
!> \param l1 ...
!> \param m1 ...
!> \param l2 ...
!> \param m2 ...
!> \param clm ...
! **************************************************************************************************
   SUBROUTINE clebsch_gordon_complex(l1, m1, l2, m2, clm)
      INTEGER, INTENT(IN)                                :: l1, m1, l2, m2
      REAL(KIND=dp), DIMENSION(:), INTENT(OUT)           :: clm

      INTEGER                                            :: icase, ind, l, lm, lp, n

      l = l1+l2
      IF (l > lmax) CALL clebsch_gordon_init(l)
      n = l/2+1
      IF (n > SIZE(clm)) STOP "Array too small"

      IF ((m1 >= 0 .AND. m2 >= 0) .OR. (m1 < 0 .AND. m2 < 0)) THEN
         icase = 1
      ELSE
         icase = 2
      END IF
      ind = order(l1, m1, l2, m2)

      DO lp = MOD(l, 2), l, 2
         lm = lp/2+1
         clm(lm) = cg_table(ind, lm, icase)
      END DO

   END SUBROUTINE clebsch_gordon_complex

! **************************************************************************************************
!> \brief ...
!> \param l1 ...
!> \param m1 ...
!> \param l2 ...
!> \param m2 ...
!> \param rlm ...
! **************************************************************************************************
   SUBROUTINE clebsch_gordon_real(l1, m1, l2, m2, rlm)
      INTEGER, INTENT(IN)                                :: l1, m1, l2, m2
      REAL(KIND=dp), DIMENSION(:, :), INTENT(OUT)        :: rlm

      INTEGER                                            :: icase1, icase2, ind, l, lm, lp, mm(2), n
      REAL(KIND=dp)                                      :: xsi

      l = l1+l2
      IF (l > lmax) CALL clebsch_gordon_init(l)
      n = l/2+1
      IF (n > SIZE(rlm, 1)) STOP "Array too small"

      ind = order(l1, m1, l2, m2)
      mm = getm(m1, m2)
      IF ((m1 >= 0 .AND. m2 >= 0) .OR. (m1 < 0 .AND. m2 < 0)) THEN
         icase1 = 1
         icase2 = 2
      ELSE
         icase1 = 2
         icase2 = 1
      END IF

      DO lp = MOD(l, 2), l, 2
         lm = lp/2+1
         xsi = get_factor(m1, m2, mm(1))
         rlm(lm, 1) = xsi*cg_table(ind, lm, icase1)
         xsi = get_factor(m1, m2, mm(2))
         rlm(lm, 2) = xsi*cg_table(ind, lm, icase2)
      END DO

   END SUBROUTINE clebsch_gordon_real


! **************************************************************************************************
!> \brief ...
!> \param m1 ...
!> \param m2 ...
!> \return ...
! **************************************************************************************************
   FUNCTION getm(m1, m2) RESULT(m)
      INTEGER, INTENT(IN)                                :: m1, m2
      INTEGER, DIMENSION(2)                              :: m

      INTEGER                                            :: mm, mp

      mp = m1+m2
      mm = m1-m2
      IF (m1*m2 < 0 .OR. (m1*m2 == 0 .AND. (m1 < 0 .OR. m2 < 0))) THEN
         mp = -ABS(mp)
         mm = -ABS(mm)
      ELSE
         mp = ABS(mp)
         mm = ABS(mm)
      END IF
      m(1) = mp
      m(2) = mm
   END FUNCTION getm

! **************************************************************************************************
!> \brief ...
!> \param m1 ...
!> \param m2 ...
!> \param m ...
!> \return ...
! **************************************************************************************************
   FUNCTION get_factor(m1, m2, m) RESULT(f)
      INTEGER, INTENT(IN)                                :: m1, m2, m
      REAL(KIND=dp)                                      :: f

      INTEGER                                            :: mx, my

      f = 1.0_dp
      IF (ABS(m1) >= ABS(m2)) THEN
         mx = m1
         my = m2
      ELSE
         mx = m2
         my = m1
      ENDIF
      IF (mx*my == 0) THEN
         f = 1.0_dp
      ELSE IF (m == 0) THEN
         IF (ABS(mx) /= ABS(my)) WRITE (*, '(A,3I6)') " 1) Illegal Case ", m1, m2, m
         IF (mx*my > 0) THEN
            f = 1.0_dp
         ELSE
            f = 0.0_dp
         END IF
      ELSE IF (ABS(mx)+ABS(my) == m) THEN
         f = osq2
         IF (mx < 0) f = -osq2
      ELSE IF (ABS(mx)+ABS(my) == -m) THEN
         f = osq2
      ELSE IF (ABS(mx)-ABS(my) == -m) THEN
         IF (mx*my > 0) WRITE (*, '(A,3I6)') " 2) Illegal Case ", m1, m2, m
         IF (mx > 0) f = -osq2
         IF (mx < 0) f = osq2
      ELSE IF (ABS(mx)-ABS(my) == m) THEN
         IF (mx*my < 0) WRITE (*, '(A,3I6)') " 3) Illegal Case ", m1, m2, m
         f = osq2
      ELSE
         WRITE (*, '(A,3I6)') " 4) Illegal Case ", m1, m2, m
      END IF
   END FUNCTION get_factor

! **************************************************************************************************
!> \brief ...
! **************************************************************************************************
   SUBROUTINE clebsch_gordon_deallocate()
      IF (ALLOCATED(cg_table)) THEN
         DEALLOCATE (cg_table)
      END IF
   END SUBROUTINE clebsch_gordon_deallocate

! **************************************************************************************************
!> \brief ...
!> \param l ...
! **************************************************************************************************
   SUBROUTINE clebsch_gordon_init(l)
      INTEGER, INTENT(IN)                                :: l

      INTEGER                                            :: i1, i2, ix, iy, l1, l2, lp, m, &
                                                            m1, m2, ml, mp, n


      sq2 = SQRT(2.0_dp)
      osq2 = 1.0_dp/sq2

      IF (l < 0) STOP "l < 0"
      IF (ALLOCATED(cg_table)) THEN
         DEALLOCATE (cg_table)
      END IF
      ! maximum size of table
      n = (l**4+6*l**3+15*l**2+18*l+8)/8
      m = l+1
      ALLOCATE (cg_table(n, m, 2))
      lmax = l

      DO l1 = 0, lmax
         DO m1 = 0, l1
            iy = (l1*(l1+1))/2+m1+1
            DO l2 = l1, lmax
               ml = 0
               IF (l1 == l2) ml = m1
               DO m2 = ml, l2
                  ix = (l2*(l2+1))/2+m2+1
                  i1 = (ix*(ix-1))/2+iy
                  DO lp = MOD(l1+l2, 2), l1+l2, 2
                     i2 = lp/2+1
                     mp = m2+m1
                     cg_table(i1, i2, 1) = cgc(l1, m1, l2, m2, lp, mp)
                     mp = ABS(m2-m1)
                     IF (m2 >= m1) THEN
                        cg_table(i1, i2, 2) = cgc(l1, m1, lp, mp, l2, m2)
                     ELSE
                        cg_table(i1, i2, 2) = cgc(l2, m2, lp, mp, l1, m1)
                     END IF
                  END DO
               END DO
            END DO
         END DO
      END DO


   END SUBROUTINE clebsch_gordon_init

! **************************************************************************************************
!> \brief ...
!> \param l1 ...
!> \param m1 ...
!> \param l2 ...
!> \param m2 ...
!> \param lp ...
!> \param mp ...
!> \return ...
! **************************************************************************************************
   FUNCTION cgc(l1, m1, l2, m2, lp, mp)
      INTEGER, INTENT(IN)                                :: l1, m1, l2, m2, lp, mp
      REAL(KIND=dp)                                      :: cgc


      INTEGER                                            :: la, lb, ll, ma, mb, s, t, tmax, tmin, &
                                                            z1, z2, z3, z4, z5
      REAL(KIND=dp)                                      :: f1, f2, pref

! m1 >= 0; m2 >= 0; mp >= 0

      IF (m1 < 0 .OR. m2 < 0 .OR. mp < 0) THEN
         WRITE (*, *) l1, l2, lp
         WRITE (*, *) m1, m2, mp
         STOP "Illegal input values"
      END IF
      IF (l2 < l1) THEN
         la = l2
         ma = m2
         lb = l1
         mb = m1
      ELSE
         la = l1
         ma = m1
         lb = l2
         mb = m2
      END IF

      IF (MOD(la+lb+lp, 2) == 0 .AND. la+lb >= lp .AND. lp >= lb-la &
          .AND. lb-mb >= 0) THEN
         ll = (2*lp+1)*(2*la+1)*(2*lb+1)
         pref = 1.0_dp/SQRT(4.0_dp*pi)*0.5_dp*SQRT(REAL(ll, dp)* &
                                                   (sfac(lp-mp)/sfac(lp+mp))* &
                                                   (sfac(la-ma)/sfac(la+ma))*(sfac(lb-mb)/sfac(lb+mb)))
         s = (la+lb+lp)/2
         tmin = MAX(0, -lb+la-mp)
         tmax = MIN(lb+la-mp, lp-mp, la-ma)
         f1 = REAL(2*(-1)**(s-lb-ma), KIND=dp)*(sfac(lb+mb)/sfac(lb-mb))* &
              sfac(la+ma)/(sfac(s-lp)*sfac(s-lb))*sfac(2*s-2*la)/sfac(s-la)* &
              (sfac(s)/sfac(2*s+1))
         f2 = 0.0_dp
         DO t = tmin, tmax
            z1 = lp+mp+t
            z2 = la+lb-mp-t
            z3 = lp-mp-t
            z4 = lb-la+mp+t
            z5 = la-ma-t
            f2 = f2+(-1)**t*(sfac(z1)/(sfac(t)*sfac(z3)))*(sfac(z2)/(sfac(z4)*sfac(z5)))
         END DO
         cgc = pref*f1*f2
      ELSE
         cgc = 0.0_dp
      END IF

   END FUNCTION cgc

! **************************************************************************************************
!> \brief ...
!> \param n ...
!> \return ...
! **************************************************************************************************
   FUNCTION sfac(n) RESULT(fval)
      INTEGER                                            :: n
      REAL(KIND=dp)                                      :: fval

      INTEGER                                            :: i

      IF (n > maxfac) THEN
         fval = fac(maxfac)
         DO i = maxfac+1, n
            fval = REAL(i, dp)*fval
         END DO
      ELSE IF (n >= 0) THEN
         fval = fac(n)
      ELSE
         fval = 1.0_dp
      END IF
   END FUNCTION sfac

! **************************************************************************************************
!> \brief ...
!> \param l1 ...
!> \param m1 ...
!> \param l2 ...
!> \param m2 ...
!> \return ...
! **************************************************************************************************
   FUNCTION order(l1, m1, l2, m2) RESULT(ind)
      INTEGER, INTENT(IN)                                :: l1, m1, l2, m2
      INTEGER                                            :: ind

      INTEGER                                            :: i1, i2, ix, iy

      i1 = (l1*(l1+1))/2+ABS(m1)+1
      i2 = (l2*(l2+1))/2+ABS(m2)+1
      ix = MAX(i1, i2)
      iy = MIN(i1, i2)
      ind = (ix*(ix-1))/2+iy
   END FUNCTION order

END PROGRAM print_clebsch_gordan_real