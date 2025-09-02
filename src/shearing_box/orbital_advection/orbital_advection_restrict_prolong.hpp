#ifndef ORBTIAL_ADVECTION_RESTRICT_PROLONG_HPP_
#define ORBTIAL_ADVECTION_RESTRICT_PROLONG_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file restrict_prolong.hpp
//! \brief prolongation and restrict operators for cell-centered and face-centered variables,

// #include "athena.hpp"



//----------------------------------------------------------------------------------------
//! \fn ProlongateCC()
//! \brief 2nd-order (piecewise-linear) prolongation operator for cell-centered variables using in orbital advection

KOKKOS_INLINE_FUNCTION
void ProlongateCC(const int m, const int k, const int j, const int i,
               const int dm, const int fk, const int fj, const int fi,
               const int v, const bool multi_d, const bool three_d,
               const DvceArray5D<Real> &ca, const DvceArray5D<Real> &a) {
  // calculate x1-gradient using the min-mod limiter
  Real dl = ca(m,v,k,j,i  ) - ca(m,v,k,j,i-1);
  Real dr = ca(m,v,k,j,i+1) - ca(m,v,k,j,i  );
  Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

  // calculate x2-gradient using the min-mod limiter
  Real dvar2 = 0.0;
  if (multi_d) {
    dl = ca(m,v,k,j  ,i) - ca(m,v,k,j-1,i);
    dr = ca(m,v,k,j+1,i) - ca(m,v,k,j  ,i);
    dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  // calculate x1-gradient using the min-mod limiter
  Real dvar3 = 0.0;
  if (three_d) {
    dl = ca(m,v,k  ,j,i) - ca(m,v,k-1,j,i);
    dr = ca(m,v,k+1,j,i) - ca(m,v,k  ,j,i);
    dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  // interpolate to the finer grid
  a(dm,v,fk,fj,fi  ) = ca(m,v,k,j,i) - dvar1 - dvar2 - dvar3;
  a(dm,v,fk,fj,fi+1) = ca(m,v,k,j,i) + dvar1 - dvar2 - dvar3;
  if (multi_d) {
    a(dm,v,fk,fj+1,fi  ) = ca(m,v,k,j,i) - dvar1 + dvar2 - dvar3;
    a(dm,v,fk,fj+1,fi+1) = ca(m,v,k,j,i) + dvar1 + dvar2 - dvar3;
  }
  if (three_d) {
    a(dm,v,fk+1,fj  ,fi  ) = ca(m,v,k,j,i) - dvar1 - dvar2 + dvar3;
    a(dm,v,fk+1,fj  ,fi+1) = ca(m,v,k,j,i) + dvar1 - dvar2 + dvar3;
    a(dm,v,fk+1,fj+1,fi  ) = ca(m,v,k,j,i) - dvar1 + dvar2 + dvar3;
    a(dm,v,fk+1,fj+1,fi+1) = ca(m,v,k,j,i) + dvar1 + dvar2 + dvar3;
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void RestrictCC
//!  \brief Restricts cell-centered variables to coarse mesh

KOKKOS_INLINE_FUNCTION
void RestrictCC(const int m, const int k, const int j, const int i,
               const int dm, const int ck, const int cj, const int ci,
               const int v, const bool multi_d, const bool three_d,
               const DvceArray5D<Real> &a, const DvceArray5D<Real> &ca) {

  // restrict in 1D
  if (!multi_d) {
    ca(dm,v,ck,cj,ci) = 0.5*(a(m,v,k,j,i) + a(m,v,k,j,i+1));
  
  // restrict in 2D
  } else if (!three_d) {
      ca(dm,v,ck,cj,ci) = 0.25*(a(m,v,k,j  ,i) + a(m,v,k,j  ,i+1)
                              + a(m,v,k,j+1,i) + a(m,v,k,j+1,i+1));
  // restrict in 3D
  } else {
      ca(dm,v,ck,cj,ci) = 0.125*(a(m,v,k  ,j  ,i) + a(m,v,k  ,j  ,i+1)
                               + a(m,v,k  ,j+1,i) + a(m,v,k  ,j+1,i+1)
                               + a(m,v,k+1,j,  i) + a(m,v,k+1,j,  i+1)
                               + a(m,v,k+1,j+1,i) + a(m,v,k+1,j+1,i+1));
  }
  return;
}




//----------------------------------------------------------------------------------------
//! \fn ProlongFCSharedFaces()
//! \brief 2nd-order (piecewise-linear) prolongation operator for face-centered variables
//! on shared X1, X2 and X3-faces between fine and coarse cells

KOKKOS_INLINE_FUNCTION
void ProlongateFCSharedFaces(const int m, const int k, const int j, const int i,
                   const int dm, const int fk, const int fj, const int fi,
                   const bool multi_d, const bool three_d,
                   const DvceFaceFld4D<Real> &cb, const DvceArray5D<Real> &b) {

  Real dvar1, dvar2, dvar3;
  Real dl, dr;


  // Prolongate b.x1f (v=0) by interpolating in x2/x3
  dvar2 = 0.0;
  if (multi_d) {
    dl = cb.x1f(m,k,j  ,i) - cb.x1f(m,k,j-1,i);
    dr = cb.x1f(m,k,j+1,i) - cb.x1f(m,k,j  ,i);
    dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  dvar3 = 0.0;
  if (three_d) {
    dl = cb.x1f(m,k  ,j,i) - cb.x1f(m,k-1,j,i);
    dr = cb.x1f(m,k+1,j,i) - cb.x1f(m,k  ,j,i);
    dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  b(dm,0,fk,fj,fi) = cb.x1f(m,k,j,i) - dvar2 - dvar3;
  if (multi_d) {
    b(dm,0,fk,fj+1,fi) = cb.x1f(m,k,j,i) + dvar2 - dvar3;
  }
  if (three_d) {
    b(dm,0,fk+1,fj  ,fi) = cb.x1f(m,k,j,i) - dvar2 + dvar3;
    b(dm,0,fk+1,fj+1,fi) = cb.x1f(m,k,j,i) + dvar2 + dvar3;
  }


  // Prolongate b.x2f (v=1) by interpolating in x1/x3
  dl = cb.x2f(m,k,j,i  ) - cb.x2f(m,k,j,i-1);
  dr = cb.x2f(m,k,j,i+1) - cb.x2f(m,k,j,i  );
  dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

  dvar3 = 0.0;
  if (three_d) {
    dl = cb.x2f(m,k  ,j,i) - cb.x2f(m,k-1,j,i);
    dr = cb.x2f(m,k+1,j,i) - cb.x2f(m,k  ,j,i);
    dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  b(dm,1,fk  ,fj,fi  ) = cb.x2f(m,k,j,i) - dvar1 - dvar3;
  b(dm,1,fk  ,fj,fi+1) = cb.x2f(m,k,j,i) + dvar1 - dvar3;
  if (three_d) {
    b(dm,1,fk+1,fj,fi  ) = cb.x2f(m,k,j,i) - dvar1 + dvar3;
    b(dm,1,fk+1,fj,fi+1) = cb.x2f(m,k,j,i) + dvar1 + dvar3;
  }

  // Prolongate b.x3f (v=2) by interpolating in x1/x2
  dl = cb.x3f(m,k,j,i  ) - cb.x3f(m,k,j,i-1);
  dr = cb.x3f(m,k,j,i+1) - cb.x3f(m,k,j,i  );
  dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

  dvar2 = 0.0;
  if (multi_d) {
    dl = cb.x3f(m,k,j  ,i) - cb.x3f(m,k,j-1,i);
    dr = cb.x3f(m,k,j+1,i) - cb.x3f(m,k,j  ,i);
    dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  b(dm,2,fk,fj  ,fi  ) = cb.x3f(m,k,j,i) - dvar1 - dvar2;
  b(dm,2,fk,fj  ,fi+1) = cb.x3f(m,k,j,i) + dvar1 - dvar2;
  if (multi_d) {
    b(dm,2,fk,fj+1,fi  ) = cb.x3f(m,k,j,i) - dvar1 + dvar2;
    b(dm,2,fk,fj+1,fi+1) = cb.x3f(m,k,j,i) + dvar1 + dvar2;
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn ProlongateFCInternal()
//! \brief 2nd-order prolongation operator for face-centered variables on internal edges
//! of new fine cells within one coarse cell using divergence-preserving interpolation
//! scheme of Toth & Roe, JCP 180, 736 (2002).

KOKKOS_INLINE_FUNCTION
void ProlongateFCInternal(const int m, const int fk, const int fj, const int fi,
                       const bool three_d, const DvceArray5D<Real> &b) {
  // Prolongate internal fields in 3D
  if (three_d) {
    Real Uxx  = 0.0, Vyy  = 0.0, Wzz  = 0.0;
    Real Uxyz = 0.0, Vxyz = 0.0, Wxyz = 0.0;
    for (int jj=0; jj<2; jj++) {
      int jsgn = 2*jj - 1;
      int fjj  = fj + jj, fjp = fj + 2*jj;
      for (int ii=0; ii<2; ii++) {
        int isgn = 2*ii - 1;
        int fii = fi + ii, fip = fi + 2*ii;
        Uxx += isgn*(jsgn*(b(m,1,fk  ,fjp,fii) + b(m,1,fk+1,fjp,fii)) +
                          (b(m,2,fk+2,fjj,fii) - b(m,2,fk  ,fjj,fii)));

        Vyy += jsgn*(     (b(m,2,fk+2,fjj,fii) - b(m,2,fk  ,fjj,fii)) +
                     isgn*(b(m,0,fk  ,fjj,fip) + b(m,0,fk+1,fjj,fip)));

        Wzz +=       isgn*(b(m,0,fk+1,fjj,fip) - b(m,0,fk  ,fjj,fip)) +
                     jsgn*(b(m,1,fk+1,fjp,fii) - b(m,1,fk  ,fjp,fii));

        Uxyz += isgn*jsgn*(b(m,0,fk+1,fjj,fip) - b(m,0,fk  ,fjj,fip));
        Vxyz += isgn*jsgn*(b(m,1,fk+1,fjp,fii) - b(m,1,fk  ,fjp,fii));
        Wxyz += isgn*jsgn*(b(m,2,fk+2,fjj,fii) - b(m,2,fk  ,fjj,fii));
      }
    }
    Uxx *= 0.125;  Vyy *= 0.125;  Wzz *= 0.125;
    Uxyz *= 0.0625; Vxyz *= 0.0625; Wxyz *= 0.0625;

    b(m,0,fk  ,fj  ,fi+1) = 0.5*(b(m,0,fk  ,fj  ,fi  ) + b(m,0,fk  ,fj  ,fi+2))
                            + Uxx - Vxyz - Wxyz;
    b(m,0,fk  ,fj+1,fi+1) = 0.5*(b(m,0,fk  ,fj+1,fi  ) + b(m,0,fk  ,fj+1,fi+2))
                            + Uxx - Vxyz + Wxyz;
    b(m,0,fk+1,fj  ,fi+1) = 0.5*(b(m,0,fk+1,fj  ,fi  ) + b(m,0,fk+1,fj  ,fi+2))
                            + Uxx + Vxyz - Wxyz;
    b(m,0,fk+1,fj+1,fi+1) = 0.5*(b(m,0,fk+1,fj+1,fi  ) + b(m,0,fk+1,fj+1,fi+2))
                            + Uxx + Vxyz + Wxyz;
    b(m,1,fk  ,fj+1,fi  ) = 0.5*(b(m,1,fk  ,fj  ,fi  ) + b(m,1,fk  ,fj+2,fi  ))
                            + Vyy - Uxyz - Wxyz;
    b(m,1,fk  ,fj+1,fi+1) = 0.5*(b(m,1,fk  ,fj  ,fi+1) + b(m,1,fk  ,fj+2,fi+1))
                            + Vyy - Uxyz + Wxyz;
    b(m,1,fk+1,fj+1,fi  ) = 0.5*(b(m,1,fk+1,fj  ,fi  ) + b(m,1,fk+1,fj+2,fi  ))
                            + Vyy + Uxyz - Wxyz;
    b(m,1,fk+1,fj+1,fi+1) = 0.5*(b(m,1,fk+1,fj  ,fi+1) + b(m,1,fk+1,fj+2,fi+1))
                            + Vyy + Uxyz + Wxyz;
    b(m,2,fk+1,fj  ,fi  ) = 0.5*(b(m,2,fk+2,fj  ,fi  ) + b(m,2,fk  ,fj  ,fi  ))
                            + Wzz - Uxyz - Vxyz;
    b(m,2,fk+1,fj  ,fi+1) = 0.5*(b(m,2,fk+2,fj  ,fi+1) + b(m,2,fk  ,fj  ,fi+1))
                            + Wzz - Uxyz + Vxyz;
    b(m,2,fk+1,fj+1,fi  ) = 0.5*(b(m,2,fk+2,fj+1,fi  ) + b(m,2,fk  ,fj+1,fi  ))
                            + Wzz + Uxyz - Vxyz;
    b(m,2,fk+1,fj+1,fi+1) = 0.5*(b(m,2,fk+2,fj+1,fi+1) + b(m,2,fk  ,fj+1,fi+1))
                            + Wzz + Uxyz + Vxyz;

  // Prolongate internal fields in 2D
  } else {
    Real tmp1 = 0.25*(b(m,1,fk,fj+2,fi+1) - b(m,1,fk,fj,  fi+1)
                    - b(m,1,fk,fj+2,fi  ) + b(m,1,fk,fj,  fi  ));
    Real tmp2 = 0.25*(b(m,0,fk,fj,  fi  ) - b(m,0,fk,fj,  fi+2)
                    - b(m,0,fk,fj+1,fi  ) + b(m,0,fk,fj+1,fi+2));
    b(m,0,fk,fj  ,fi+1) = 0.5*(b(m,0,fk,fj,  fi  ) + b(m,0,fk,fj,  fi+2)) + tmp1;
    b(m,0,fk,fj+1,fi+1) = 0.5*(b(m,0,fk,fj+1,fi  ) + b(m,0,fk,fj+1,fi+2)) + tmp1;
    b(m,1,fk,fj+1,fi  ) = 0.5*(b(m,1,fk,fj,  fi  ) + b(m,1,fk,fj+2,fi  )) + tmp2;
    b(m,1,fk,fj+1,fi+1) = 0.5*(b(m,1,fk,fj,  fi+1) + b(m,1,fk,fj+2,fi+1)) + tmp2;
  }
  return;
}




//----------------------------------------------------------------------------------------
//! \fn void RestrictFC
//! \brief Restricts face-centered variables to coarse mesh. Only need to restrict
//! B1 and B3 for orbital advection.

KOKKOS_INLINE_FUNCTION
void RestrictFC(const int m, const int k, const int j, const int i,
               const int dm, const int ck, const int cj, const int ci,
               const int n, const bool multi_d, const bool three_d,
               const DvceFaceFld4D<Real> &b, const DvceArray5D<Real> &cb) {

  // restrict in 1D
  if (!multi_d) {
    // restrict B1
    cb(dm,0,ck,cj,ci) = b.x1f(m,ck,cj,i);

    // restrict B3
    Real b3coarse = 0.5*(b.x3f(m,ck,cj,i) + b.x3f(m,ck,cj,i+1));
    cb(dm,2,ck  ,cj,ci) = b3coarse;
    cb(dm,2,ck+1,cj,ci) = b3coarse;


  // restrict in 2D
  } else if (!three_d) {
    // restrict B1
    cb(dm,0,ck,cj,ci) = 0.5*(b.x1f(m,ck,j,i) + b.x1f(m,ck,j+1,i));
    
    // restrict B3
    Real b3coarse = 0.25*(b.x3f(m,ck,j  ,i) + b.x3f(m,ck,j  ,i+1)
                        + b.x3f(m,ck,j+1,i) + b.x3f(m,ck,j+1,i+1));
    cb(dm,2,ck  ,cj,ci) = b3coarse;
    cb(dm,2,ck+1,cj,ci) = b3coarse;


  // restrict in 3D
  } else {
    // restrict B1
    cb(dm,0,ck,cj,ci) =
      0.25*(b.x1f(m,k  ,j,i) + b.x1f(m,k  ,j+1,i)
          + b.x1f(m,k+1,j,i) + b.x1f(m,k+1,j+1,i));

    // restrict B3
    cb(dm,2,ck,cj,ci) =
      0.25*(b.x3f(m,k,j  ,i) + b.x3f(m,k,j  ,i+1)
          + b.x3f(m,k,j+1,i) + b.x3f(m,k,j+1,i+1));
  }
  return;
}



#endif // ORBTIAL_ADVECTION_RESTRICT_PROLONG_HPP_
