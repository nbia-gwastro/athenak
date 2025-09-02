//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file orbital_advection_fc.cpp
//! \brief functions to pack/send and recv/unpack boundary values for face-centered (FC)
//! variables in the orbital advection step used with the shearing box. Data is shifted
//! by the appropriate offset during the recv/unpack step, so these functions both
//! communicate the data and perform the shift.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "shearing_box/shearing_box.hpp"
#include "shearing_box/remap_fluxes.hpp"
#include "orbital_advection.hpp"
#include "orbital_advection_restrict_prolong.hpp"
#include "mhd/mhd.hpp"

//----------------------------------------------------------------------------------------
// OrbitalAdvectionFC derived class constructor:

OrbitalAdvectionFC::OrbitalAdvectionFC(MeshBlockPack *pp, ParameterInput *pin) :
  OrbitalAdvection(pp, pin) {
}

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionFC::PackAndSendFC()
//! \brief Pack face-centered fields into boundary buffers and send to neighbors for
//! the orbital advection step. Only ghost zones on the x2-faces (Y-faces) are passed.
//! Note only B3 and B1 need be passed.

TaskStatus OrbitalAdvectionFC::PackAndSendFC(DvceFaceFld4D<Real> &b) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;

  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &ng = pmy_pack->pmesh->mb_indcs.ng;
  int nnghbrs = 8; // number of neighbors on x2-faces (Y-faces)

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  int nmn = nmb*nnghbrs;  // only consider 8 neighbors (x2-faces) 
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmn, Kokkos::AUTO);
  Kokkos::parallel_for("oa-packB", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbrs); // MeshBlock index
    const int n = (tmember.league_rank() - m*(nnghbrs)); // neighbor index

    // indices of x2-face buffers in nghbr view
    int nnghbr = n + nnghbrs;

    // only load buffers when neighbor exists
    if (nghbr.d_view(m,nnghbr).gid >= 0) {
      int il, iu, jl, ju, kl, ku;
      // if neighbor is at coarser level, use coar indices to pack buffer
      if (nghbr.d_view(m,nnghbr).lev < mblev.d_view(m)) {
        il = sbuf[n].icoar.bis;
        iu = sbuf[n].icoar.bie;
        jl = sbuf[n].icoar.bjs;
        ju = sbuf[n].icoar.bje;
        kl = sbuf[n].icoar.bks;
        ku = sbuf[n].icoar.bke;
      // if neighbor is at same level, use same indices to pack buffer
      } else if (nghbr.d_view(m,nnghbr).lev == mblev.d_view(m)) {
        il = sbuf[n].isame.bis;
        iu = sbuf[n].isame.bie;
        jl = sbuf[n].isame.bjs;
        ju = sbuf[n].isame.bje;
        kl = sbuf[n].isame.bks;
        ku = sbuf[n].isame.bke;
      // if neighbor is at finer level, use fine indices to pack buffer
      } else {
        il = sbuf[n].ifine.bis;
        iu = sbuf[n].ifine.bie;
        jl = sbuf[n].ifine.bjs;
        ju = sbuf[n].ifine.bje;
        kl = sbuf[n].ifine.bks;
        ku = sbuf[n].ifine.bke;
      }
      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int nji = nj*ni;
      int nkji = nk*nj*ni;

      // index of recv'ing (destination) MB and buffer MB IDs are stored
      // sequentially in MeshBlockPacks, so array index equals (target_id - first_id)
      int dm = nghbr.d_view(m,nnghbr).gid - mbgid.d_view(0);
      int dn = nghbr.d_view(m,nnghbr).dest;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;

        // copy B1/B3 directly into recv buffer if MeshBlocks on same rank
        if (nghbr.d_view(m,nnghbr).rank == my_rank) {
          // if neighbor is at same level, load data directly from b
          if (nghbr.d_view(m,nnghbr).lev == mblev.d_view(m)) {
            rbuf[dn-nnghbrs].vars(dm,0,(k-kl),(j-jl),(i-il)) = b.x1f(m,k,j,i);
            rbuf[dn-nnghbrs].vars(dm,1,(k-kl),(j-jl),(i-il)) = b.x2f(m,k,j,i);
            rbuf[dn-nnghbrs].vars(dm,2,(k-kl),(j-jl),(i-il)) = b.x3f(m,k,j,i);
          
          // else if neighbor is at finer level, prolongate data from b and load
          } else if (nghbr.d_view(m,nnghbr).lev > mblev.d_view(m)) {
            int fi = 2*(i - il);
            int fj = 2*(j - jl);
            int fk = 2*(k - kl);
            ProlongateFCSharedFaces(m, k, j, i, dm, fk, fj, fi, multi_d, three_d, b, rbuf[dn-nnghbrs].vars);

          // else neighbor is at coarser level, restrict b and load
          } else {
            int fi = 2*(i - il) + il;
            int fj = 2*(j - jl) + jl;
            int fk = 2*(k - kl) + kl;
            if (n>=4) { // if neighbor is on x2-face in positive direction, shift starting j index
              fj += (jl-ng);
            }
            RestrictFC(m, fk, fj, fi, dm, (k-kl), (j-jl), (i-il), n, multi_d, three_d, b, rbuf[dn-nnghbrs].vars);
          }
        // else copy B1/B3 into send buffer for MPI communication below
        } else {
          if (nghbr.d_view(m,nnghbr).lev == mblev.d_view(m)) {
            sbuf[n].vars(m,0,(k-kl),(j-jl),(i-il)) = b.x1f(m,k,j,i);
            sbuf[n].vars(m,1,(k-kl),(j-jl),(i-il)) = b.x2f(m,k,j,i);
            sbuf[n].vars(m,2,(k-kl),(j-jl),(i-il)) = b.x3f(m,k,j,i);

          // else if neighbor is at finer level, prolongate data from b and load
          } else if (nghbr.d_view(m,nnghbr).lev > mblev.d_view(m)) {
            int fi = 2*(i - il);
            int fj = 2*(j - jl);
            int fk = 2*(k - kl);
            ProlongateFCSharedFaces(m, k, j, i, m, fk, fj, fi, multi_d, three_d, b, sbuf[n].vars);

          // else neighbor is at coarser level, restrict b and load
          } else {
            int fi = 2*(i - il) + il;
            int fj = 2*(j - jl) + jl;
            int fk = 2*(k - kl) + kl;
            if (n>=4) { // if neighbor is on x2-face in positive direction, shift starting j index
              fj += (jl-ng);
            }
            RestrictFC(m, fk, fj, fi, m, (k-kl), (j-jl), (i-il), n, multi_d, three_d, b, sbuf[n].vars);
          }
        }
      });
      tmember.team_barrier();

      // if neighbor is at finer level, also prolongate and load internal faces
      if (nghbr.d_view(m,nnghbr).lev > mblev.d_view(m)) {
        int ni_ = iu - il + 1;
        int nj_ = ju - jl + 1;
        int nk_ = ku - kl + 1;
        int nji_ = nj_*ni_;
        int nkji_ = nk_*nj_*ni_;
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji_), [&](const int idx) {
          int k = (idx)/nji_;
          int j = (idx - k*nji_)/ni_;
          int i = (idx - k*nji_ - j*ni_) + il;
          k += kl;
          j += jl;

          // copy B1/B3 directly into recv buffer if MeshBlocks on same rank
          if (nghbr.d_view(m,nnghbr).rank == my_rank) {
            int fi = 2*(i - il);
            int fj = 2*(j - jl);
            int fk = 2*(k - kl);
            ProlongateFCInternal(dm, fk, fj, fi, three_d, rbuf[dn-nnghbrs].vars);

          // else copy B1/B3 into send buffer for MPI communication below
          } else {
            int fi = 2*(i - il);
            int fj = 2*(j - jl);
            int fk = 2*(k - kl);
            ProlongateFCInternal(m, fk, fj, fi, three_d, sbuf[n].vars);
          }
        });
      } // end if-neighbor-is-finer block
    } // end if-neighbor-exists block
  }); // end par_for_outer

#if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI
  Kokkos::fence();
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbrs; ++n) {
      // indices of x2-face buffers in nghbr view
      int nnghbr = n + nnghbrs;
      if (nghbr.h_view(m,nnghbr).gid >= 0) {  // neighbor exists and not a physical bndry
        // index and rank of destination Neighbor
        int dn = nghbr.h_view(m,nnghbr).dest;
        int drank = nghbr.h_view(m,nnghbr).rank;
        if (drank != my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,nnghbr).gid - pmy_pack->pmesh->gids_eachrank[drank];
          int tag = CreateBvals_MPI_Tag(lid, dn);

          // get ptr to send buffer when neighbor is at coarser/same/fine level
          using Kokkos::ALL;
          auto send_ptr = Kokkos::subview(sbuf[n].vars, m, ALL, ALL, ALL, ALL);
          int data_size = send_ptr.size();

          int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_orb_advect, &(sbuf[n].vars_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \!fn void OrbitalAdvectionFC::RecvAndUnpackFC()
//! \brief Receive and unpack boundary buffers for FC fields with orbital advection, and
//! apply shift in x2- (y-) direction across entire MeshBlock. Since CT is required to
//! update fields, the algorithm used here is different from that used for CC variables in
//! RecvAndUnpackCC(). Here an effective electric field is computed including both the
//! integer and fractional cell shifts. These fields are then used to update B using CT.
//! The fields themselves are not directly remapped like the CC variables.

TaskStatus OrbitalAdvectionFC::RecvAndUnpackFC(DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld_orb,
                                             ReconstructionMethod rcon) {
  int nmb = pmy_pack->nmb_thispack;
  int nnghbrs = 8; // number of neighbors on x2-faces (Y-faces)
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recvbuf;
#if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed

  bool bflag = false;
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbrs; ++n) {
      // indices of x2-face buffers in nghbr view
      int nnghbr = n + nnghbrs;
      if (nghbr.h_view(m,nnghbr).gid >= 0) { // neighbor exists and not a physical bndry
        if (nghbr.h_view(m,nnghbr).rank != global_variable::my_rank) {
          int test;
          int ierr = MPI_Test(&(rbuf[n].vars_req[m]), &test, MPI_STATUS_IGNORE);
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          if (!(static_cast<bool>(test))) {
            bflag = true;
          }
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in testing non-blocking receives"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // exit if recv boundary buffer communications have not completed
  if (bflag) {return TaskStatus::incomplete;}
#endif

  //----- STEP 2: buffers have all completed, so unpack and compute effective EMF
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &mbgid = pmy_pack->pmb->mb_gid;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &ng = indcs.ng;
  int jfs = ng + maxjshift;
  int jfe = jfs + indcs.nx2 - 1;
  int nfx = indcs.nx2 + 2*(ng + maxjshift);

  auto &mbsize = pmy_pack->pmb->mb_size;
  Real &dt = pmy_pack->pmesh->dt;
  Real qo = qshear*omega0;

  int scr_lvl=0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(nfx) * 2;
  auto &emfx = efld_orb.x1e;
  auto &emfz = efld_orb.x3e;
  par_for_outer("oa-unB",DevExeSpace(),scr_size,scr_lvl,0,(nmb-1),0,1,ks,ke+1,is,ie+1,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int v, const int k, const int i) {
    ScrArray1D<Real> b0_(member.team_scratch(scr_lvl), nfx); // 1D slice of data
    ScrArray1D<Real> flx(member.team_scratch(scr_lvl), nfx); // "flux" at faces
    int v_ = v*2; // convert v to 0 for B1, 2 for B3
    Real &x1min = mbsize.d_view(m).x1min;
    Real &x1max = mbsize.d_view(m).x1max;
    int nx1 = indcs.nx1;

    Real x1;
    if (v_==0) {
      // B1 located at x1-cell faces
      x1 = LeftEdgeX(i-is, nx1, x1min, x1max);
    } else if (v_==2) {
      // B3 located at x1-cell centers
      x1 = CellCenterX(i-is, nx1, x1min, x1max);
    }
    Real yshear = -(qo)*x1*dt;
    int joffset = static_cast<int>(yshear/(mbsize.d_view(m).dx2));

    // Load scratch array with no shift
    // loop over x2 indices (and 4 neighbors of each side)
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(member, nfx*4), [&](const int idx) {
      int jf = idx / 4;
      int n = (idx - jf * 4);

      if (jf < jfs) {
        // Load from L boundary buffer of neighbors 8-12
        if (nghbr.d_view(m,n+nnghbrs).gid >= 0) {
          // if neighbor is at same or coarser level, load directly from buffer
          if (nghbr.d_view(m,n+nnghbrs).lev <= mblev.d_view(m)) {
            b0_(jf) = rbuf[n].vars(m,v_,(k-ks),jf,(i-is));
          // else neighbor is at finer level, load using fine indices in buffer
          // and check the neighbor is the correct one of the four for this i and k
          } else {
            int il_ = rbuf[n].ifine.bis;
            int iu_ = rbuf[n].ifine.bie;
            int kl_ = rbuf[n].ifine.bks;
            int ku_ = rbuf[n].ifine.bke;
            if ((i>=il_) & (i<=iu_) & (k>=kl_) & (k<=ku_)) {
              b0_(jf) = rbuf[n].vars(m,v_,(k-kl_),jf,(i-il_));
            }
          }
        }
      } else if (jf <= jfe) {
        // Load from array itself (addressed with j=jf-jfs+js)
        if (v_==0) {
          b0_(jf) = b0.x1f(m,k,(jf-jfs+js),i);
        } else if (v_==2) {
          b0_(jf) = b0.x3f(m,k,(jf-jfs+js),i);
        }
      } else {
        // Load scratch arrays from R boundary buffer of neighbors 12-16
        if (nghbr.d_view(m,n+nnghbrs+4).gid >= 0) {
          // if neighbor is at same or coarser level, load directly from buffer
          if (nghbr.d_view(m,n+nnghbrs+4).lev <= mblev.d_view(m)) {
            b0_(jf) = rbuf[n+4].vars(m,v_,(k-ks),jf-(jfe+1),(i-is));
          // else neighbor is at finer level, load using fine indices in buffer
          // and check the neighbor is the correct one of the four for this i and k
          } else {
            int il_ = rbuf[n+4].ifine.bis;
            int iu_ = rbuf[n+4].ifine.bie;
            int kl_ = rbuf[n+4].ifine.bks;
            int ku_ = rbuf[n+4].ifine.bke;
            if ( (i>=il_) & (i<=iu_) & (k>=kl_) & (k<=ku_) ) {
              b0_(jf) = rbuf[n+4].vars(m,v_,(k-kl_),jf-(jfe+1),(i-il_));
            }                
          }
        }
      }
    });
    member.team_barrier();

    // Compute x2-fluxes at shifted cell faces
    Real epsi = fmod(yshear,(mbsize.d_view(m).dx2))/(mbsize.d_view(m).dx2);
    switch (rcon) {
      case ReconstructionMethod::dc:
        DC_RemapFlx(member, (jfs-joffset), (jfe+1-joffset), epsi, b0_, flx);
        break;
      case ReconstructionMethod::plm:
        PLM_RemapFlx(member, (jfs-joffset), (jfe+1-joffset), epsi, b0_, flx);
        break;
      case ReconstructionMethod::ppm4:
      case ReconstructionMethod::ppmx:
      case ReconstructionMethod::wenoz:
        PPMX_RemapFlx(member, (jfs-joffset), (jfe+1-joffset), epsi, b0_, flx);
        break;
      default:
        break;
    }
    member.team_barrier();


    // Compute emfz =  VyBx, which is at cell-face in x1-direction
    if (v_==0) {
      par_for_inner(member, js, je+1, [&](const int j) {
        int jf = j-js + jfs;
        emfz(m,k,j,i) = flx(jf-joffset);
        // Sum integer offsets into effective EMFs
        for (int jj=1; jj<=joffset; jj++) {
          emfz(m,k,j,i) += b0_(jf-jj);
        }
        for (int jj=(joffset+1); jj<=0; jj++) {
          emfz(m,k,j,i) -= b0_(jf-jj);
        }
        // scale by dx3 to account to refinement in CT update
        emfz(m,k,j,i) = emfz(m,k,j,i)*mbsize.d_view(m).dx3;
      });
      member.team_barrier();

    // Compute emfx = -VyBz, which is at cell-center in x1-direction
    } else if (v_==2) {
      par_for_inner(member, js, je+1, [&](const int j) {
        int jf = j-js + jfs;
        emfx(m,k,j,i) = -flx(jf-joffset);
        // Sum integer offsets into effective EMFs
        for (int jj=1; jj<=joffset; jj++) {
          emfx(m,k,j,i) -= b0_(jf-jj);
        }
        for (int jj=(joffset+1); jj<=0; jj++) {
          emfx(m,k,j,i) += b0_(jf-jj);
        }
        // scale by dx1 to account to refinement in CT update
        emfx(m,k,j,i) = emfx(m,k,j,i)*mbsize.d_view(m).dx1;
      });
      member.team_barrier();
    }
  });
  return TaskStatus::complete;
}


//----------------------------------------------------------------------------------------
//! \fn  void OrbitalAdvectionFC::CT_OA
//  \brief Constrained Transport implementation of dB/dt = -Curl(E), where E=-(v X B)
//  for the orbital advection step. Here the EMFs computed in RecvAndUnpackFC()
//  are used to update the face-centered fields. 
//  To be clear, the edge-centered variable 'efld' stores E = -(v X B).
TaskStatus OrbitalAdvectionFC::CT_OA(DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld_orb) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmy_pack->nmb_thispack;

  auto emfx = efld_orb.x1e;
  auto emfz = efld_orb.x3e;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const bool &three_d_ = pmy_pack->pmesh->three_d;

  // Update face-centered fields using CT
  //---- update B1 (only for 2D/3D problems)
  if (pmy_pack->pmesh->multi_d) {
    par_for("oaCT-b1", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) -= (emfz(m,k,j+1,i) - emfz(m,k,j,i))/mbsize.d_view(m).dx3;
    });
  }

  //---- update B2 (curl terms in 1D and 3D problems)
  par_for("oaCT-b2", DevExeSpace(), 0, nmb-1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real dydx = mbsize.d_view(m).dx2/mbsize.d_view(m).dx1;
    b0.x2f(m,k,j,i) += dydx*(emfz(m,k,j,i+1) - emfz(m,k,j,i))/mbsize.d_view(m).dx3;
    if (three_d_) {
      Real dydz = mbsize.d_view(m).dx2/mbsize.d_view(m).dx3;
      b0.x2f(m,k,j,i) -= dydz*(emfx(m,k+1,j,i) - emfx(m,k,j,i))/mbsize.d_view(m).dx1;
    }
  });

  //---- update B3 (curl terms in 1D and 2D/3D problems)
  if (pmy_pack->pmesh->multi_d) {
    par_for("oaCT-b3", DevExeSpace(), 0, nmb-1, ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x3f(m,k,j,i) += (emfx(m,k,j+1,i) - emfx(m,k,j,i))/mbsize.d_view(m).dx1;
    });
  }

  return TaskStatus::complete;
}
