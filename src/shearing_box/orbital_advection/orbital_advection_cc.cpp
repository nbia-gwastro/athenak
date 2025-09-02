//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file orbital_advection_cc.cpp
//! \brief functions to pack/send and recv/unpack boundary values for cell-centered (CC)
//! variables in the orbital advection step. Data is shifted by the appropriate offset
//! during the recv/unpack step, so these functions both communicate the data and perform
//! the shift.

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
// OrbitalAdvectionCC derived class constructor:

OrbitalAdvectionCC::OrbitalAdvectionCC(MeshBlockPack *pp, ParameterInput *pin, int nv) :
    OrbitalAdvection(pp, pin) {
}

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionCC::PackAndSendCC()
//! \brief Pack cell-centered variables into boundary buffers and send to neighbors for
//! the orbital advection step.  Only ghost zones on the x2-faces (Y-faces) are passed.
//!
//! Input arrays must be 5D Kokkos View dimensioned (nmb, nvar, nx3, nx2, nx1)

TaskStatus OrbitalAdvectionCC::PackAndSendCC(DvceArray5D<Real> &a) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR

  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;

  auto &ng = pmy_pack->pmesh->mb_indcs.ng;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  int nnghbrs = 8; // number of neighbors on x2-faces (Y-faces)

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  int nmnv = nmb*nnghbrs*nvar;  // only consider 8 neighbors (x2-faces)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("oa-pack", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbrs*nvar);
    const int n = (tmember.league_rank() - m*(nnghbrs*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbrs*nvar) - n*nvar);

    // indices of x2-face buffers in nghbr view
    int nnghbr = n + nnghbrs;

    // only load buffers when neighbor exists
    if (nghbr.d_view(m,nnghbr).gid >= 0) {
      // if neighbor is at coarser level, use coar indices to pack buffer
      int il, iu, jl, ju, kl, ku;
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

        // copy directly into recv buffer if MeshBlocks on same rank
        if (nghbr.d_view(m,nnghbr).rank == my_rank) {
          // if neighbor is at same level, load data directly from a
          if (nghbr.d_view(m,nnghbr).lev == mblev.d_view(m)) {
            rbuf[dn-nnghbrs].vars(dm,v,(k-kl),(j-jl),(i-il)) = a(m,v,k,j,i);

          // else if neighbor is at finer level, prolongate data from a and load
          } else if (nghbr.d_view(m,nnghbr).lev > mblev.d_view(m)) {
            // indices in fine neighbor corresponding to (i,j,k) in coarse
            int fi = 2*(i - il);
            int fj = 2*(j - jl);
            int fk = 2*(k - kl);
            ProlongateCC(m, k, j, i, dm, fk, fj, fi, v, multi_d, three_d, a, rbuf[dn-nnghbrs].vars);

          // else neighbor is at coarser level, restrict a and load
          } else {
            int fi = 2*(i - il) + il;
            int fj = 2*(j - jl) + jl;
            int fk = 2*(k - kl) + kl;
            if (n>=4) { // if neighbor is on x2-face in positive direction, shift starting j index
              fj += (jl-ng);
            }
            RestrictCC(m, fk, fj, fi, dm, (k-kl), (j-jl), (i-il), v, multi_d, three_d, a, rbuf[dn-nnghbrs].vars);         
          }


        // else copy into send buffer for MPI communication below
        } else {
          // if neighbor is at same or finer level, load data directly from a
          if (nghbr.d_view(m,nnghbr).lev == mblev.d_view(m)) {
            sbuf[n].vars(m,v,(k-kl),(j-jl),(i-il)) = a(m,v,k,j,i);

          // else if neighbor is at finer level, prolongate data from a and load
          } else if (nghbr.d_view(m,nnghbr).lev > mblev.d_view(m)) {
            // fine loading indices corresponding to (i,j,k) in current coarse
            int fi = 2*(i - il);
            int fj = 2*(j - jl);
            int fk = 2*(k - kl);
            ProlongateCC(m, k, j, i, m, fk, fj, fi, v, multi_d, three_d, a, sbuf[n].vars);

          // else neighbor is at coarser level, restrict a and load
          } else {
            // indices in current fine MB corresponding to loading indices (i,j,k) in coarse
            int fi = 2*(i - il) + il;
            int fj = 2*(j - jl) + jl;
            int fk = 2*(k - kl) + kl;
            if (n>=4) { // if neighbor is on x2-face in positive direction, shift starting j index
              fj += (jl-ng);
            }
            RestrictCC(m, fk, fj, fi, m, (k-kl), (j-jl), (i-il), v, multi_d, three_d, a, sbuf[n].vars);
          }
        }
      }); 
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
      if (nghbr.h_view(m,nnghbr).gid >= 0) { // neighbor exists and not a physical bndry
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
//! \!fn void OrbitalAdvectionCC::RecvAndUnpackCC()
//! \brief Receive and unpack boundary buffers for CC variables with orbital advection,
//! and apply shift in x2- (y-) direction across entire MeshBlock by applying both an
//! integer shift and a fractional offset to input array a.

TaskStatus OrbitalAdvectionCC::RecvAndUnpackCC(DvceArray5D<Real> &a,
                                               ReconstructionMethod rcon) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbrs = 8; // number of neighbors on x2-faces (Y-faces)
  auto &rbuf = recvbuf;
  auto &nghbr = pmy_pack->pmb->nghbr;
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

  //----- STEP 2: buffers have all completed, so unpack and apply shift

  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  auto &mblev = pmy_pack->pmb->mb_lev;

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
  par_for_outer("oa-unpk",DevExeSpace(),scr_size,scr_lvl,0,(nmb-1),0,(nvar-1),ks,ke,is,ie,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int v, const int k, const int i) {
    ScrArray1D<Real> a_(member.team_scratch(scr_lvl), nfx); // 1D slice of data
    ScrArray1D<Real> flx(member.team_scratch(scr_lvl), nfx); // "flux" at faces

    Real &x1min = mbsize.d_view(m).x1min;
    Real &x1max = mbsize.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real yshear = -(qo)*x1v*dt;
    int joffset = static_cast<int>(yshear/(mbsize.d_view(m).dx2));

    // Load scratch array with no shift
    // loop over x2 indices (and 4 neighbors of each side)
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(member, (nfx)*4), [&](const int idx) {
      int jf = idx / 4;
      int n = (idx - jf * 4);
      if (jf < jfs) {
        // Load from L boundary buffers neighbors 8-12
        if (nghbr.d_view(m,n+nnghbrs).gid >= 0) {
          // if neighbor is at same or coarser level, load directly from buffer
          if (nghbr.d_view(m,n+nnghbrs).lev <= mblev.d_view(m)) {
            a_(jf) = rbuf[n].vars(m,v,(k-ks),jf,(i-is));

          // else neighbor is at finer level, load using fine indices in buffer
          // and check the neighbor is the correct one of the four for this i and k
          } else {
            int il = rbuf[n].ifine.bis;
            int iu = rbuf[n].ifine.bie;
            int kl = rbuf[n].ifine.bks;
            int ku = rbuf[n].ifine.bke;
            if ((i>=il) & (i<=iu) & (k>=kl) & (k<=ku)) {
              a_(jf) = rbuf[n].vars(m,v,(k-kl),jf,(i-il));
            }
          }
        }

      } else if (jf <= jfe) {
        // Load from conserved variables themselves (addressed with j=jf-jfs+js)
        a_(jf) = a(m,v,k,jf-jfs+js,i);

      } else {
        // Load from R boundary buffers neighbors 12-16
        if (nghbr.d_view(m,n+nnghbrs+4).gid >= 0) {
          // if neighbor is at same or coarser level, load directly from buffer
          if (nghbr.d_view(m,n+nnghbrs+4).lev <= mblev.d_view(m)) {
            a_(jf) = rbuf[n+4].vars(m,v,(k-ks),jf-(jfe+1),(i-is));

          // else neighbor is at finer level, load using fine indices in buffer
          // and check the neighbor is adjacent
          } else {
            int il = rbuf[n+4].ifine.bis;
            int iu = rbuf[n+4].ifine.bie;
            int kl = rbuf[n+4].ifine.bks;
            int ku = rbuf[n+4].ifine.bke;
            if ( (i>=il) & (i<=iu) & (k>=kl) & (k<=ku) ) {
              a_(jf) = rbuf[n+4].vars(m,v,(k-kl),jf-(jfe+1),(i-il));
            }                
          }
        }
      }
    });
    member.team_barrier();

    // Compute "fluxes" at shifted cell faces
    Real epsi = fmod(yshear,(mbsize.d_view(m).dx2))/(mbsize.d_view(m).dx2);
    switch (rcon) {
      case ReconstructionMethod::dc:
        DC_RemapFlx(member, (jfs-joffset), (jfe+1-joffset), epsi, a_, flx);
        break;
      case ReconstructionMethod::plm:
        PLM_RemapFlx(member, (jfs-joffset), (jfe+1-joffset), epsi, a_, flx);
        break;
      case ReconstructionMethod::ppm4:
      case ReconstructionMethod::ppmx:
      case ReconstructionMethod::wenoz:
        PPMX_RemapFlx(member, (jfs-joffset), (jfe+1-joffset), epsi, a_, flx);
        break;
      default:
        break;
    }
    member.team_barrier();

    // Update CC variables with both integer shift (from a_) and a conservative remap
    // for the remaining fraction of a cell using upwind "fluxes"
    par_for_inner(member, js, je, [&](const int j) {
      int jf = j-js + jfs;
      a(m,v,k,j,i) = a_(jf-joffset) - (flx(jf+1-joffset) - flx(jf-joffset));
    });
  });

  return TaskStatus::complete;
}
