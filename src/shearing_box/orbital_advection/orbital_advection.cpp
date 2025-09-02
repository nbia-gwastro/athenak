//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file orbital_advection.cpp
//! \brief constructor for OrbitalAdvection abstract base class.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/nghbr_index.hpp"
#include "coordinates/cell_locations.hpp"
#include "shearing_box/shearing_box.hpp"
#include "shearing_box/remap_fluxes.hpp"
#include "orbital_advection.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"

//----------------------------------------------------------------------------------------
//! OrbitalAdvection base class constructor
//! Called by Hydro and MHD constructors, so cannot access any data inside Hydro/MHD
//! classes as it may not be properly allocated yet.

OrbitalAdvection::OrbitalAdvection(MeshBlockPack *ppack, ParameterInput *pin) :
    maxjshift(1),
    shearing_box_r_phi(false),     // 2D r-phi not yet implemented
    pmy_pack(ppack) {
  // Read shear rate and orbital frequency
  qshear = pin->GetReal("shearing_box","qshear");
  omega0 = pin->GetReal("shearing_box","omega0");

  // estimate maximum integer shift in x2-direction for orbital advection
  Real xmin = fabs(ppack->pmesh->mesh_size.x1min);
  Real xmax = fabs(ppack->pmesh->mesh_size.x1max);
  maxjshift = 2*(static_cast<int>((ppack->pmesh->cfl_no)*std::max(xmin,xmax)/2)) + 2; // TODO: what is the optimal/minimum choice here?

  for (int n=0; n<8; ++n) {
#if MPI_PARALLEL_ENABLED
  // For orbital advection, communication is only with x2-face neighbors -> up to 8 blocks
  // initialize vectors of MPI request in 2 elements of fixed length arrays
    int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
    sendbuf[n].vars_req = new MPI_Request[nmb];
    recvbuf[n].vars_req = new MPI_Request[nmb];
    for (int m=0; m<nmb; ++m) {
      sendbuf[n].vars_req[m] = MPI_REQUEST_NULL;
      recvbuf[n].vars_req[m] = MPI_REQUEST_NULL;
    }
#endif
      // initialize data sizes in each send/recv buffer to zero
      sendbuf[n].isame_ndatx1 = 0; sendbuf[n].isame_ndatx2 = 0; sendbuf[n].isame_ndatx3 = 0;
      sendbuf[n].icoar_ndatx1 = 0; sendbuf[n].icoar_ndatx2 = 0; sendbuf[n].icoar_ndatx3 = 0;
      sendbuf[n].ifine_ndatx1 = 0; sendbuf[n].ifine_ndatx2 = 0; sendbuf[n].ifine_ndatx3 = 0;
      recvbuf[n].isame_ndatx1 = 0; recvbuf[n].isame_ndatx2 = 0; recvbuf[n].isame_ndatx3 = 0;
      recvbuf[n].icoar_ndatx1 = 0; recvbuf[n].icoar_ndatx2 = 0; recvbuf[n].icoar_ndatx3 = 0;
      recvbuf[n].ifine_ndatx1 = 0; recvbuf[n].ifine_ndatx2 = 0; recvbuf[n].ifine_ndatx3 = 0;
  }

#if MPI_PARALLEL_ENABLED
  // create unique communicators for shearing box
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_orb_advect);
#endif


}

//----------------------------------------------------------------------------------------
// OrbitalAdvection base class destructor

OrbitalAdvection::~OrbitalAdvection() {
#if MPI_PARALLEL_ENABLED
  for (int n=0; n<8; ++n) {
    delete [] sendbuf[n].vars_req;
    delete [] recvbuf[n].vars_req;
  }
#endif
}



//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvection::InitializeBuffers
//! \brief initialize each element of send/recv MeshBoundaryBuffers fixed-length arrays
//!
//! NOTE: order of vector elements is crucial and cannot be changed.  It must match
//! order of boundaries in nghbr vector
//! NOTE2: work here cannot be done in MeshBoundaryValues constructor since it calls pure
//! virtual functions that only get instantiated when the derived classes are constructed

void OrbitalAdvection::InitializeBuffers(const int nvar) {
  // set number of subblocks in x2- and x3-dirs
  int nfx = 1, nfy = 1, nfz = 1;
  if (pmy_pack->pmesh->multilevel) {
    nfx = 2;
    if (pmy_pack->pmesh->multi_d) nfy = 2;
    if (pmy_pack->pmesh->three_d) nfz = 2;
  }

  // initialize buffers used for uniform grid and SMR/AMR calculations
  // only add buffers on x2 faces; NeighborIndex = [8,...,15]
  int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));
  if (pmy_pack->pmesh->multi_d) {
    for (int m=-1; m<=1; m+=2) {
      for (int fz=0; fz<nfz; fz++) {
        for (int fx=0; fx<nfx; fx++) {
          int indx = NeighborIndex(0,m,0,fx,fz);
          InitSendIndices(sendbuf[indx-8],0, m, 0, fx, fz);
          InitRecvIndices(recvbuf[indx-8],0, m, 0, fx, fz);
          sendbuf[indx-8].AllocateBuffers(nmb, nvar);
          recvbuf[indx-8].AllocateBuffers(nmb, nvar);
          indx++;
        }
      }
    }
  };
  return;
}