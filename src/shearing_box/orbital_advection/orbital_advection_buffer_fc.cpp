//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file buffs_fc.cpp
//  \brief functions to allocate and initialize buffers for face-centered variables

#include <cstdlib>
#include <iostream>
#include <algorithm> // max

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "shearing_box/shearing_box.hpp"
#include "orbital_advection.hpp"

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionFC::InitSendIndices
//! \brief Calculates indices of cells used to pack buffers and send FC data for buffers
//! on same/coarser and finer levels.  The same sets of indices are used for all three
//! components (x1f,x2f,x3f) of face-centered fields.
//!
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0).  The arguments f1/2 are the coordinates
//! of subblocks within faces/edges (only relevant with SMR/AMR)

void OrbitalAdvectionFC::InitSendIndices(OrbitalAdvectionBoundaryBuffer &buf,
                                           int ox1, int ox2, int ox3, int f1, int f2) {
  auto &mb_indcs  = pmy_pack->pmesh->mb_indcs;
  int ng  = mb_indcs.ng;
  int ng1 = ng - 1;

  // set indices for sends to neighbors on SAME level
  // Formulae same as in LoadBoundaryBufferSameLevel() in src/bvals/fc/bvals_fc.cpp
  // for uniform grid: face-neighbors take care of the overlapping faces

  {auto &isame = buf.isame;       // indices of buffer for neighbor same level
  isame.bis = mb_indcs.is,                        isame.bie = mb_indcs.ie + 1;
  if (ox2 > 0) {
    isame.bjs = mb_indcs.je - (ng1 + maxjshift),     isame.bje = mb_indcs.je;
  } else {
    isame.bjs = mb_indcs.js,                         isame.bje = mb_indcs.js + (ng1 + maxjshift);
  }
  isame.bks = mb_indcs.ks,                           isame.bke = mb_indcs.ke + 1;
    
  buf.isame_ndatx1 = (isame.bie - isame.bis + 1);
  buf.isame_ndatx2 = (isame.bje - isame.bjs + 1);
  buf.isame_ndatx3 = (isame.bke - isame.bks + 1);
  }

  // set indices for sends to neighbors on COARSER level (matches recv from FINER)
  // Formulae same as in LoadBoundaryBufferToCoarser() in src/bvals/fc/bvals_fc.cpp
  // Identical to send indices for same level replacing is,ie,.. with cis,cie,...
  {auto &icoar = buf.icoar;   // indices of buffer for neighbor coarser level
  icoar.bis = mb_indcs.cis,                          icoar.bie = mb_indcs.cie + 1;
  if (ox2 > 0) {
    icoar.bjs = mb_indcs.cje - (ng1 + maxjshift),       icoar.bje = mb_indcs.cje;
  } else {
    icoar.bjs = mb_indcs.cjs,                           icoar.bje = mb_indcs.cjs + (ng1 + maxjshift);
  }
  icoar.bks = mb_indcs.cks,                             icoar.bke = mb_indcs.cke + 1;

  buf.icoar_ndatx1 = (icoar.bie - icoar.bis + 1);
  buf.icoar_ndatx2 = (icoar.bje - icoar.bjs + 1);
  buf.icoar_ndatx3 = (icoar.bke - icoar.bks + 1);
  }

  // set indices for sends to neighbors on FINER level (matches recv from COARSER)
  // Formulae same as in LoadBoundaryBufferToFiner() in src/bvals/fc/bvals_fc.cpp
  //
  // Subtle issue: shared face fields on edges of MeshBlock (B1 at [is,ie+1],
  // B2 at [js;je+1], B3 at [ks;ke+1]) are communicated, replacing values on coarse mesh
  // in target MeshBlock, but these values will only be used for prolongation.
  {auto &ifine = buf.ifine;    // indices of buffer for neighbor finer level
  int cnx1 = mb_indcs.cnx1;
  int cnx2 = mb_indcs.cnx2;
  int cnx3 = mb_indcs.cnx3;
  if (f1 == 1) {
    ifine.bis = mb_indcs.is + cnx1,                   ifine.bie = mb_indcs.ie + 1;
  } else {
    ifine.bis = mb_indcs.is,                          ifine.bie = mb_indcs.ie + 1 - cnx1;
  }
  if (ox2 > 0) {
    ifine.bjs = mb_indcs.je - (ng1 + maxjshift)/2,    ifine.bje = mb_indcs.je;
  } else {
    ifine.bjs = mb_indcs.js,                          ifine.bje = mb_indcs.js + (ng1 + maxjshift)/2;
  }
  ifine.bks = mb_indcs.ks,                            ifine.bke = mb_indcs.ke + 1;

  if (mb_indcs.nx3 > 1) {
    if (f2 == 1) {
      ifine.bks += cnx3;
    } else {
      ifine.bke -= cnx3;
    }
  }
  buf.ifine_ndatx1 = (ifine.bie - ifine.bis + 1);
  buf.ifine_ndatx2 = (ifine.bje - ifine.bjs + 1);
  buf.ifine_ndatx3 = (ifine.bke - ifine.bks + 1);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesFC::InitRecvIndices
//! \brief Calculates indices of cells into which receive buffers are unpacked for FC data
//! on same/coarser/finer levels, and for prolongation from coarse to fine.  Three sets of
//! indices are needed for each of the three components (x1f,x2f,x3f) of face-centered
//! fields.
//!
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0).  The arguments f1/2 are the coordinates
//! of subblocks within faces/edges (only relevant with SMR/AMR)
//!
//! NOTE: these indices are the same as for sending, except for the finer level

void OrbitalAdvectionFC::InitRecvIndices(OrbitalAdvectionBoundaryBuffer &buf,
                                           int ox1, int ox2, int ox3, int f1, int f2) {
  auto &mb_indcs  = pmy_pack->pmesh->mb_indcs;
  int ng  = mb_indcs.ng;
  int ng1 = ng - 1;

  // set indices for sends to neighbors on SAME level
  // Formulae same as in LoadBoundaryBufferSameLevel() in src/bvals/fc/bvals_fc.cpp
  // for uniform grid: face-neighbors take care of the overlapping faces

  {auto &isame = buf.isame;       // indices of buffer for neighbor same level
  isame.bis = mb_indcs.is,                        isame.bie = mb_indcs.ie + 1;
  if (ox2 > 0) {
    isame.bjs = mb_indcs.je - (ng1 + maxjshift),     isame.bje = mb_indcs.je;
  } else {
    isame.bjs = mb_indcs.js,                         isame.bje = mb_indcs.js + (ng1 + maxjshift);
  }
  isame.bks = mb_indcs.ks,                           isame.bke = mb_indcs.ke + 1;
    
  buf.isame_ndatx1 = (isame.bie - isame.bis + 1);
  buf.isame_ndatx2 = (isame.bje - isame.bjs + 1);
  buf.isame_ndatx3 = (isame.bke - isame.bks + 1);
  }

  // set indices for sends to neighbors on COARSER level (matches recv from FINER)
  // Formulae same as in LoadBoundaryBufferToCoarser() in src/bvals/fc/bvals_fc.cpp
  // Identical to send indices for same level replacing is,ie,.. with cis,cie,...
  {auto &icoar = buf.icoar;   // indices of buffer for neighbor coarser level
  icoar.bis = mb_indcs.cis,                          icoar.bie = mb_indcs.cie + 1;
  if (ox2 > 0) {
    icoar.bjs = mb_indcs.cje - (ng1 + maxjshift),       icoar.bje = mb_indcs.cje;
  } else {
    icoar.bjs = mb_indcs.cjs,                           icoar.bje = mb_indcs.cjs + (ng1 + maxjshift);
  }
  icoar.bks = mb_indcs.cks,                             icoar.bke = mb_indcs.cke + 1;

  buf.icoar_ndatx1 = (icoar.bie - icoar.bis + 1);
  buf.icoar_ndatx2 = (icoar.bje - icoar.bjs + 1);
  buf.icoar_ndatx3 = (icoar.bke - icoar.bks + 1);
  }

  // set indices for sends to neighbors on FINER level (matches recv from COARSER)
  // Formulae same as in LoadBoundaryBufferToFiner() in src/bvals/fc/bvals_fc.cpp
  //
  // Subtle issue: shared face fields on edges of MeshBlock (B1 at [is,ie+1],
  // B2 at [js;je+1], B3 at [ks;ke+1]) are communicated, replacing values on coarse mesh
  // in target MeshBlock, but these values will only be used for prolongation.
  {auto &ifine = buf.ifine;    // indices of buffer for neighbor finer level
  int cnx1 = mb_indcs.cnx1;
  int cnx2 = mb_indcs.cnx2;
  int cnx3 = mb_indcs.cnx3;
  if (f1 == 1) {
    ifine.bis = mb_indcs.is + cnx1,                    ifine.bie = mb_indcs.ie + 1;
  } else {
    ifine.bis = mb_indcs.is,                           ifine.bie = mb_indcs.ie - cnx1;
  }
  if (ox2 > 0) {
    ifine.bjs = mb_indcs.je - (ng1 + maxjshift)/2,    ifine.bje = mb_indcs.je;
  } else {
    ifine.bjs = mb_indcs.js,                          ifine.bje = mb_indcs.js + (ng1 + maxjshift)/2;
  }
  ifine.bks = mb_indcs.ks,                            ifine.bke = mb_indcs.ke + 1;
  if (mb_indcs.nx3 > 1) {
    if (f2 == 1) {
      ifine.bks += cnx3;
    } else {
      ifine.bke -= (cnx3+1);
    }
  }
  buf.ifine_ndatx1 = (ifine.bie - ifine.bis + 1);
  buf.ifine_ndatx2 = (ifine.bje - ifine.bjs + 1);
  buf.ifine_ndatx3 = (ifine.bke - ifine.bks + 1);
  }
  return;
}