//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file buffs_cc.cpp
//  \brief functions to allocate and initialize buffers for cell-centered variables

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "shearing_box/shearing_box.hpp"
#include "orbital_advection.hpp"

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionCC::InitSendIndices
//! \brief Calculates indices of cells used to pack buffers and send CC data for buffers
//! on same/coarser/finer levels. Only one set of indices is needed.
//!
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0).  The arguments f1/2 are the coordinates
//! of subblocks within faces/edges (only relevant with SMR/AMR)

void OrbitalAdvectionCC::InitSendIndices(OrbitalAdvectionBoundaryBuffer &buf,
                                          int ox1, int ox2, int ox3, int f1, int f2) {
  auto &mb_indcs  = pmy_pack->pmesh->mb_indcs;
  int ng  = mb_indcs.ng;
  int ng1 = ng - 1;

  // set indices for sends to neighbors on SAME level
  // Formulae taken from LoadBoundaryBufferSameLevel() in src/bvals/cc/bvals_cc.cpp
  // if ((f1 == 0) && (f2 == 0)) {  // this buffer used for same level (e.g. #8 (n=-1) & 12 (n=1))
  {auto &isame = buf.isame;    // indices of buffer for neighbor same level
  isame.bis = mb_indcs.is;
  isame.bie = mb_indcs.ie;
  isame.bjs = (ox2 > 0) ? (mb_indcs.je - (ng1 + maxjshift)) : mb_indcs.js;
  isame.bje = (ox2 < 0) ? (mb_indcs.js + (ng1 + maxjshift)) : mb_indcs.je;
  isame.bks = mb_indcs.ks;
  isame.bke = mb_indcs.ke;

    // store number of data elements in each direction
  buf.isame_ndatx1 = (isame.bie - isame.bis + 1);
  buf.isame_ndatx2 = (isame.bje - isame.bjs + 1);
  buf.isame_ndatx3 = (isame.bke - isame.bks + 1);
  }

  // set indices for sends to neighbors on COARSER level (matches recvs from FINER)
  // Formulae taken from LoadBoundaryBufferToCoarser() in src/bvals/cc/bvals_cc.cpp
  {auto &icoar = buf.icoar;  // indices of buffer for neighbor coarser level
  icoar.bis = mb_indcs.cis;
  icoar.bie = mb_indcs.cie;
  icoar.bjs = (ox2 > 0) ? (mb_indcs.cje - (ng1 + maxjshift)) : mb_indcs.cjs;
  icoar.bje = (ox2 < 0) ? (mb_indcs.cjs + (ng1 + maxjshift)) : mb_indcs.cje;
  icoar.bks = mb_indcs.cks;
  icoar.bke = mb_indcs.cke;

  // store number of data elements in each direction
  buf.icoar_ndatx1 = (icoar.bie - icoar.bis + 1);
  buf.icoar_ndatx2 = (icoar.bje - icoar.bjs + 1);
  buf.icoar_ndatx3 = (icoar.bke - icoar.bks + 1);
  }

  // set indices for sends to neighbors on FINER level (matches recvs from COARSER)
  // Formulae taken from LoadBoundaryBufferToFiner() src/bvals/cc/bvals_cc.cpp
  {auto &ifine = buf.ifine;  // indices of buffer for neighbor finer level
  ifine.bis = mb_indcs.is;
  ifine.bie = mb_indcs.ie;
  ifine.bjs = (ox2 > 0) ? (mb_indcs.je - (ng1 + maxjshift)/2) : mb_indcs.js;
  ifine.bje = (ox2 < 0) ? (mb_indcs.js + (ng1 + maxjshift)/2) : mb_indcs.je;
  ifine.bks = mb_indcs.ks;
  ifine.bke = mb_indcs.ke;
  // need to add internal edges on faces, and internal corners on edges
  if (f1 == 1) {
    ifine.bis += mb_indcs.cnx1;
  } else {
    ifine.bie -= mb_indcs.cnx1;
  }
  if (mb_indcs.nx3 > 1) {
    if (f2 == 1) {
      ifine.bks += mb_indcs.cnx3;
    } else {
      ifine.bke -= mb_indcs.cnx3;
    }
  }

  // store number of data elements in each direction
  buf.ifine_ndatx1 = (ifine.bie - ifine.bis + 1);
  buf.ifine_ndatx2 = (ifine.bje - ifine.bjs + 1);
  buf.ifine_ndatx3 = (ifine.bke - ifine.bks + 1);
  }
}





//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionCC::InitRecvIndices
//! \brief Calculates indices of cells used to pack buffers and send CC data for buffers
//! on same/coarser/finer levels. Only one set of indices is needed.
//!
//! The arguments ox1/2/3 are integer (+/- 1) offsets in each dir that specifies buffer
//! relative to center of MeshBlock (0,0,0).  The arguments f1/2 are the coordinates
//! of subblocks within faces/edges (only relevant with SMR/AMR)
//!
//! NOTE: For CC variables, the recv indices are identical to the send indices 

void OrbitalAdvectionCC::InitRecvIndices(OrbitalAdvectionBoundaryBuffer &buf,
                                          int ox1, int ox2, int ox3, int f1, int f2) {
  auto &mb_indcs  = pmy_pack->pmesh->mb_indcs;
  int ng  = mb_indcs.ng;
  int ng1 = ng - 1;

  // set indices for sends to neighbors on SAME level
  // Formulae taken from LoadBoundaryBufferSameLevel() in src/bvals/cc/bvals_cc.cpp
  // if ((f1 == 0) && (f2 == 0)) {  // this buffer used for same level (e.g. #8 (n=-1) & 12 (n=1))
  {auto &isame = buf.isame;    // indices of buffer for neighbor same level
  isame.bis = mb_indcs.is;
  isame.bie = mb_indcs.ie;
  isame.bjs = (ox2 > 0) ? (mb_indcs.je - (ng1 + maxjshift)) : mb_indcs.js;
  isame.bje = (ox2 < 0) ? (mb_indcs.js + (ng1 + maxjshift)) : mb_indcs.je;
  isame.bks = mb_indcs.ks;
  isame.bke = mb_indcs.ke;

    // store number of data elements in each direction
  buf.isame_ndatx1 = (isame.bie - isame.bis + 1);
  buf.isame_ndatx2 = (isame.bje - isame.bjs + 1);
  buf.isame_ndatx3 = (isame.bke - isame.bks + 1);
  }

  // set indices for sends to neighbors on COARSER level (matches recvs from FINER)
  // Formulae taken from LoadBoundaryBufferToCoarser() in src/bvals/cc/bvals_cc.cpp
  {auto &icoar = buf.icoar;  // indices of buffer for neighbor coarser level
  icoar.bis = mb_indcs.cis;
  icoar.bie = mb_indcs.cie;
  icoar.bjs = (ox2 > 0) ? (mb_indcs.cje - (ng1 + maxjshift)) : mb_indcs.cjs;
  icoar.bje = (ox2 < 0) ? (mb_indcs.cjs + (ng1 + maxjshift)) : mb_indcs.cje;
  icoar.bks = mb_indcs.cks;
  icoar.bke = mb_indcs.cke;

  // store number of data elements in each direction
  buf.icoar_ndatx1 = (icoar.bie - icoar.bis + 1);
  buf.icoar_ndatx2 = (icoar.bje - icoar.bjs + 1);
  buf.icoar_ndatx3 = (icoar.bke - icoar.bks + 1);
  }

  // set indices for sends to neighbors on FINER level (matches recvs from COARSER)
  // Formulae taken from LoadBoundaryBufferToFiner() src/bvals/cc/bvals_cc.cpp
  {auto &ifine = buf.ifine;  // indices of buffer for neighbor finer level
  ifine.bis = mb_indcs.is;
  ifine.bie = mb_indcs.ie;
  ifine.bjs = (ox2 > 0) ? (mb_indcs.je - (ng1 + maxjshift)/2) : mb_indcs.js;
  ifine.bje = (ox2 < 0) ? (mb_indcs.js + (ng1 + maxjshift)/2) : mb_indcs.je;
  ifine.bks = mb_indcs.ks;
  ifine.bke = mb_indcs.ke;
  // need to add internal edges on faces, and internal corners on edges
  if (f1 == 1) {
    ifine.bis += mb_indcs.cnx1;
  } else {
    ifine.bie -= mb_indcs.cnx1;
  }
  if (mb_indcs.nx3 > 1) {
    if (f2 == 1) {
      ifine.bks += mb_indcs.cnx3;
    } else {
      ifine.bke -= mb_indcs.cnx3;
    }
  }

  // store number of data elements in each direction
  buf.ifine_ndatx1 = (ifine.bie - ifine.bis + 1);
  buf.ifine_ndatx2 = (ifine.bje - ifine.bjs + 1);
  buf.ifine_ndatx3 = (ifine.bke - ifine.bks + 1);
  }
}