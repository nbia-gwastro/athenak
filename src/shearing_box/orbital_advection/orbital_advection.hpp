#ifndef SHEARING_BOX_ORBITAL_ADVECTION_HPP_
#define SHEARING_BOX_ORBITAL_ADVECTION_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file orbital_advection.hpp
//! \brief definitions for classes that implement orbital advection abstract base and
//! derived classes (for CC and FC variables).

#include "athena.hpp"
#include "parameter_input.hpp"
#include "shearing_box/shearing_box.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"



//----------------------------------------------------------------------------------------
//! \struct BufferIndcs
//! \brief indices for range of cells packed/unpacked into boundary buffers

struct OrbitalAdvectionBufferIndcs {
  int bis,bie,bjs,bje,bks,bke;  // start/end buffer ("b") indices in each dir
  OrbitalAdvectionBufferIndcs() :
    bis(0), bie(0), bjs(0), bje(0), bks(0), bke(0) {}
};

//----------------------------------------------------------------------------------------
//! \struct OrbitalAdvectionBoundaryBuffer
//! \brief container for storing boundary buffers. Used by the orbital advection methods for both CC and FC variables.
//! Basically a slightly simplified version of the MeshBoundaryBuffer struct.

struct OrbitalAdvectionBoundaryBuffer {
  // fixed-length-3 arrays used to store indices of each buffer for cell-centered vars, or
  // each component of a face-centered vector field ([0,1,2] --> [x1f, x2f, x3f]). For

  OrbitalAdvectionBufferIndcs isame;  // indices for pack/unpack when dest/src at same level
  OrbitalAdvectionBufferIndcs icoar;  // indices for pack/unpack when dest/src at coarser level
  OrbitalAdvectionBufferIndcs ifine;  // indices for pack/unpack when dest/src at finer level

  // Maximum number of data elements in each direction eg. (bie-bis+1) across 3 components of above
  int isame_ndatx1, isame_ndatx2, isame_ndatx3; 
  int icoar_ndatx1, icoar_ndatx2, icoar_ndatx3;
  int ifine_ndatx1, ifine_ndatx2, ifine_ndatx3;

  // View that store buffer data on device
  DvceArray5D<Real> vars;
#if MPI_PARALLEL_ENABLED
  // vector of length (number of MBs) to hold MPI request
  // Using STL vector causes problems with some GPU compilers, so just use plain C array
  MPI_Request *vars_req; //, *flux_req;
#endif

  // function to allocate memory for buffers for variables
  // Must only be called after BufferIndcs above are initialized
  void AllocateBuffers(int nmb, int nvars) {

    int nmax_x1 = std::max(isame_ndatx1, std::max(icoar_ndatx1, ifine_ndatx1) );
    int nmax_x2 = std::max(isame_ndatx2, std::max(icoar_ndatx2, ifine_ndatx2) );
    int nmax_x3 = std::max(isame_ndatx3, std::max(icoar_ndatx3, ifine_ndatx3) );

    Kokkos::realloc(vars, nmb, nvars, 2*nmax_x3, 2*nmax_x2, 2*nmax_x1);
  }
};

//----------------------------------------------------------------------------------------
//! \class OrbitalAdvection
//  \brief Abstract base class for orbital advection of CC and FC variables

class OrbitalAdvection {
 public:
  OrbitalAdvection(MeshBlockPack *ppack, ParameterInput *pin);
  ~OrbitalAdvection();

  // data
  int maxjshift;            // maximum integer shift of any cell in orbital advection
  Real qshear, omega0;      // Copies needed for all OA functions
  bool shearing_box_r_phi;  // true for 2D r-phi shearing box - NOT YET IMPLEMENTED FOR MHD

  // data buffers for orbital advection. Only two x2-faces communicate of maximum 8 meshblocks
  OrbitalAdvectionBoundaryBuffer sendbuf[8], recvbuf[8];

#if MPI_PARALLEL_ENABLED
  // unique MPI communicator for orbital advection
  MPI_Comm comm_orb_advect;
#endif

  // functions
  virtual void InitSendIndices(OrbitalAdvectionBoundaryBuffer &b,int o1,int o2,int o3,int f1,int f2)=0;
  virtual void InitRecvIndices(OrbitalAdvectionBoundaryBuffer &b,int o1,int o2,int o3,int f1,int f2)=0;

  void InitializeBuffers(const int nvar);

  TaskStatus InitRecv(const int nvar);
  TaskStatus ClearRecv();
  TaskStatus ClearSend();


 protected:
  // must use pointer to MBPack and not parent physics module since parent can be one of
  // many types (Hydro, MHD, Radiation, etc.)
  MeshBlockPack *pmy_pack;
};

//----------------------------------------------------------------------------------------
//! \class OrbitalAdvectionCC
//  \brief Derived class implementing orbital advection of CC variables

class OrbitalAdvectionCC : public OrbitalAdvection {
 public:
  OrbitalAdvectionCC(MeshBlockPack *ppack, ParameterInput *pin, int nvar);

  // functions to initialize indices for packing/unpacking buffers
  void InitSendIndices(OrbitalAdvectionBoundaryBuffer &b,int o1,int o2,int o3,int f1,int f2) override;
  void InitRecvIndices(OrbitalAdvectionBoundaryBuffer &b,int o1,int o2,int o3,int f1,int f2) override;

  // functions to communicate CC data with orbital advection
  TaskStatus PackAndSendCC(DvceArray5D<Real> &a);
  TaskStatus RecvAndUnpackCC(DvceArray5D<Real> &a, ReconstructionMethod rcon);

};

//----------------------------------------------------------------------------------------
//! \class OrbitalAdvectionFC
//  \brief Derived class implementing orbital advection of FC variables

class OrbitalAdvectionFC : public OrbitalAdvection {
 public:
  OrbitalAdvectionFC(MeshBlockPack *ppack, ParameterInput *pin);

  // functions to initialize indices for packing/unpacking buffers
  void InitSendIndices(OrbitalAdvectionBoundaryBuffer &b,int o1,int o2,int o3,int f1,int f2) override;
  void InitRecvIndices(OrbitalAdvectionBoundaryBuffer &b,int o1,int o2,int o3,int f1,int f2) override;

  // functions to communicate FC data with orbital advection plus final constrained transport step
  TaskStatus PackAndSendFC(DvceFaceFld4D<Real> &b);
  TaskStatus RecvAndUnpackFC(DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld_orb, ReconstructionMethod rcon);
  TaskStatus CT_OA(DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld_orb);

};

#endif // SHEARING_BOX_ORBITAL_ADVECTION_HPP_
