#ifndef OUTPUTS_OUTPUTS_HPP_
#define OUTPUTS_OUTPUTS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file outputs.hpp
//  \brief provides classes to handle ALL types of data output

#include <string>
#include <vector>

#include "Kokkos_ScatterView.hpp"

#include "athena.hpp"
#include "io_wrapper.hpp"

#define NHISTORY_VARIABLES 20
#if NHISTORY_VARIABLES > NREDUCTION_VARIABLES
    #error NHISTORY > NREDUCTION in outputs.hpp
#endif

#define NOUTPUT_CHOICES 158
// choices for output variables used in <ouput> blocks in input file
// TO ADD MORE CHOICES:
//   - add more strings to array below, change NOUTPUT_CHOICES above appropriately
//   - add code to load new variables in BaseOutputType constructor
//   - may need to change index limits that test whether physics is defined for
//     requested output variable near start of BaseOutputType constructor (TODO)
static const char *var_choice[NOUTPUT_CHOICES] = {
  // hydro variables (0-13)
  "hydro_u_d", "hydro_u_m1", "hydro_u_m2", "hydro_u_m3", "hydro_u_e",     "hydro_u",
  "hydro_w_d", "hydro_w_vx", "hydro_w_vy", "hydro_w_vz", "hydro_w_e",     "hydro_w",
  "hydro_u_s", "hydro_w_s",
  // hydro derived variables (14-15)
  "hydro_wz",   "hydro_w2",
  // MHD variables (16-43)
  "mhd_u_d",   "mhd_u_m1",   "mhd_u_m2",   "mhd_u_m3",   "mhd_u_e",       "mhd_u",
  "mhd_w_d",   "mhd_w_vx",   "mhd_w_vy",   "mhd_w_vz",   "mhd_w_e",       "mhd_w",
  "mhd_u_s",   "mhd_w_s",    "mhd_wz",     "mhd_w2",
  "mhd_bcc1",  "mhd_bcc2",   "mhd_bcc3",   "mhd_bcc",    "mhd_u_bcc",     "mhd_w_bcc",
  "mhd_bfc1",  "mhd_bfc2",   "mhd_bfc3",   "mhd_bfc",    "mhd_u_bfc",     "mhd_w_bfc",

  // MHD derived variables (44-51)
  "mhd_jz",    "mhd_j2",     "mhd_curv",   "mhd_k_jxb",  "mhd_curv_perp", "mhd_bmag",
  "mhd_divb", "mhd_jcon",
  // useful for coarsened binary output (52-53)
  "hydro_sgs", "mhd_sgs",
  // dynamo wavenumber scales (54)
  "mhd_dynamo_ks",
  // turbulence (55)
  "turb_force",
  // radiation (56-72, 73-92)
  "rad_coord",     "rad_fluid",      "rad_coord_fluid",
  "rad_hydro_u_d", "rad_hydro_u_m1", "rad_hydro_u_m2", "rad_hydro_u_m3", "rad_hydro_u_e",
  "rad_hydro_u",   "rad_hydro_w_d",  "rad_hydro_w_vx", "rad_hydro_w_vy", "rad_hydro_w_vz",
  "rad_hydro_w_e", "rad_hydro_w",    "rad_hydro_u_s",  "rad_hydro_w_s",
  "rad_mhd_u_d",   "rad_mhd_u_m1",   "rad_mhd_u_m2",   "rad_mhd_u_m3",   "rad_mhd_u_e",
  "rad_mhd_u",     "rad_mhd_w_d",    "rad_mhd_w_vx",   "rad_mhd_w_vy",   "rad_mhd_w_vz",
  "rad_mhd_w_e",   "rad_mhd_w",      "rad_mhd_u_s",    "rad_mhd_w_s",    "rad_mhd_bcc1",
  "rad_mhd_bcc2",  "rad_mhd_bcc3",   "rad_mhd_bcc",    "rad_mhd_u_bcc",  "rad_mhd_w_bcc",

  // ADM (93-110)
  "adm_gxx", "adm_gxy", "adm_gxz", "adm_gyy", "adm_gyz", "adm_gzz",
  "adm_Kxx", "adm_Kxy", "adm_Kxz", "adm_Kyy", "adm_Kyz", "adm_Kzz",
  "adm_psi4",
  "adm_alpha", "adm_betax", "adm_betay", "adm_betaz",
  "adm",

  // Z4c (111-133)
  "z4c_chi",
  "z4c_gxx", "z4c_gxy", "z4c_gxz", "z4c_gyy", "z4c_gyz", "z4c_gzz",
  "z4c_Khat",
  "z4c_Axx", "z4c_Axy", "z4c_Axz", "z4c_Ayy", "z4c_Ayz", "z4c_Azz",
  "z4c_Gamx", "z4c_Gamy", "z4c_Gamz",
  "z4c_Theta",
  "z4c_alpha",
  "z4c_betax", "z4c_betay", "z4c_betaz",
  "z4c",

  // Weyl (134-136)
  "weyl_rpsi4", "weyl_ipsi4",
  "weyl",

  // ADM constraints (137-144)
  "con_C",
  "con_H",
  "con_M",
  "con_Z",
  "con_Mx", "con_My", "con_Mz",
  "con",

  // Tmunu (145-155)
  "tmunu_Sxx", "tmunu_Sxy", "tmunu_Sxz", "tmunu_Syy", "tmunu_Syz", "tmunu_Szz",
  "tmunu_E",
  "tmunu_Sx", "tmunu_Sy", "tmunu_Sz",
  "tmunu",

  // Particles (156-157)
  "prtcl_all", "prtcl_d"
};


// forward declarations
class Mesh;
class ParameterInput;

//----------------------------------------------------------------------------------------
//! \struct OutputParameters
//  \brief container for parameters read from <output> block in the input file by the
//  Outputs constructor.

struct OutputParameters {
  int block_number;
  std::string block_name;
  Real last_time, dt;
  int dcycle;                 // enables outputs every 'dcycle'
  int file_number;
  std::string file_basename;
  std::string file_type;
  std::string file_id;
  std::string variable;
  bool include_gzs;
  int gid;
  bool slice1, slice2, slice3;
  Real slice_x1, slice_x2, slice_x3;
  bool user_hist_only;
  std::string data_format;
  bool contains_derived=false;
  // DBF parameters for coarsened binary:
  // cannot be less than 2 and must be a power of 2 and
  // cannot be greater than shortest meshblock dimension
  int coarsen_factor;
  bool compute_moments; // if true then will compute
  // <q>, <q^2>, <q^3>, <q^4> for each variable q
  // DBF parameters for PDF:
  // number of derived variables, index of current derived variable
  int n_derived=0, i_derived=0;
  std::string variable_2; // DBF: for 2d PDFs
  Real bin_min, bin_max;
  Real bin2_min, bin2_max;
  int nbin=0, nbin2=0;
  bool logscale=true, logscale2=true;
  bool mass_weighted=false;
  bool single_file_per_rank=false; // DBF: parameter for single file per rank
};

//----------------------------------------------------------------------------------------
//! \struct OutputVariableInfo
//  \brief  container for various properties of each output variable

struct OutputVariableInfo {
  std::string label;             // "name" of variable
  int data_index;                // index of variable in device array
  DvceArray5D<Real> *data_ptr;   // ptr to device array containing variable
  // constructor(s)
  OutputVariableInfo(std::string lab, int indx, DvceArray5D<Real> *ptr) :
    label(lab), data_index(indx), data_ptr(ptr) {}
};

//----------------------------------------------------------------------------------------
//! \struct OutputMeshBlockInfo
//  \brief  container for various properties of each output MeshBlock

struct OutputMeshBlockInfo {
  int mb_gid;                        // gid of output MB
  int ois, oie, ojs, oje, oks, oke;  // start/end indices of data to be output on MB
  Real x1min, x1max;  // physical X1 size of output MB
  Real x2min, x2max;  // physical X2 size of output MB
  Real x3min, x3max;  // physical X3 size of output MB
  // constructor
  OutputMeshBlockInfo(int id, int is, int ie, int js, int je, int ks, int ke,
    Real x1min, Real x1max, Real x2min, Real x2max, Real x3min, Real x3max) :
    mb_gid(id), ois(is), oie(ie), ojs(js), oje(je), oks(ks), oke(ke),
    x1min(x1min), x1max(x1max), x2min(x2min), x2max(x2max), x3min(x3min), x3max(x3max) {}
};

//----------------------------------------------------------------------------------------
//! \struct HistoryData
//  \brief  container for history data for different physics modules

struct HistoryData {
  int nhist;
  PhysicsModule physics;
  std::string label[NHISTORY_VARIABLES];
  Real hdata[NHISTORY_VARIABLES];
  bool header_written;
  // constructor
  explicit HistoryData(PhysicsModule name) : physics(name), header_written(false) {}
};

//----------------------------------------------------------------------------------------
//! \struct TrackedParticleData
//! \brief data (tag, pos, vel) output for tracked particles

struct TrackedParticleData {
  int tag;
  Real x,y,z;
  Real vx,vy,vz;
};

//----------------------------------------------------------------------------------------
// \brief abstract base class for different output types (modes/formats); node in
//        std::list of BaseTypeOutput created & stored in the Outputs class

class BaseTypeOutput {
 public:
  BaseTypeOutput(ParameterInput *pin, Mesh *pm, OutputParameters oparams);
  virtual ~BaseTypeOutput() = default;
  // copy constructor and assignment operator
  BaseTypeOutput(const BaseTypeOutput& copy_other) = default;
  BaseTypeOutput& operator=(const BaseTypeOutput& copy_other) = default;
  // move constructor and assignment operator
  BaseTypeOutput(BaseTypeOutput&&) = default;
  BaseTypeOutput& operator=(BaseTypeOutput&&) = default;

  // data
  OutputParameters out_params;   // params read from <output> block for this type
  DvceArray5D<Real> derived_var; // array to store output variables computed from u0/b0

  // function which computes derived output variables like vorticity and current density
  void ComputeDerivedVariable(std::string name, Mesh *pm);

  // virtual functions may be over-ridden in derived classes
  virtual void LoadOutputData(Mesh *pm);
  virtual void WriteOutputFile(Mesh *pm, ParameterInput *pin) = 0;

  // Functions to detect big endian machine, and to byte-swap 32-bit words.  The vtk
  // legacy format requires data to be stored as big-endian.
  int IsBigEndian() {
    std::int32_t n = 1;
    char *ep = reinterpret_cast<char *>(&n);
    return (*ep == 0); // Returns 1 (true) on a big endian machine
  }
  inline void Swap4Bytes(void *vdat) {
    char tmp, *dat = static_cast<char *>(vdat);
    tmp = dat[0];  dat[0] = dat[3];  dat[3] = tmp;
    tmp = dat[1];  dat[1] = dat[2];  dat[2] = tmp;
  }

 protected:
  // CC output data on host with dims (n,m,k,j,i) except
  // for restarts, where dims are (m,n,k,j,i)
  HostArray5D<Real> outarray;
  HostArray5D<Real> outarray_hyd, outarray_mhd, outarray_rad,
                    outarray_force, outarray_z4c, outarray_adm;
  HostFaceFld4D<Real> outfield;  // FC output field on host
  std::vector<int> noutmbs;   // with MPI, number of output MBs across all ranks
  int noutmbs_min;            // with MPI, minimum number of output MBs across all ranks
  int noutmbs_max;            // with MPI, maximum number of output MBs across all ranks

  // Following vector will be of length (# output MeshBlocks)
  // With slicing, this may not be same as # of MeshBlocks in calculation
  std::vector<OutputMeshBlockInfo> outmbs;

  // Following vector will be of length (# output variables)
  std::vector<OutputVariableInfo> outvars;
};


//----------------------------------------------------------------------------------------
//! \class FormattedTableOutput
//  \brief derived BaseTypeOutput class for formatted table (tabular) data

class FormattedTableOutput : public BaseTypeOutput {
 public:
  FormattedTableOutput(ParameterInput *pin, Mesh *pm, OutputParameters oparams);
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
};

//----------------------------------------------------------------------------------------
//! \class HistoryOutput
//  \brief derived BaseTypeOutput class for history data

class HistoryOutput : public BaseTypeOutput {
 public:
  HistoryOutput(ParameterInput *pin, Mesh *pm, OutputParameters oparams);

  // vector of length [# of physics modules] containing hdata arrays
  std::vector<HistoryData> hist_data;

  void LoadOutputData(Mesh *pm) override;
  void LoadHydroHistoryData(HistoryData *pdata, Mesh *pm);
  void LoadMHDHistoryData(HistoryData *pdata, Mesh *pm);
  void LoadZ4cHistoryData(HistoryData *pdata, Mesh *pm);
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
};

//----------------------------------------------------------------------------------------
//! \class CoarsenedBinaryOutput
//  \brief derived BaseTypeOutput class for coarsened binary grid data

class CoarsenedBinaryOutput : public BaseTypeOutput {
 public:
  CoarsenedBinaryOutput(ParameterInput *pin, Mesh *pm, OutputParameters oparams);

  // void CoarsenVariable(const DvceArray3D<Real>& full_data,
  //                            DvceArray3D<Real>& coarsen_data,
  //                            const int coarsen_factor);
  void LoadOutputData(Mesh *pm) override;
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
};

//----------------------------------------------------------------------------------------
//! \struct PDFData
//  \brief  container for PDF data

struct PDFData {
  int pdf_dimension;
  int nbin, nbin2;
  Kokkos::View<Real*> bins;
  Kokkos::View<Real*> bins2;
  bool bins_written;
  // if logscale is true then this step is the log10 of the step size
  Real step_size, step_size2;
  bool mass_weighted;
  bool logscale, logscale2;

  DvceArray2D<Real> result_; // resulting histogram
  Kokkos::Experimental::ScatterView<Real **, LayoutWrapper> scatter_result;

  PDFData(int dim, int nbinVal, int nbin2Val)
    : pdf_dimension(dim), nbin(nbinVal), nbin2(nbin2Val),
      bins("bins", nbin + 1), bins2("bins2", nbin2 + 1),
      bins_written(false), mass_weighted(false), logscale(false), logscale2(false) {
  }
};

//----------------------------------------------------------------------------------------
//! \class PDFOutput
//  \brief derived BaseTypeOutput class for pdf data

class PDFOutput : public BaseTypeOutput {
 public:
  PDFOutput(ParameterInput *pin, Mesh *pm, OutputParameters oparams);

  PDFData pdf_data;

  void LoadOutputData(Mesh *pm) override;
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
};

//----------------------------------------------------------------------------------------
//! \class MeshVTKOutput
//  \brief derived BaseTypeOutput class for mesh data in VTK (legacy) format

class MeshVTKOutput : public BaseTypeOutput {
 public:
  MeshVTKOutput(ParameterInput *pin, Mesh *pm, OutputParameters oparams);
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
};

//----------------------------------------------------------------------------------------
//! \class ParticleVTKOutput
//  \brief derived BaseTypeOutput class for particle data in VTK (legacy) format

class ParticleVTKOutput : public BaseTypeOutput {
 public:
  ParticleVTKOutput(ParameterInput *pin, Mesh *pm, OutputParameters oparams);
  void LoadOutputData(Mesh *pm) override;
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
 protected:
  int npout_thisrank;
  int npout_total;
  HostArray2D<Real> outpart_rdata;
  HostArray2D<int>  outpart_idata;
};

//----------------------------------------------------------------------------------------
//! \class MeshBinaryOutput
//  \brief derived BaseTypeOutput class for binary mesh data (nbf format in pegasus++)
class MeshBinaryOutput : public BaseTypeOutput {
 public:
  MeshBinaryOutput(ParameterInput *pin, Mesh *pm, OutputParameters oparams);
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
};

//----------------------------------------------------------------------------------------
//! \class BoundaryFaceOutput
//  \brief derived BaseTypeOutput class that dumps the first ghost-cell layer on a
//   single physical face per output cadence into a self-contained per-snapshot
//   BCTABLE-format file. A separate concatenator stitches the snapshots into a
//   single time series file consumable by the BCTableReader-style problem
//   generators. Assumes MHD and root-level boundary blocks only.

class BoundaryFaceOutput : public BaseTypeOutput {
 public:
  BoundaryFaceOutput(ParameterInput *pin, Mesh *pm, OutputParameters oparams);
  void LoadOutputData(Mesh *pm) override;
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;

 private:
  struct FaceBlock {
    int      m_local;           // local meshblock index in this pack
    uint32_t lx_a, lx_b;        // root-level lloc in the two perp dirs
  };

  void rebuild_face_blocks(Mesh *pm);

  int face_id_{-1};            // 0..5 ix1,ox1,ix2,ox2,ix3,ox3
  int dtype_code_{1};          // 1=float64, 2=float32
  int elem_size_{8};
  uint32_t N1_{0}, N2_{0};     // global face raster dims
  uint32_t window_a_{0};       // per-block extent in faster perp dim (=nx_a+2*ng)
  uint32_t window_b_{0};       // per-block extent in slower perp dim (=nx_b+2*ng)
  int      cached_nmb_thispack_{-1};
  // Last chunk-subdir we created on disk, to avoid an mkdir on every cadence.
  // chunk = file_number / BCFACE_CHUNK_SIZE; -1 = none created yet.
  int      last_chunk_dir_{-1};

  std::vector<FaceBlock> face_blocks_;

  // Device-side packing buffers. The kernel fills slab_d_ from u0/b0 once per
  // cadence, then we mirror only this small array to host for the row writes.
  // Sized to (nblocks, 7, window_b, window_a). Persistent across cadences;
  // realloc'd only when nblocks changes.
  DvceArray4D<Real>      slab_d_;
  DvceArray1D<int>       m_local_d_;       // (nblocks,) device map b_idx -> m_local
  HostArray1D<int>       m_local_h_;       // (nblocks,) host scratch for above
  // File destination displacements for this rank's contribution, in oldtype
  // units relative to the BCTABLE payload offset. One entry per (block,
  // variable, slow-row); each entry points at the start of a window_a-long run.
  // Stored in monotonically-non-decreasing order (required by MPI file views).
  std::vector<int>       disps_;
  // Sorted-and-permuted source buffer matching disps_ order. Sized to
  // nblocks*nvar*window_b*window_a*elem_size_ bytes; reinterpret as float or
  // double per dtype_code_. Avoids per-cadence allocation.
  std::vector<char>      permuted_buf_;
};

//----------------------------------------------------------------------------------------
//! \class RestartOutput
//  \brief derived BaseTypeOutput class for restarts

class RestartOutput : public BaseTypeOutput {
 public:
  RestartOutput(ParameterInput *pin, Mesh *pm, OutputParameters oparams);
  void LoadOutputData(Mesh *pm) override;
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
};

// Forward declaration
class CartesianGrid;

//----------------------------------------------------------------------------------------
//! \class CartesianGridOutput
//  \brief derived BaseTypeOutput class for output on a Cartesian grid
class CartesianGridOutput : public BaseTypeOutput {
  struct MetaData {
    int cycle;
    float time;
    float center[3];
    float extent[3];
    int numpoints[3];
    bool is_cheb;
    int noutvars;
  };
 public:
  CartesianGridOutput(ParameterInput *pin, Mesh *pm, OutputParameters oparams);
  ~CartesianGridOutput();
  //! Interpolate the data on the Cartesian grid and handle MPI communication
  void LoadOutputData(Mesh *pm) override;
  //! Write the data to file
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
 private:
  CartesianGrid *pcart;
  MetaData md;
};

// Forward declaration
class SphericalSurface;

//----------------------------------------------------------------------------------------
//! \class SphericalGridOutput
//  \brief derived BaseTypeOutput class for output on a Cartesian grid
class SphericalSurfaceOutput : public BaseTypeOutput {
 public:
  SphericalSurfaceOutput(ParameterInput *pin, Mesh *pm, OutputParameters oparams);
  ~SphericalSurfaceOutput();
  //! Interpolate the data on the Cartesian grid and handle MPI communication
  void LoadOutputData(Mesh *pm) override;
  //! Write the data to file
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
 private:
  SphericalSurface *psurf;
};
//----------------------------------------------------------------------------------------
//! \class EventLogOutput
//  \brief derived BaseTypeOutput class for event counter data

class EventLogOutput : public BaseTypeOutput {
 public:
  EventLogOutput(ParameterInput *pin, Mesh *pm, OutputParameters oparams);

  // various flags to denote output status
  bool header_written=false;
  bool no_output=true;

  void LoadOutputData(Mesh *pm) override;
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
};

//----------------------------------------------------------------------------------------
//! \class TrackedParticleOutput
//  \brief derived BaseTypeOutput class for tracked particle data in binary format

class TrackedParticleOutput : public BaseTypeOutput {
 public:
  TrackedParticleOutput(ParameterInput *pin, Mesh *pm, OutputParameters oparams);
  void LoadOutputData(Mesh *pm) override;
  void WriteOutputFile(Mesh *pm, ParameterInput *pin) override;
 protected:
  int ntrack;           // total number of tracked particles across all ranks
  int ntrack_thisrank;  // number of tracked particles this rank (guess)
  int npout;            // number of tracked particles to be written this rank
  bool header_written;
  std::vector<int> npout_eachrank;
  HostArray1D<TrackedParticleData> outpart;
};

//----------------------------------------------------------------------------------------
//! \class Outputs
//  \brief root class for all Athena++ outputs. Provides a std::vector of BaseTypeOutputs,
//   with each element representing one mode/format of output to be made.

class Outputs {
 public:
  Outputs(ParameterInput *pin, Mesh *pm);
  ~Outputs();

  // use vector of pointers to BaseTypeOutputs since it is an abstract base class
  std::vector<BaseTypeOutput*> pout_list;
};

#endif // OUTPUTS_OUTPUTS_HPP_
