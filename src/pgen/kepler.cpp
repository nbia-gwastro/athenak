//========================================================================================
// AthenaK astrophysical plasma code
// Keplerian Disk Problem Generator with Modular Gravity Source
//========================================================================================

#include <iostream>
#include <sstream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen.hpp"

#define NDIM 3
#define NCONS 5

// ----------------------------------------------------------------------------
namespace {
  struct Buffer{
    bool is_enabled;
    Real onset_width;
    Real onset_radius;
    Real outer_radius;
    Real driving_rate;
  };

  struct PointMass{
    Real mass;
    Real x;
    Real y;
    Real z;
    Real vx;
    Real vy;
    Real vz;
    Real rs;
  };

  struct DiskModel{
    Real rho0;
    Real p0;
  };

  Real GM;
  Real rs;
  Real rho0;
  Real p0;
  Buffer buffer;
  PointMass central_mass;
  DiskModel disk;

  void UserSourceTerms(Mesh *pm, const Real dt);

  // Also might need to capture gloval variables into KOKKOS_LAMBDAs with Kokkos::View
}

// ----------------------------------------------------------------------------
namespace {
  KOKKOS_INLINE_FUNCTION
  void InitializePrims(const Real *coord, const PointMass &p, const DiskModel &disk, Real *prim) {
    Real x = coord[0];
    Real y = coord[1];
    Real r = sqrt(x * x + y * y);
    Real rsoft = sqrt(r * r + p.rs * p.rs);
    Real vphi = sqrt(p.mass / rsoft);
    Real vx = -vphi * (y / r);
    Real vy =  vphi * (x / r);

    Real fluff = 1e-5;
    Real rcav = 3.0;    // TODO: Make these things input parameters
    Real rout = 7.0;
    Real fcav = fluff + (1. - fluff) * exp(-pow(r / rcav, -4));
    Real fout = 1. - (fluff + (1. - fluff) * exp(-pow(r / rout, -12)));

    prim[0] = disk.rho0; // * fout;
    prim[1] = vx;
    prim[2] = vy;
    prim[3] = 0.0;
    prim[4] = disk.p0;
    return;
  }

  // void InitializeCons(const Coordinate &coord, Conserved *cons, Real gamma) {
  //   Real r = sqrt(coord.x * coord.x + coord.y * coord.y);
  //   Real rsoft = sqrt(r * r + rs * rs);
  //   Real vphi = sqrt(GM / rsoft);
  //   Real vx = -vphi * (coord.y / r);
  //   Real vy =  vphi * (coord.x / r);

  //   Real fluff = 1e-5;
  //   Real rcav = 3.0;    // TODO: Make these things input parameters
  //   Real rout = 7.0;
  //   Real fcav = fluff + (1. - fluff) * exp(-pow(r / rcav, -4));
  //   Real fout = 1. - (fluff + (1. - fluff) * exp(-pow(r / rout, -12)));

  //   Real rho = rho0 * fout;
  //   cons.rho = rho;
  //   cons.px = rho * vx;
  //   cons.py = rho * vy;
  //   cons.pz = 0.0;
  //   cons.en = rho * p0 / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy);
  //   return;
  // }
} // namespace

// ----------------------------------------------------------------------------
namespace {
  KOKKOS_INLINE_FUNCTION
  void PointMassGravity(const Real *coord, const Real *prim, const PointMass &p, const Real dt, Real *delta_cons) {
    // Place holders for generalization
    Real dx = coord[0] - p.x;
    Real dy = coord[1] - p.y;
    Real dz = coord[2] - p.z;
    Real dr = std::sqrt(dx * dx + dy * dy);
    Real fx = - p.mass * coord[0] * pow(dr * dr + p.rs * p.rs, -3./2.);
    Real fy = - p.mass * coord[1] * pow(dr * dr + p.rs * p.rs, -3./2.);
    delta_cons[1] += dt * prim[0] * fx;
    delta_cons[2] += dt * prim[0] * fy;
    delta_cons[4] += dt * (fx * prim[1] + fy * prim[2]);
    return;
  }

  KOKKOS_INLINE_FUNCTION
  void BufferSourceTerm(const Real *coord, const Real *prim, const Buffer &buffer, const PointMass &p, const DiskModel &disk, const Real dt, const Real gamma, Real *delta_cons) {
    if (buffer.is_enabled)
    {
      Real x = coord[0];
      Real y = coord[1];
      Real rc = sqrt(x * x + y * y);
      Real driving_rate = buffer.driving_rate;
      Real outer_radius = buffer.outer_radius;
      Real onset_width = buffer.onset_width;
      Real onset_radius = buffer.onset_radius;

      Real rho = prim[0];
      Real px = rho * prim[1];
      Real py = rho * prim[2];
      Real pz = rho * prim[3];
      Real ke = 0.5 * (px * px + py * py + pz * pz) / rho;
      Real en = prim[4] / (gamma - 1.0) + ke;
      if (rc > onset_radius)
      {
        Real ptarget[NCONS];
        InitializePrims(coord, p, disk, ptarget);
        Real den0 = ptarget[0];
        Real px0 = ptarget[0] * ptarget[1];
        Real py0 = ptarget[0] * ptarget[2];
        Real pz0 = ptarget[0] * ptarget[3];
        Real ke0 = 0.5 * (px0 * px0 + py0 * py0 + pz0 * pz0) / den0;
        Real en0 = ptarget[4] / (gamma - 1.0) + ke0;
        Real omega0 = sqrt(1.0 * pow(onset_radius, -3.0)); // GM = 1, TODO: generalize
        Real buffer_rate = driving_rate * omega0 * (rc - onset_radius) / (outer_radius - onset_radius);
        Real buffer_reduction = dt * buffer_rate < 1.0 ? dt * buffer_rate : 1.0;
        delta_cons[0] += buffer_reduction * (den0 - rho);
        delta_cons[1] += buffer_reduction * (px0 - px);
        delta_cons[2] += buffer_reduction * (py0 - py);
        delta_cons[3] += buffer_reduction * (pz0 - pz);
        delta_cons[4] += buffer_reduction * (en0 - en);
      }
      return;
    }
  }

  void UserSourceTerms(Mesh *pm, const Real dt) {
    auto &indcs = pm->mb_indcs;
    int is = indcs.is, ie = indcs.ie;
    int js = indcs.js, je = indcs.je;
    int ks = indcs.ks, ke = indcs.ke;
    int nmb1 = pm->pmb_pack->nmb_thispack - 1;
    auto &size = pm->pmb_pack->pmb->mb_size;

    // DvceArray5D<Real> u0_, w0_;
    auto u0_ = pm->pmb_pack->phydro->u0;
    auto w0_ = pm->pmb_pack->phydro->w0;
    Real gamma = pm->pmb_pack->phydro->peos->eos_data.gamma;

    const Buffer buffer_ = buffer;
    const DiskModel disk_ = disk;
    const PointMass central_mass_ = central_mass;
    par_for("central-potential", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      bool is_ideal = pm->pmb_pack->phydro->peos->eos_data.is_ideal;
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx1 = indcs.nx1;
      int nx2 = indcs.nx2;
      int nx3 = indcs.nx3;
      Real x = CellCenterX(i - is, nx1, x1min, x1max);
      Real y = CellCenterX(j - js, nx2, x2min, x2max);
      Real z = CellCenterX(k - ks, nx3, x3min, x3max);
      Real rho = w0_(m, IDN, k, j, i);
      Real vx = w0_(m, IVX, k, j, i);
      Real vy = w0_(m, IVY, k, j, i);
      Real vz = w0_(m, IVZ, k, j, i);
      Real p = is_ideal ? w0_(m, IEN, k, j, i) : 0.0;
      
      Real cc[NDIM] = {x, y, z};// cc[0]=x; cc[1]=y; cc[2]=z;
      Real pc[NCONS] = {rho, vx, vy, vz, p}; //pc[0]=rho; pc[1]=vx; pc[2]=vy; pc[3]=vz; pc[4]=p;
      Real du[NCONS] = {0.0, 0.0, 0.0, 0.0, 0.0}; //du[0]=0.0; du[1]=0.0; du[2]=0.0; du[3]=0.0; du[4]=0.0;

      PointMassGravity(cc, pc, central_mass_, dt, du); //TODO : generalize to binary
      BufferSourceTerm(cc, pc, buffer_, central_mass_, disk_, dt, gamma, du);
      // Other sources
      // - SinkSourceTerm
      // 

      u0_(m, IM1, k, j, i) += du[1];
      u0_(m, IM2, k, j, i) += du[2];
      u0_(m, IM3, k, j, i) += du[3];
      if (is_ideal) {
        u0_(m, IEN, k, j, i) += du[4];
      }
    });

    return;
  }
} // namespace


//----------------------------------------------------------------------------------------
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  // Read parameters
  GM = pin->GetOrAddReal("problem", "GM", 1.0);
  rs = pin->GetOrAddReal("problem", "rsoft", 0.05);
  rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  p0 = pin->GetOrAddReal("problem", "p0", 1e-6);

  // Set PointMass(es)
  central_mass.mass = GM;
  central_mass.rs = rs;
  central_mass.x = 0.0;
  central_mass.y = 0.0;
  central_mass.z = 0.0;
  central_mass.vx = 0.0;
  central_mass.vy = 0.0;
  central_mass.vz = 0.0;

  // Set Disk model
  disk.rho0 = rho0;
  disk.p0 = p0;

  // Read buffer params
  // Formally check if there is a buffer sections in input?
  // if (exists) -> is_enabled = True
  buffer.is_enabled = static_cast<bool>(pin->GetOrAddReal("buffer", "buffer_is_enabled", 0.0));
  buffer.onset_width = pin->GetOrAddReal("buffer", "onset_width", 1.0);
  buffer.outer_radius = pin->GetReal("mesh", "x1max"); // generalized to square domain for now
  buffer.onset_radius = buffer.outer_radius - buffer.onset_width;
  buffer.driving_rate = pin->GetOrAddReal("buffer", "driving_rate", 1000.0);

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  auto &u0_ = pmbp->phydro->u0;

  // Initialize disk using Kokkos parallel loop
  const DiskModel disk_ = disk;
  const PointMass central_mass_ = central_mass;
  par_for("pgen_keplerian_disk", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx1 = indcs.nx1;
    int nx2 = indcs.nx2;
    int nx3 = indcs.nx3;
    Real x = CellCenterX(i - is, nx1, x1min, x1max);
    Real y = CellCenterX(j - js, nx2, x2min, x2max);
    Real z = CellCenterX(k - ks, nx3, x3min, x3max);

    Real cc[NDIM] = {x, y, z}; //cc[0] = x; cc[1] = y; cc[2] = z;
    
    // InitializeCons(cc, &cons0, pmbp->phydro->peos->eos_data.gamma);

    // u0_(m, IDN, k, j, i) = cons0.rho;
    // u0_(m, IM1, k, j, i) = cons0.px;
    // u0_(m, IM2, k, j, i) = cons0.py;
    // u0_(m, IM3, k, j, i) = cons0.pz;
    // if (pmbp->phydro->peos->eos_data.is_ideal) {
    //   u0_(m, IEN, k, j, i) = cons.en;
    // }

    Real prim0[NCONS];
    InitializePrims(cc, central_mass_, disk_, prim0);
    u0_(m, IDN, k, j, i) = prim0[0];
    u0_(m, IM1, k, j, i) = prim0[0] * prim0[1];
    u0_(m, IM2, k, j, i) = prim0[0] * prim0[2];
    u0_(m, IM3, k, j, i) = 0.0;
    if (pmbp->phydro->peos->eos_data.is_ideal) {
      u0_(m, IEN, k, j, i) = prim0[0] * prim0[4] / (pmbp->phydro->peos->eos_data.gamma - 1.0) + 0.5 * prim0[0] * (prim0[1] * prim0[1] + prim0[2] * prim0[2]);
    }
  });

  // Register source function
  user_srcs_func = &UserSourceTerms;

  return;
}