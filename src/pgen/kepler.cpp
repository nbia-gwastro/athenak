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

#define NDIMS 3
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

  struct Sink{
    int  type;
    Real rsink;
    Real tsink;
  };

  struct PointMass{
    Real mass;
    Real x;
    Real y;
    Real z;
    Real vx;
    Real vy;
    Real vz;
  };

  struct DiskModel{
    Real sig0;
    Real pre0;
    Real rcav;
    Real fluff;
  };

  bool is_binary;
  Real binary_mass;
  Real mass_ratio;
  Real rsoft;
  Real rcav;
  Real fluff;
  Sink sink;
  Buffer buffer;
  DiskModel disk;
  void KeplerSourceTerms(Mesh *pm, const Real dt);
  void ComputeBinaryPositions(const Real time, PointMass *binary);
}

// ----------------------------------------------------------------------------
namespace {
  KOKKOS_INLINE_FUNCTION
  void InitializePrims(
    const Real *coord, 
    const Real central_mass, 
    const DiskModel &disk, 
    Real *prim) 
  {
    Real x = coord[0];
    Real y = coord[1];
    Real r = sqrt(x * x + y * y);
    Real rs = sqrt(r * r + rsoft * rsoft);
    Real vphi = sqrt(central_mass / rs);
    Real vx = -vphi * (y / r);
    Real vy =  vphi * (x / r);   
    Real fcav = disk.fluff + (1. - disk.fluff) * exp(-pow(r / disk.rcav, -4));
    prim[0] = disk.sig0; //* fcav;
    prim[1] = vx;
    prim[2] = vy;
    prim[3] = 0.0;
    prim[4] = disk.pre0;
    return;
  }
} // namespace

// ----------------------------------------------------------------------------
namespace {
  KOKKOS_INLINE_FUNCTION
  void PointMassGravity(
    const Real *coord, 
    const Real *prim, 
    const PointMass &p, 
    const Real dt, 
    Real *delta_cons) 
  {
    Real dx = coord[0] - p.x;
    Real dy = coord[1] - p.y;
    Real dz = coord[2] - p.z;
    Real dr = sqrt(dx * dx + dy * dy);
    Real fx = - p.mass * dx * pow(dr * dr + rsoft * rsoft, -3./2.);
    Real fy = - p.mass * dy * pow(dr * dr + rsoft * rsoft, -3./2.);
    delta_cons[1] += dt * prim[0] * fx;
    delta_cons[2] += dt * prim[0] * fy;
    delta_cons[4] += dt * (fx * prim[1] + fy * prim[2]);
    return;
  }

  KOKKOS_INLINE_FUNCTION
  void SinkSourceTerm(
    const Real *coord,
    const Real *prim,
    const PointMass &p,
    const Sink &sink,
    const Real dt,
    Real *delta_cons)
  {
    Real omega = 1;  // TODO: un-hardcode
    Real dx = coord[0] - p.x;
    Real dy = coord[1] - p.y;
    Real dz = coord[2] - p.z;
    Real dr = std::sqrt(dx * dx + dy * dy + dz * dz);
    Real dr_cyl = std::sqrt(dx * dx + dy * dy);
    Real tau = sink.tsink * omega * exp(-pow(dr / sink.rsink, 4.0));
    Real mdot = -tau * prim[0];
    switch(sink.type) {
      case 1: // standard / acceleration free
      {
        delta_cons[0] += mdot * dt;
        delta_cons[1] += mdot * prim[1] * dt;
        delta_cons[2] += mdot * prim[2] * dt;
        delta_cons[3] += mdot * prim[3] * dt;
        // delta_cons[4] += (mdot * eps + 0.5 * mdot * (vx * vx + vy * vy + vz * vz)) * dt
        break;
      }
      case 2: // torque-free
      {
        Real vpx = p.vx;
        Real vpy = p.vy;
        Real rhatx = dx / (dr_cyl + 1e-12);
        Real rhaty = dy / (dr_cyl + 1e-12);
        Real phatx = -dy / (dr_cyl + 1e-12);
        Real phaty =  dx / (dr_cyl + 1e-12);
        Real dvdotrhat = (prim[1] - vpx) * rhatx + (prim[2] - vpy) * rhaty;
        Real vxstar = dvdotrhat * rhatx + vpx;
        Real vystar = dvdotrhat * rhaty + vpy;
        Real dvphi = (prim[1] - vpx) * phatx + (prim[2] - vpy) * phaty;
        delta_cons[0] += mdot * dt;
        delta_cons[1] += mdot * vxstar * dt;
        delta_cons[2] += mdot * vystar * dt;
        delta_cons[3] += mdot * prim[3] * dt;
        // Check energy line from new paper
        // delta_cons[4] += (mdot * eps + 0.5 * mdot * (vxstar * vxstar + vystar * vystar) - 0.5 * mdot * dvphi * dvphi) * dt;
        break;
      }
      default:
      {
        break;
      }
    }
  }


  KOKKOS_INLINE_FUNCTION
  void BufferSourceTerm(
    const Real *coord, 
    const Real *prim, 
    const Buffer &buffer, 
    const Real central_mass, 
    const DiskModel &disk, 
    const Real dt, 
    const Real gamma, 
    Real *delta_cons) 
  {
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
        InitializePrims(coord, central_mass, disk, ptarget);
        Real den0 = ptarget[0];
        Real px0 = den0 * ptarget[1];
        Real py0 = den0 * ptarget[2];
        Real pz0 = den0 * ptarget[3];
        Real ke0 = 0.5 * (px0 * px0 + py0 * py0 + pz0 * pz0) / den0;
        Real en0 = ptarget[4] / (gamma - 1.0) + ke0;
        Real omega0 = sqrt(1.0 * pow(onset_radius, -3.0)); // GM = 1, TODO: generalize
        Real buffer_rate = driving_rate * omega0 * (rc - onset_radius) / (outer_radius - onset_radius);
        Real buffer_reduction = dt * buffer_rate < 1.0 ? dt * buffer_rate : 1.0;
        delta_cons[0] += buffer_reduction * (den0 - rho);
        delta_cons[1] += buffer_reduction * (px0 - px);
        delta_cons[2] += buffer_reduction * (py0 - py);
        delta_cons[3] += buffer_reduction * (pz0 - pz);
        // delta_cons[4] += buffer_reduction * (en0 - en);
      }
      return;
    }
  }

  // --------------------------------------------------------------------------
  void ComputeBinaryPositions(Real time, PointMass *binary) {
    Real omega = 1.0; // TODO: un-hardcode
    Real q = mass_ratio;
    Real m1 = binary_mass / (1.0 + q);
    Real m2 = q * m1;
    Real r1 = q   / (1.0 + q); // * a = 1
    Real r2 = 1.0 / (1.0 + q); // * a = 1
    PointMass p1, p2;
    p1.mass = m1;
    p2.mass = m2;
    p1.x = -r1 * cos(omega * time) * is_binary; // If a single black hole,
    p2.x =  r2 * cos(omega * time) * is_binary; // moves each component to
    p1.y = -r1 * sin(omega * time) * is_binary; // the origin
    p2.y =  r2 * sin(omega * time) * is_binary;
    p1.vx =  r1 * omega * sin(omega * time) * is_binary; 
    p2.vx = -r2 * omega * sin(omega * time) * is_binary; 
    p1.vy = -r1 * omega * cos(omega * time) * is_binary; 
    p2.vy =  r2 * omega * cos(omega * time) * is_binary;
    binary[0] = p1;
    binary[1] = p2;
    return;
  }

  void KeplerSourceTerms(Mesh *pm, const Real dt) {
    auto &indcs = pm->mb_indcs;
    int is = indcs.is, ie = indcs.ie;
    int js = indcs.js, je = indcs.je;
    int ks = indcs.ks, ke = indcs.ke;
    int nmb1 = pm->pmb_pack->nmb_thispack - 1;
    auto &size = pm->pmb_pack->pmb->mb_size;

    auto u0_ = pm->pmb_pack->phydro->u0;
    auto w0_ = pm->pmb_pack->phydro->w0;
    Real gamma = pm->pmb_pack->phydro->peos->eos_data.gamma;

    Real time = pm->time;
    PointMass binary[2];
    ComputeBinaryPositions(time, binary);

    const Sink sink_ = sink;
    const Buffer buffer_ = buffer;
    const DiskModel disk_ = disk;
    const Real central_mass_ = binary_mass;
    const bool is_ideal = pm->pmb_pack->phydro->peos->eos_data.is_ideal;
    par_for("central-potential", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
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
      
      Real cc[NDIMS] = {x, y, z};
      Real pc[NCONS] = {rho, vx, vy, vz, p};
      Real du[NCONS] = {0.0, 0.0, 0.0, 0.0, 0.0};

      PointMassGravity(cc, pc, binary[0], dt, du);
      PointMassGravity(cc, pc, binary[1], dt, du);
      SinkSourceTerm(cc, pc, binary[0], sink_, dt, du);
      SinkSourceTerm(cc, pc, binary[1], sink_, dt, du);
      BufferSourceTerm(cc, pc, buffer_, central_mass_, disk_, dt, gamma, du);
      
      u0_(m, IDN, k, j, i) += du[0];
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

  // Initialize binary
  is_binary = static_cast<int>(pin->GetOrAddReal("problem", "is_binary", 1));
  binary_mass = pin->GetOrAddReal("problem", "GM", 1.0);
  rsoft = pin->GetOrAddReal("problem", "softening_radius", 0.05);
  mass_ratio = pin->GetOrAddReal("problem", "mass_ratio", 1.0);

  // Set Disk model
  disk.sig0 = pin->GetOrAddReal("problem", "sigma0", 1.0);
  disk.pre0 = pin->GetOrAddReal("problem", "pressure0", 1e-6);
  disk.rcav = pin->GetOrAddReal("problem", "cavity_radius", 3.0);
  disk.fluff = pin->GetOrAddReal("problem", "fluff", 1e-5);

  // Read sink parameters
  sink.type = static_cast<int>(pin->GetOrAddReal("problem", "sink_type", 0));
  sink.rsink = pin->GetOrAddReal("problem", "sink_radius", 0.05);
  sink.tsink = pin->GetOrAddReal("problem", "sink_rate", 1.0);

  // Read buffer params
  buffer.is_enabled = static_cast<bool>(pin->GetOrAddReal("buffer", "buffer_is_enabled", 1.0));
  buffer.onset_width = pin->GetOrAddReal("buffer", "onset_width", 1.0);
  buffer.outer_radius = pin->GetReal("mesh", "x1max");
  buffer.onset_radius = buffer.outer_radius - buffer.onset_width;
  buffer.driving_rate = pin->GetOrAddReal("buffer", "driving_rate", 1000.0);

  // Index boilerplate
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // Initialize
  auto &u0_ = pmbp->phydro->u0;
  const DiskModel disk_ = disk;
  const Real central_mass_ = binary_mass;
  const bool is_ideal = pmbp->phydro->peos->eos_data.is_ideal;
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

    Real cc[NDIMS] = {x, y, z};

    Real prim0[NCONS];
    InitializePrims(cc, central_mass_, disk_, prim0);
    u0_(m, IDN, k, j, i) = prim0[0];
    u0_(m, IM1, k, j, i) = prim0[0] * prim0[1];
    u0_(m, IM2, k, j, i) = prim0[0] * prim0[2];
    u0_(m, IM3, k, j, i) = 0.0;
    if (is_ideal) {
      u0_(m, IEN, k, j, i) = prim0[0] * prim0[4] / (pmbp->phydro->peos->eos_data.gamma - 1.0) + 0.5 * prim0[0] * (prim0[1] * prim0[1] + prim0[2] * prim0[2]);
    }
  });

  // Register source functions
  user_srcs_func = &KeplerSourceTerms;

  return;
}
