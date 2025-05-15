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

namespace {
  Real GM;
  Real rs;
  void CentralPotential(Mesh *pm, const Real dt);
}

namespace {
  //----------------------------------------------------------------------------------------
  //! \fn
  //  \brief Gravitational acceleration from a central point mass
  void CentralPotential(Mesh *pm, const Real dt) {
    auto &indcs = pm->mb_indcs;
    int is = indcs.is, ie = indcs.ie;
    int js = indcs.js, je = indcs.je;
    int ks = indcs.ks, ke = indcs.ke;
    int nmb1 = pm->pmb_pack->nmb_thispack - 1;
    auto &size = pm->pmb_pack->pmb->mb_size;

    // DvceArray5D<Real> u0_, w0_;
    auto u0_ = pm->pmb_pack->phydro->u0;
    auto w0_ = pm->pmb_pack->phydro->w0;

    // Place holders for generalization
    Real xi = 0;
    Real yi = 0;
    Real zi = 0;

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

      // distance from BH
      Real dx = x - xi;
      Real dy = y - yi;
      Real dz = z - zi;
      Real dr = std::sqrt(dx * dx + dy * dy);
      Real fx = - GM * x * pow(dr * dr + rs * rs, -3./2.);
      Real fy = - GM * y * pow(dr * dr + rs * rs, -3./2.);

      Real rho = w0_(m, IDN, k, j, i);
      Real vx  = w0_(m, IVX, k, j, i);
      Real vy  = w0_(m, IVY, k, j, i);
      u0_(m, IM1, k, j, i) += dt * rho * fx;
      u0_(m, IM2, k, j, i) += dt * rho * fy;
      if (pm->pmb_pack->phydro->peos->eos_data.is_ideal) {
        u0_(m, IEN, k, j, i) = dt * (fx * vx + fy * vy);
      }
    });

    return;
  }
} // namespace


//----------------------------------------------------------------------------------------
//! \fn
//  \brief Problem Generator for Keplerian Disk in Cartesian coordinates
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  // Read parameters
  GM = pin->GetOrAddReal("problem", "GM", 1.0);
  rs = pin->GetOrAddReal("problem", "rsoft", 0.05);
  Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  Real p0   = pin->GetOrAddReal("problem", "p0", 1e-6);

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  auto &u0_ = pmbp->phydro->u0;

  // Initialize disk using Kokkos parallel loop
  par_for("pgen_keplerian_disk", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x = CellCenterX(i - is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real y = CellCenterX(j - js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real z = CellCenterX(k - ks, nx3, x3min, x3max);

    Real r = sqrt(x * x + y * y);
    Real rsoft = sqrt(r * r + rs * rs);
    Real vphi = sqrt(GM / rsoft);
    Real vx = -vphi * (y / r);
    Real vy =  vphi * (x / r);

    Real rcav = 3.0;
    Real delta = 1e-5;
    Real fcav = delta + (1. - delta) * exp(-pow(r / rcav, -4));

    Real rout = 7.0;
    Real fout = 1. - (delta + (1. - delta) * exp(-pow(r / rout, -12)));

    Real rho = rho0 * fout;
    u0_(m, IDN, k, j, i) = rho;
    u0_(m, IM1, k, j, i) = rho * vx;
    u0_(m, IM2, k, j, i) = rho * vy;
    u0_(m, IM3, k, j, i) = 0.0;
    if (pmbp->phydro->peos->eos_data.is_ideal) {
      u0_(m, IEN, k, j, i) = rho * p0 / (pmbp->phydro->peos->eos_data.gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy);
    }
  });

  // Register gravity source function
  user_srcs_func = &CentralPotential;

  return;
}