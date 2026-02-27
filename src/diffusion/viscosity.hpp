#ifndef DIFFUSION_VISCOSITY_HPP_
#define DIFFUSION_VISCOSITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file viscosity.hpp
//  \brief Contains data and functions that implement various formulations for
//  viscosity. Currently only Navier-Stokes (uniform, isotropic) shear viscosity
//  is implemented. TODO: add Braginskii viscosity

#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"

KOKKOS_INLINE_FUNCTION
Real AlphaSingle(Real z, Real y, Real x, Real z_bh, Real y_bh, Real x_bh, Real GM, Real nu0, Real omega0) {
  Real dx = x - x_bh;
  Real dy = y - y_bh;
  Real dz = z - z_bh;
  Real dr = std::sqrt(dx*dx+dy*dy+dz*dz);
  Real fac_ = 1/std::sqrt(std::pow(dr,-3)*GM + omega0**2);
  return nu0*fac_;
}

KOKKOS_INLINE_FUNCTION
Real AlphaBinary(Real z, Real y, Real x, Real z_bh, Real y_bh, Real x_bh, Real GM, Real nu0, Real omega0) {
  Real dx1 = x - x_bh;
  Real dy1 = y - y_bh;
  Real dz1 = z - z_bh;
  Real dr1 = std::sqrt(dx1*dx1+dy1*dy1+dz1*dz1);

  Real dx2 = x + x_bh;
  Real dy2 = y + y_bh;
  Real dz2 = z + z_bh;
  Real dr2 = std::sqrt(dx2*dx2+dy2*dy2+dz2*dz2);
  
  Real fac_ = 1/std::sqrt(std::pow(dr1,-3)*GM/2 + std::pow(dr2,-3)*GM/2 + omega0**2);
  return nu0*fac_;
}

struct AlphaViscModel{
  Real nu0;
  Real x_bh;
  Real y_bh;
  Real z_bh;
  Real GM;
  Real omega0;

  Real omega;
  Real a_b;

  int model_id;

  KOKKOS_INLINE_FUNCTION
  Real operator()(Real z, Real y, Real x) const {
    switch (model_id) {
      case 0: return AlphaSingle(z,y,x,z_bh,y_bh,x_bh,GM,nu0,omega0);
      case 1: return AlphaBinary(z,y,x,z_bh,y_bh,x_bh,GM,nu0,omega0);
      default: return nu0;
    }
  }
};

//----------------------------------------------------------------------------------------
//! \class Viscosity
//  \brief data and functions that implement viscosity in Hydro and MHD

class Viscosity {
 public:
  Viscosity(std::string block, MeshBlockPack *pp, ParameterInput *pin);
  ~Viscosity();

  // data
  Real dtnew;
  Real nu_iso;     // coefficient of isotropic kinematic shear viscosity

  bool alpha_visc;
  AlphaViscModel alpha_visc_model;

  // function to add viscous fluxes to Hydro and/or MHD fluxes
  void IsotropicViscousFlux(const DvceArray5D<Real> &w, const Real nu,
                            const EOS_Data &eos, DvceFaceFld5D<Real> &f);

  // function to add viscous fluxes to Hydro and/or MHD fluxes, with spatially dependent coefficient of isotropic kinematic shear viscosity
  void AlphaViscousFlux(const DvceArray5D<Real> &w, const Real nu,
                            const EOS_Data &eos, DvceFaceFld5D<Real> &f);

 private:
  MeshBlockPack* pmy_pack;
};

#endif // DIFFUSION_VISCOSITY_HPP_
