#include "force_calculation_bem.hpp"
#include "velocity_fields.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "continuous_space.hpp"
#include "discontinuous_space.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include "parametrized_mesh.hpp"
#include <Eigen/Dense>

Eigen::Vector2d exkite(double t) {
  return Eigen::Vector2d(0.3+.35 * std::cos(t) + .1625 * std::cos(2 * t),
                         0.5+.35 * std::sin(t));
}

Eigen::Vector2d exdkite(double t) {
  return Eigen::Vector2d(-.35 * std::sin(t) - 2 * .1625 * std::sin(2 * t),
                         .35 * std::cos(t));
}

Eigen::VectorXd get_kite_params(unsigned N) {
  /*
  // Calculating the length of the kite
  unsigned N_length = 500; // No. of points used in the calculation
  Eigen::VectorXd pts_length = Eigen::VectorXd::LinSpaced(N_length,0,2*M_PI);
  double L = 0;
  for (unsigned i = 0 ; i < N_length-1 ; ++i)
    L += (exkite(pts_length(i)) - exkite(pts_length(i+1))).norm();

  std::cout << "found length of the kite: " << L << std::endl;
  */
  double L = 2.46756;
  // Solving the equation for Phi using explicit timestepping
  unsigned k = 20; // multiplicity factor
  double h = L/N/k; // step size
  Eigen::VectorXd phi_full = Eigen::VectorXd::Constant(N*k,0);
  Eigen::VectorXd phi = Eigen::VectorXd::Constant(N,0);
  for (unsigned i = 1 ; i < N*k ; ++i)
    phi_full(i) = phi_full(i-1) + h /(exdkite(phi_full(i-1))).norm();

  for (unsigned i = 0 ; i < N ; ++i)
    phi(i) = phi_full(i*k);

  return phi;
}

int main() {
  std::cout << "Calculate force using BEM" << std::endl;
  std::cout << "####################################" << std::endl;

  std::ofstream out("kite_bem.txt");
  out << "Calculate force using BEM" << std::endl;
  out << "####################################" << std::endl;
  // Gauss quadrature order
  unsigned order = 16;

  double epsilon1 = 1., epsilon2 = 5.;

  auto g = [](Eigen::Vector2d x) {
    return 2. - x[0];
  };

  auto eta = [](Eigen::Vector2d x) {
    return 0;
  };

  auto dir_sel = [](Eigen::Vector2d x) {
    return (x[0] - 2. > -1e-7 || x[0] + 2. < 1e-7);
  };

  auto bdry_sel = [](Eigen::Vector2d x) {
    return (x[0] - 2. > -1e-7 || x[0] + 2. < 1e-7 ||
            x[1] - 2. > -1e-7 || x[1] + 2. < 1e-7);
  };

  parametricbem2d::ContinuousSpace<1> space_d;
  parametricbem2d::DiscontinuousSpace<0> space_n;
  transmission_bem::NuConstant nu_x(Eigen::Vector2d(1., 0.), bdry_sel);
  transmission_bem::NuConstant nu_y(Eigen::Vector2d(0., 1.), bdry_sel);

  // Outer square vertices
  Eigen::Vector2d NEo(2., 2.);
  Eigen::Vector2d NWo(-2., 2.);
  Eigen::Vector2d SEo(2., -2.);
  Eigen::Vector2d SWo(-2., -2.);
  // Outer square edges
  parametricbem2d::ParametrizedLine Or(SEo, NEo); // right
  parametricbem2d::ParametrizedLine ot(NEo, NWo); // top
  parametricbem2d::ParametrizedLine ol(NWo, SWo); // left
  parametricbem2d::ParametrizedLine ob(SWo, SEo); // bottom

  std::cout << "shape of inner dielectic: kite" << std::endl;
  std::cout << "quadrature order: " << order << std::endl;
  std::cout << "epsilon1: " << epsilon1 << std::endl;
  std::cout << "epsilon2: " << epsilon2 << std::endl;
  std::cout << "------------------------------------" << std::endl;
  std::cout << std::setw(10) << "1/h"
            << std::setw(25) << "Fx"
            << std::setw(25) << "Fy" << std::endl;

  out << "shape of inner dielectic: kite" << std::endl;
  out << "quadrature order: " << order << std::endl;
  out << "epsilon1: " << epsilon1 << std::endl;
  out << "epsilon2: " << epsilon2 << std::endl;
  out << "------------------------------------" << std::endl;
  out << std::setw(10) << "1/h"
      << std::setw(25) << "Fx"
      << std::setw(25) << "Fy" << std::endl;

  // numpanels = Number of panels per unit
  for (unsigned numpanels = 1; numpanels <= 512; numpanels *= 2) {
    unsigned temp = numpanels;

    parametricbem2d::PanelVector panels_kite;
    double lkite = 2.46756; // length of kite
    unsigned N = ceil(lkite * numpanels);
    Eigen::VectorXd meshpts = get_kite_params(N);
    Eigen::VectorXd tempp(N+1);
    tempp << meshpts, 2*M_PI;
    tempp = -tempp + Eigen::VectorXd::Constant(N+1,2*M_PI); // clockwise
    //std::cout << "temp: " << tempp.transpose() << std::endl;
    // Defining the kite domain
    Eigen::MatrixXd cos_list_o(2, 2);
    cos_list_o << .35, .1625, 0, 0;
    Eigen::MatrixXd sin_list_o(2, 2);
    sin_list_o << 0, 0, .35, 0;
    for (unsigned i = 0 ; i < N ; ++i) {
      panels_kite.push_back(
          std::make_shared<parametricbem2d::ParametrizedFourierSum>(
              Eigen::Vector2d(0.3, 0.5), cos_list_o, sin_list_o, 
              tempp(i), tempp(i+1)));
    }

    // Meshing the sqkite equivalently in the parameter mesh
    // Creating the ParametricMesh object
    parametricbem2d::PanelVector panels;

    // Panels for the edges of outer square
    parametricbem2d::PanelVector panels_or(Or.split(4*temp));
    parametricbem2d::PanelVector panels_ot(ot.split(4*temp));
    parametricbem2d::PanelVector panels_ol(ol.split(4*temp));
    parametricbem2d::PanelVector panels_ob(ob.split(4*temp));

    panels.insert(panels.end(), panels_kite.begin(), panels_kite.end());
    panels.insert(panels.end(), panels_or.begin(), panels_or.end());
    panels.insert(panels.end(), panels_ot.begin(), panels_ot.end());
    panels.insert(panels.end(), panels_ol.begin(), panels_ol.end());
    panels.insert(panels.end(), panels_ob.begin(), panels_ob.end());

    parametricbem2d::ParametrizedMesh mesh(panels);

    double Fx = transmission_bem::CalculateForce(mesh, space_d, space_n, nu_x,
                    dir_sel, g, eta, epsilon1, epsilon2, order);
    double Fy = transmission_bem::CalculateForce(mesh, space_d, space_n, nu_y,
                    dir_sel, g, eta, epsilon1, epsilon2, order);
    
    std::cout.precision(std::numeric_limits<double>::digits10);
    std::cout << std::setw(10) << temp
              << std::setw(25) << Fx
              << std::setw(25) << Fy << std::endl;

    out.precision(std::numeric_limits<double>::digits10);
    out << std::setw(10) << temp
        << std::setw(25) << Fx
        << std::setw(25) << Fy << std::endl;
  }

  return 0;
}