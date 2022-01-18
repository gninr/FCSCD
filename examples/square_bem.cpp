#include "force_calculation_bem.hpp"
#include "velocity_fields.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "continuous_space.hpp"
#include "discontinuous_space.hpp"
#include "parametrized_line.hpp"
#include "parametrized_mesh.hpp"
#include <Eigen/Dense>

int main() {
  std::cout << "Calculate force using BEM" << std::endl;
  std::cout << "####################################" << std::endl;

  std::ofstream out("square_bem.txt");
  out << "Calculate force using BEM" << std::endl;
  out << "####################################" << std::endl;
  // Gauss quadrature order
  unsigned order = 16;

  double epsilon1 = 1., epsilon2 = 5.;

  transmission_bem::G_CONST g([](Eigen::Vector2d x) { return 2. - x[0]; });

  transmission_bem::ETA_CONST eta([](Eigen::Vector2d x) { return 0; });

  auto dir_sel = [](Eigen::Vector2d x) {
    return (x[0] - 2. > -1e-7 || x[0] + 2. < 1e-7);
  };

  auto bdry_sel = [](Eigen::Vector2d x) {
    return (x[0] - 2. > -1e-7 || x[0] + 2. < 1e-7 ||
            x[1] - 2. > -1e-7 || x[1] + 2. < 1e-7);
  };

  parametricbem2d::ContinuousSpace<1> space_d;
  parametricbem2d::DiscontinuousSpace<0> space_n;

  // Inner square vertices
  Eigen::Vector2d NE(1., 1.);
  Eigen::Vector2d NW(0., 1.);
  Eigen::Vector2d SE(1., 0.);
  Eigen::Vector2d SW(0., 0.);
  // Inner square edges
  parametricbem2d::ParametrizedLine ir(NE, SE); // right
  parametricbem2d::ParametrizedLine it(NW, NE); // top
  parametricbem2d::ParametrizedLine il(SW, NW); // left
  parametricbem2d::ParametrizedLine ib(SE, SW); // bottom

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

  std::cout << "shape of inner dielectic: square" << std::endl;
  std::cout << "quadrature order: " << order << std::endl;
  std::cout << "epsilon1: " << epsilon1 << std::endl;
  std::cout << "epsilon2: " << epsilon2 << std::endl;
  std::cout << "------------------------------------" << std::endl;
  std::cout << std::setw(10) << "1/h"
            << std::setw(25) << "Pullback"
            << std::setw(25) << "Fx"
            << std::setw(25) << "Fy"
            << std::setw(25) << "Stress Tensor"
            << std::setw(25) << "Fx"
            << std::setw(25) << "Fy" << std::endl;

  out << "shape of inner dielectic: square" << std::endl;
  out << "quadrature order: " << order << std::endl;
  out << "epsilon1: " << epsilon1 << std::endl;
  out << "epsilon2: " << epsilon2 << std::endl;
  out << "------------------------------------" << std::endl;
  out << std::setw(10) << "1/h"
      << std::setw(25) << "Pullback"
      << std::setw(25) << "Fx"
      << std::setw(25) << "Fy"
      << std::setw(25) << "Stress Tensor"
      << std::setw(25) << "Fx"
      << std::setw(25) << "Fy" << std::endl;

  for (unsigned numpanels = 4; numpanels <= 256; numpanels *= 2) {
    unsigned temp = numpanels;

    // Panels for the edges of inner square
    parametricbem2d::PanelVector panels_ir(ir.split(temp));
    parametricbem2d::PanelVector panels_ib(ib.split(temp));
    parametricbem2d::PanelVector panels_il(il.split(temp));
    parametricbem2d::PanelVector panels_it(it.split(temp));

    // Panels for the edges of outer square
    parametricbem2d::PanelVector panels_or(Or.split(4*temp));
    parametricbem2d::PanelVector panels_ot(ot.split(4*temp));
    parametricbem2d::PanelVector panels_ol(ol.split(4*temp));
    parametricbem2d::PanelVector panels_ob(ob.split(4*temp));
    
    // Creating the ParametricMesh object
    parametricbem2d::PanelVector panels;

    panels.insert(panels.end(), panels_ir.begin(), panels_ir.end());
    panels.insert(panels.end(), panels_ib.begin(), panels_ib.end());
    panels.insert(panels.end(), panels_il.begin(), panels_il.end());
    panels.insert(panels.end(), panels_it.begin(), panels_it.end());

    panels.insert(panels.end(), panels_or.begin(), panels_or.end());
    panels.insert(panels.end(), panels_ot.begin(), panels_ot.end());
    panels.insert(panels.end(), panels_ol.begin(), panels_ol.end());
    panels.insert(panels.end(), panels_ob.begin(), panels_ob.end());

    parametricbem2d::ParametrizedMesh mesh(panels);

    std::cout.precision(std::numeric_limits<double>::digits10);
    std::cout << std::setw(10) << temp;
    
    out.precision(std::numeric_limits<double>::digits10);
    out << std::setw(10) << temp;
    
    transmission_bem::CalculateForce(mesh, space_d, space_n, bdry_sel, dir_sel,
        g, eta, epsilon1, epsilon2, order, out);
  }

  return 0;
}