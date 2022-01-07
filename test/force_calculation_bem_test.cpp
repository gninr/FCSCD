#include "force_calculation_bem.hpp"

#include <cmath>
#include <iostream>

#include "continuous_space.hpp"
#include "discontinuous_space.hpp"
#include "parametrized_circular_arc.hpp"
#include "parametrized_line.hpp"
#include "parametrized_mesh.hpp"
#include "velocity_fields.hpp"
#include <Eigen/Dense>

int main() {
  std::cout << "Calculate force using BEM" << std::endl;
  std::cout << "####################################" << std::endl;
  // Gauss quadrature order
  unsigned order = 8;
  std::cout << "Gauss Quadrature used with order = " << order << std::endl;

  // Inner square vertices
  Eigen::Vector2d NE(1, 1);
  Eigen::Vector2d NW(0, 1);
  Eigen::Vector2d SE(1, 0);
  Eigen::Vector2d SW(0, 0);
  // Inner square edges
  parametricbem2d::ParametrizedLine ir(NE, SE); // right
  parametricbem2d::ParametrizedLine it(NW, NE); // top
  parametricbem2d::ParametrizedLine il(SW, NW); // left
  parametricbem2d::ParametrizedLine ib(SE, SW); // bottom

  // Outer square vertices
  Eigen::Vector2d NEo(1.1, 1.1);
  Eigen::Vector2d NWo(-1.1, 1.1);
  Eigen::Vector2d SEo(1.1, -1.1);
  Eigen::Vector2d SWo(-1.1, -1.1);
  // Outer square edges
  parametricbem2d::ParametrizedLine Or(SEo, NEo); // right
  parametricbem2d::ParametrizedLine ot(NEo, NWo); // top
  parametricbem2d::ParametrizedLine ol(NWo, SWo); // left
  parametricbem2d::ParametrizedLine ob(SWo, SEo); // bottom
  
  unsigned nsplit = 16;

  // Panels for the edges of the inner square
  parametricbem2d::PanelVector panels_ir(ir.split(nsplit));
  parametricbem2d::PanelVector panels_ib(ib.split(nsplit));
  parametricbem2d::PanelVector panels_il(il.split(nsplit));
  parametricbem2d::PanelVector panels_it(it.split(nsplit));

  // Panels for the edges of outer square
  parametricbem2d::PanelVector panels_or(Or.split(nsplit));
  parametricbem2d::PanelVector panels_ot(ot.split(2*nsplit));
  parametricbem2d::PanelVector panels_ol(ol.split(2*nsplit));
  parametricbem2d::PanelVector panels_ob(ob.split(3*nsplit));

  // Creating the ParametricMesh object
  parametricbem2d::PanelVector panels;

  /*
  panels.insert(panels.end(), panels_ir.begin(), panels_ir.end());
  panels.insert(panels.end(), panels_ib.begin(), panels_ib.end());
  panels.insert(panels.end(), panels_il.begin(), panels_il.end());
  panels.insert(panels.end(), panels_it.begin(), panels_it.end());
  */
  Eigen::Vector2d center(0.5, 0.5);
  double r = 0.5;
  parametricbem2d::ParametrizedCircularArc icirc(center, r, 0., 2. * M_PI);
  parametricbem2d::PanelVector panels_i(icirc.split(4*nsplit));
  panels.insert(panels.end(), panels_i.begin(), panels_i.end());

  panels.insert(panels.end(), panels_or.begin(), panels_or.end());
  panels.insert(panels.end(), panels_ot.begin(), panels_ot.end());
  panels.insert(panels.end(), panels_ol.begin(), panels_ol.end());
  panels.insert(panels.end(), panels_ob.begin(), panels_ob.end());

  parametricbem2d::ParametrizedMesh mesh(panels);

  double epsilon1 = 1., epsilon2 = 100.;

  auto g = [](Eigen::Vector2d x) {
    return 1.1 - x[0];
  };

  auto eta = [](Eigen::Vector2d x) {
    return 0;
  };

  auto dir_sel = [](const Eigen::Vector2d& x) {
    return (x[0] - 1.1 > -1e-7 || x[0] + 1.1 < 1e-7);
  };

  parametricbem2d::ContinuousSpace<1> space_d;
  parametricbem2d::DiscontinuousSpace<0> space_n;
  NuConstant nu_x(Eigen::Vector2d(1., 0.));
  NuConstant nu_y(Eigen::Vector2d(0., 1.));

  Eigen::Vector2d force;
  force[0] = transmission_bem::CalculateForce(mesh, space_d, space_n, nu_x,
                 dir_sel, g, eta, epsilon1, epsilon2, order);
  force[1] = transmission_bem::CalculateForce(mesh, space_d, space_n, nu_y,
                 dir_sel, g, eta, epsilon1, epsilon2, order);
  std::cout << "force = (" << force[0] << ", " << force[1] << ")" << std::endl;

  return 0;
}