#include "transmission_bem.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include "parametrized_line.hpp"
#include "parametrized_mesh.hpp"
#include <Eigen/Dense>

int main() {
  std::string filename = "transmission_bem.txt";
  std::ofstream output(filename);
  std::cout << "Solve transmission problem using BEM" << std::endl;
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
  parametricbem2d::PanelVector panels_or(Or.split(2*nsplit));
  parametricbem2d::PanelVector panels_ot(ot.split(2*nsplit));
  parametricbem2d::PanelVector panels_ol(ol.split(2*nsplit));
  parametricbem2d::PanelVector panels_ob(ob.split(2*nsplit));

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
  transmission_bem::Solution sol = transmission_bem::solve(
      mesh, space_d, space_n, dir_sel, g, eta, epsilon1, epsilon2, order);

  std::cout << "\nu_i" << std::endl;
  std::cout << sol.u_i << std::endl;
  std::cout << "\npsi_i" << std::endl;
  std::cout << sol.psi_i << std::endl;
  std::cout << "\nu" << std::endl;
  std::cout << sol.u << std::endl;
  std::cout << "\npsi" << std::endl;
  std::cout << sol.psi << std::endl;

  return 0;
}
