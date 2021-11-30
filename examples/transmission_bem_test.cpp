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
  Eigen::Vector2d NEo(3, 3);
  Eigen::Vector2d NWo(-3, 3);
  Eigen::Vector2d SEo(3, -3);
  Eigen::Vector2d SWo(-3, -3);
  // Outer square edges
  parametricbem2d::ParametrizedLine Or(SEo, NEo); // right
  parametricbem2d::ParametrizedLine ot(NEo, NWo); // top
  parametricbem2d::ParametrizedLine ol(NWo, SWo); // left
  parametricbem2d::ParametrizedLine ob(SWo, SEo); // bottom
  
  unsigned numpanels = 2;
  unsigned nd = numpanels * 12;

  // Panels for the edges of the inner square
  parametricbem2d::PanelVector panels_ir(ir.split(numpanels));
  parametricbem2d::PanelVector panels_ib(ib.split(numpanels));
  parametricbem2d::PanelVector panels_il(il.split(numpanels));
  parametricbem2d::PanelVector panels_it(it.split(numpanels));

  // Panels for the edges of outer square
  parametricbem2d::PanelVector panels_or(Or.split(6*numpanels));
  parametricbem2d::PanelVector panels_ot(ot.split(6*numpanels));
  parametricbem2d::PanelVector panels_ol(ol.split(6*numpanels));
  parametricbem2d::PanelVector panels_ob(ob.split(6*numpanels));

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

  auto g = [&](double x1, double x2) {
    return 3. - x1;
  };
  auto eta = [&](double x1, double x2) {
    return 0.;
  };
  double epsilon1 = 1., epsilon2 = 2.;

  Eigen::VectorXd sol = transmission_bem::solve(mesh, nd, g, eta,
                                                epsilon1, epsilon2, order);

  unsigned ni = mesh.getSplit();
  unsigned nn = mesh.getNumPanels() - nd - ni;
  std::cout << "\nu_i" << std::endl;
  std::cout << sol.segment(0, ni) << std::endl;
  std::cout << "\npsi_i" << std::endl;
  std::cout << sol.segment(ni, ni) << std::endl;
  std::cout << "\nu" << std::endl;
  std::cout << sol.segment(ni*2, nd-2) << std::endl;
  std::cout << "\npsi" << std::endl;
  std::cout << sol.segment(ni*2+nd-2, nn) << std::endl;

  return 0;
}
