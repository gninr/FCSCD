#include "force_calculation_bem.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "continuous_space.hpp"
#include "discontinuous_space.hpp"
#include "parametrized_circular_arc.hpp"
#include "parametrized_line.hpp"
#include "parametrized_mesh.hpp"
#include "velocity_fields.hpp"
#include <Eigen/Dense>

int main(int argc, char *argv[]) {
  std::cout << "Calculate force using BEM" << std::endl;
  std::cout << "####################################" << std::endl;

  std::ofstream out("bem_test.txt");
  out << "Calculate force using BEM" << std::endl;
  out << "####################################" << std::endl;

  // Gauss quadrature order
  unsigned order = 8;
  std::cout << "Gauss Quadrature used with order = " << order << std::endl;
  out << "Gauss Quadrature used with order = " << order << std::endl;

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
  
  unsigned nsplit = 8;

  // Panels for the edges of outer square
  parametricbem2d::PanelVector panels_or(Or.split(4*nsplit));
  parametricbem2d::PanelVector panels_ot(ot.split(4*nsplit));
  parametricbem2d::PanelVector panels_ol(ol.split(4*nsplit));
  parametricbem2d::PanelVector panels_ob(ob.split(4*nsplit));

  // Creating the ParametricMesh object
  parametricbem2d::PanelVector panels;

  // Inner edges
  // Sqaure
  if (argv[1] == std::string("0")) {
    std::cout << "Square" << std::endl;
    
    // Inner square vertices
    Eigen::Vector2d NE(0.5, 0.5);
    Eigen::Vector2d NW(-0.5, 0.5);
    Eigen::Vector2d SE(0.5, -0.5);
    Eigen::Vector2d SW(-0.5, -0.5);
    // Inner square edges
    parametricbem2d::ParametrizedLine ir(NE, SE); // right
    parametricbem2d::ParametrizedLine it(NW, NE); // top
    parametricbem2d::ParametrizedLine il(SW, NW); // left
    parametricbem2d::ParametrizedLine ib(SE, SW); // bottom

    parametricbem2d::PanelVector panels_ir(ir.split(nsplit));
    parametricbem2d::PanelVector panels_ib(ib.split(nsplit));
    parametricbem2d::PanelVector panels_il(il.split(nsplit));
    parametricbem2d::PanelVector panels_it(it.split(nsplit));

    panels.insert(panels.end(), panels_ir.begin(), panels_ir.end());
    panels.insert(panels.end(), panels_ib.begin(), panels_ib.end());
    panels.insert(panels.end(), panels_il.begin(), panels_il.end());
    panels.insert(panels.end(), panels_it.begin(), panels_it.end());
  }

  // Circle
  else {
    Eigen::Vector2d center;
    double r = 0.5;

    // No symmetry
    if (argv[1] == std::string("1")) {
      std::cout << "Circle without symmetry" << std::endl;
      center << 0.5, 0.5;
    }

    // Symmetric about x axis
    else if (argv[1] == std::string("2")) {
      std::cout << "Circle symmetric about x axis" << std::endl;
      center << 0.5, 0.0;
    }
    
    // Symmetric about y axis
    else if (argv[1] == std::string("3")) {
      std::cout << "Circle symmetric about y axis" << std::endl;
      center << 0.0, 0.5;
    }

    // Symmetric about origin
    else if (argv[1] == std::string("4")) {
      std::cout << "Circle symmetric about origin" << std::endl;
      center << 0.0, 0.0;
    }
    
    parametricbem2d::ParametrizedCircularArc icirc(center, r, 0., 2. * M_PI);
    parametricbem2d::PanelVector panels_i(icirc.split(4*nsplit));
    panels.insert(panels.end(), panels_i.begin(), panels_i.end());
  }

  panels.insert(panels.end(), panels_or.begin(), panels_or.end());
  panels.insert(panels.end(), panels_ot.begin(), panels_ot.end());
  panels.insert(panels.end(), panels_ol.begin(), panels_ol.end());
  panels.insert(panels.end(), panels_ob.begin(), panels_ob.end());

  parametricbem2d::ParametrizedMesh mesh(panels);

  double epsilon1 = 1., epsilon2 = 100.;

  transmission_bem::G_CONST g([](Eigen::Vector2d x) { return 2. - x[0]; });

  transmission_bem::ETA_CONST eta([](Eigen::Vector2d x) { return 0; });

  auto bdry_sel = [](Eigen::Vector2d x) {
    return (x[0] - 2. > -1e-7 || x[0] + 2. < 1e-7 ||
            x[1] - 2. > -1e-7 || x[1] + 2. < 1e-7);
  };

  auto dir_sel = [](Eigen::Vector2d x) {
    return (x[0] - 2. > -1e-7 || x[0] + 2. < 1e-7);
  };

  parametricbem2d::ContinuousSpace<1> space_d;
  parametricbem2d::DiscontinuousSpace<0> space_n;

  double force = transmission_bem::CalculateForce(mesh, space_d, space_n,
      bdry_sel, dir_sel, g, eta, epsilon1, epsilon2, order, out);
  std::cout << "magnitude of force: " << force << std::endl;

  return 0;
}