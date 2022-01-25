#include "force_calculation_fem.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <string>

#include <lf/fe/fe.h>
#include <lf/io/io.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/uscalfe/uscalfe.h>

int main(int argc, char *argv[]) {
  using size_type = lf::base::size_type;
  
  double epsilon1 = 1.;
  double epsilon2 = 5.;

  std::cout << "Calculate force using FEM" << std::endl;
  std::cout << "####################################" << std::endl;

  std::string filename;
  if (argc > 1) {
    epsilon2 = atof(argv[1]);
    filename = "kite_fem" + std::string(argv[1]) + ".txt";
  }
  else {
    filename = "kite_fem5.txt";
  }
  
  std::ofstream out(filename);
  out << "Calculate force using FEM" << std::endl;
  out << "####################################" << std::endl;

  auto g = [](Eigen::Vector2d x) {
    return 2. - x[0];
  };

  auto eta = [](Eigen::Vector2d x) {
    return 0;
  };

  auto dir_sel = [](Eigen::Vector2d x) {
    return (x[0] - 2. > -1e-7 || x[0] + 2. < 1e-7);
  };

  double r_in = 1.5;
  double r_out = 1.9;
  double r_diff = r_out - r_in;
  auto grad_w = [&](Eigen::Vector2d x) {
    double c = x.norm();
    Eigen::Vector2d res;
    if (c < r_in || c > r_out) {
      res = Eigen::Vector2d::Zero();
    }
    else {
      res = - 0.5 * M_PI / c / r_diff * sin(M_PI * (c - r_in) / r_diff) * x;
    }
    return res;
  };

  std::cout << "shape of inner dielectic: kite" << std::endl;
  std::cout << "epsilon1: " << epsilon1 << std::endl;
  std::cout << "epsilon2: " << epsilon2 << std::endl;
  std::cout << "------------------------------------" << std::endl;
  std::cout << std::setw(10) << "1/h"
            << std::setw(25) << "Volume Formula"
            << std::setw(25) << "Fx"
            << std::setw(25) << "Fy" 
            << std::setw(25) << "Stress Tensor"
            << std::setw(25) << "Fx"
            << std::setw(25) << "Fy" << std::endl;

  out << "shape of inner dielectic: kite" << std::endl;
  out << "epsilon1: " << epsilon1 << std::endl;
  out << "epsilon2: " << epsilon2 << std::endl;
  out << "------------------------------------" << std::endl;
  out << std::setw(10) << "1/h"
      << std::setw(25) << "Volume Formula"
      << std::setw(25) << "Fx"
      << std::setw(25) << "Fy" 
      << std::setw(25) << "Stress Tensor"
      << std::setw(25) << "Fx"
      << std::setw(25) << "Fy" << std::endl;
  
  std::shared_ptr<const lf::mesh::Mesh> mesh_p;

  for (int level = 4; level <= 9; ++level) {
    // Load mesh
    std::string mesh_file = "meshes/kite_sq" + 
                            std::to_string(level) + ".msh";
    auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
    const lf::io::GmshReader reader(std::move(mesh_factory), mesh_file);
    mesh_p = reader.mesh();
    auto fe_space =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_p);
    
    // Determine whether an edge belongs to inner boundary
    auto inner_bdry_nr = reader.PhysicalEntityName2Nr("Inner boundary");
    auto inner_bdry_sel = [&](const lf::mesh::Entity &e) {
      assert(e.RefEl() == lf::base::RefEl::kSegment());
      return reader.IsPhysicalEntity(e, inner_bdry_nr);
    };

    // Determine whether a point belongs to inner area
    auto inner_nr = reader.PhysicalEntityName2Nr("Inner domain");
    auto inner_sel = [&](const lf::mesh::Entity &e) {
      assert(e.RefEl() == lf::base::RefEl::kTria());
      return reader.IsPhysicalEntity(e, inner_nr);
    };

    std::cout.precision(std::numeric_limits<double>::digits10);
    std::cout << std::setw(10) << (1 << (level-2));

    out.precision(std::numeric_limits<double>::digits10);
    out << std::setw(10) << (1 << (level-2));
    
    Eigen::Vector2d force =
        transmission_fem::CalculateForce(fe_space, dir_sel, inner_bdry_sel,
            inner_sel, g, eta, epsilon1, epsilon2, grad_w, out);
  }

  return 0;
}
