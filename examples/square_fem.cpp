#include "force_calculation_fem.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <string>

#include <lf/geometry/geometry.h>
#include <lf/fe/fe.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/refinement/refinement.h>
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
    filename = "square_fem" + std::string(argv[1]) + ".txt";
  }
  else {
    filename = "square_fem5.txt";
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

  auto inner_bdry_sel = [](const lf::mesh::Entity &e) {
    assert(e.RefEl() == lf::base::RefEl::kSegment());
    Eigen::MatrixXd corners = lf::geometry::Corners(*(e.Geometry()));
    Eigen::Vector2d x1, x2;
    x1 = corners.col(0);
    x2 = corners.col(1);
    auto on_bdry = [](Eigen::Vector2d x) {
      return ((fabs(x[0]) < 1e-7 || fabs(x[0] - 1.) < 1e-7) &&
                    x[1] > -1e-7 && x[1] - 1. < 1e-7) ||
             ((fabs(x[1]) < 1e-7 || fabs(x[1] - 1.) < 1e-7) &&
                    x[0] > -1e-7 && x[0] - 1. < 1e-7);
    };
    return on_bdry(x1) && on_bdry(x2) &&
           (fabs(x1[0] - x2[0]) < 1e-7 || fabs(x1[1] - x2[1]) < 1e-7);
  };

  auto inner_sel = [](const lf::mesh::Entity& e) {
    assert(e.RefEl() == lf::base::RefEl::kTria());
    Eigen::MatrixXd center = Eigen::MatrixXd::Ones(2, 1) * 0.5;
    Eigen::Vector2d x = e.Geometry()->Global(center).col(0);
    return (x[0] >= 0. && x[0] <= 1. && x[1] >= 0. && x[1] <= 1.);
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

  // Generate mesh
  std::shared_ptr<lf::mesh::hybrid2d::MeshFactory> mesh_factory_ptr =
      std::make_shared<lf::mesh::hybrid2d::MeshFactory>(2);
  std::array<std::array<double, 2>, 4> node_coord{
    std::array<double, 2>({ 2.,  2.}),
    std::array<double, 2>({-2.,  2.}),
    std::array<double, 2>({-2., -2.}),
    std::array<double, 2>({ 2., -2.})
  };
  for (const auto& node : node_coord) {
    mesh_factory_ptr->AddPoint(Eigen::Vector2d({node[0], node[1]}));
  }
  mesh_factory_ptr->AddEntity(
      lf::base::RefEl::kTria(), std::vector<size_type>({0, 1, 3}),
      std::unique_ptr<lf::geometry::Geometry>(nullptr));
  mesh_factory_ptr->AddEntity(
      lf::base::RefEl::kTria(), std::vector<size_type>({1, 2, 3}),
      std::unique_ptr<lf::geometry::Geometry>(nullptr));
  std::shared_ptr<lf::mesh::Mesh> mesh_p = mesh_factory_ptr->Build();

  const int reflevels = 9;
  std::shared_ptr<lf::refinement::MeshHierarchy> multi_mesh_p =
      lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(mesh_p,
                                                              reflevels);

  std::cout << "shape of inner dielectic: square" << std::endl;
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

  out << "shape of inner dielectic: square" << std::endl;
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

  for (int level = 4; level <= reflevels; ++level) {
    mesh_p = multi_mesh_p->getMesh(level);
    auto fe_space =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_p);

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
