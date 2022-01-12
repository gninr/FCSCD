#include "force_calculation_fem.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>

#include <lf/geometry/geometry.h>
#include <lf/fe/fe.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

int main() {
  using size_type = lf::base::size_type;
  
  std::cout << "Calculate force using FEM" << std::endl;
  std::cout << "####################################" << std::endl;

  std::ofstream out("kite_fem.txt");
  out << "Calculate force using FEM" << std::endl;
  out << "####################################" << std::endl;

  // Determine whether a point belongs to inner area
  // x = 0.35 * cos(t) + 0.1625 * cos(2*t)
  // y = 0.35 * sin(t)
  auto inner = [](Eigen::Vector2d X) {
    double x = X[0] - 0.3, y = X[1] - 0.5;
    if (y > 0.35 || y < -0.35) return false;

    if (x > -0.1625) {
      double t = asin(y / 0.35);
      double x_max = 0.35 * cos(t) + 0.1625 * cos(2 * t);
      if (x > x_max) return false;
      else return true;
    }
    else {
      double t = M_PI - asin(y / 0.35);
      double x_min = 0.35 * cos(t) + 0.1625 * cos(2 * t);
      if (x < x_min) return false;
      else return true;
    }
  };

  double epsilon1 = 1.;
  double epsilon2 = 5.;
  auto epsilon = [&](Eigen::Vector2d x) {
    Eigen::Matrix2d A;
    if (inner(x)) {
      A = epsilon1 * Eigen::Matrix2d::Identity();
    } else {
      A = epsilon2 * Eigen::Matrix2d::Identity();
    }
    return A;
  };

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

  const int reflevels = 10;
  std::shared_ptr<lf::refinement::MeshHierarchy> multi_mesh_p =
      lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(mesh_p,
                                                              reflevels);

  std::cout << "shape of inner dielectic: kite" << std::endl;
  std::cout << "epsilon1: " << epsilon1 << std::endl;
  std::cout << "epsilon2: " << epsilon2 << std::endl;
  std::cout << "------------------------------------" << std::endl;
  std::cout << std::setw(10) << "1/h"
            << std::setw(25) << "Fx"
            << std::setw(25) << "Fy" << std::endl;

  out << "shape of inner dielectic: kite" << std::endl;
  out << "epsilon1: " << epsilon1 << std::endl;
  out << "epsilon2: " << epsilon2 << std::endl;
  out << "------------------------------------" << std::endl;
  out << std::setw(10) << "1/h"
      << std::setw(25) << "Fx"
      << std::setw(25) << "Fy" << std::endl;

  for (int level = 2; level <= reflevels; ++level) {
    mesh_p = multi_mesh_p->getMesh(level);
    auto fe_space =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_p);

    Eigen::Vector2d force = transmission_fem::CalculateForce(
                                fe_space, dir_sel, g, eta, epsilon, grad_w);

    std::cout.precision(std::numeric_limits<double>::digits10);
    std::cout << std::setw(10) << (1 << (level-2))
              << std::setw(25) << force[0]
              << std::setw(25) << force[1] << std::endl;

    out.precision(std::numeric_limits<double>::digits10);
    out << std::setw(10) << (1 << (level-2))
        << std::setw(25) << force[0]
        << std::setw(25) << force[1] << std::endl;
  }

  return 0;
}
