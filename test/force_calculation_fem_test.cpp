#include "force_calculation_fem.hpp"

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

  double epsilon1 = 1.;
  double epsilon2 = 100.;
  auto epsilon = [&](Eigen::Vector2d x) {
    Eigen::Matrix2d A;
    if (x[0] >= -1. && x[0] <= 1. && x[1] >= -1. && x[1] <= 1.) {
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
      res = - sin(2. * (c - r_in) / r_diff) * x / c / r_diff;
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

  const int L = 5;
  std::shared_ptr<lf::refinement::MeshHierarchy> multi_mesh_p =
      lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(mesh_p, L);
  mesh_p = multi_mesh_p->getMesh(L);
  auto fe_space =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_p);

  Eigen::Vector2d force = transmission_fem::CalculateForce(
                              fe_space, dir_sel, g, eta, epsilon, grad_w);

  std::cout << "force = (" << force[0] << ", " << force[1] << ")" << std::endl;

  return 0;
}
