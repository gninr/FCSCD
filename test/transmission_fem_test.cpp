#include "transmission_fem.hpp"

#include <iostream>
#include <lf/geometry/geometry.h>
#include <lf/fe/fe.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

int main() {
  using size_type = lf::base::size_type;

  double epsilon1 = 1.;
  double epsilon2 = 100.;
  auto epsilon = [&](Eigen::Vector2d x) {
    Eigen::Matrix2d A;
    if (x[0] >= 0. && x[0] <= 1. && x[1] >= 0. && x[1] <= 1.) {
      A = epsilon1 * Eigen::Matrix2d::Identity();
    } else {
      A = epsilon2 * Eigen::Matrix2d::Identity();
    }
    return A;
  };

  auto g = [](Eigen::Vector2d x) {
    return 1.1 - x[0];
  };

  auto eta = [](Eigen::Vector2d x) {
    return 0;
  };

  auto dir_sel = [](const Eigen::Vector2d& x) {
    return (x[0] - 1.1 > -1e-7 || x[0] + 1.1 < 1e-7);
  };

  // Generate mesh
  std::shared_ptr<lf::mesh::hybrid2d::MeshFactory> mesh_factory_ptr =
      std::make_shared<lf::mesh::hybrid2d::MeshFactory>(2);
  std::array<std::array<double, 2>, 4> node_coord{
    std::array<double, 2>({ 1.1,  1.1}),
    std::array<double, 2>({-1.1,  1.1}),
    std::array<double, 2>({-1.1, -1.1}),
    std::array<double, 2>({ 1.1, -1.1})
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

  Eigen::VectorXd sol =
      transmission_fem::Solve(fe_space, dir_sel, g, eta, epsilon);
  const lf::fe::MeshFunctionFE mf_sol(fe_space, sol);
  const lf::fe::MeshFunctionGradFE mf_grad_sol(fe_space, sol);

  std::cout << "u" << std::endl;
  auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(fe_space->Mesh(), 1)};
  for (const lf::mesh::Entity* edge : mesh_p->Entities(1)) {
    if (bd_flags(*edge)) {
      Eigen::MatrixXd endpoints = lf::geometry::Corners(*(edge->Geometry()));
      Eigen::VectorXd local_coord(1);
      local_coord << 0.5;
      if (!dir_sel(endpoints.col(0)) || !dir_sel(endpoints.col(1))) {
        auto u = mf_sol(*edge, local_coord);
        std::cout << u[0] << std::endl;
      }
    }
  }

  return 0;
}
