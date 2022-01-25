#include "transmission_fem.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <lf/geometry/geometry.h>
#include <lf/fe/fe.h>
#include <lf/io/io.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

int main(int argc, char *argv[]) {
  using size_type = lf::base::size_type;

  std::ofstream out("transmission_fem.txt");
  out.precision(std::numeric_limits<double>::digits10);

  double epsilon1 = 1.;
  double epsilon2 = 100.;
  const int L = 6;

  auto g = [](Eigen::Vector2d x) {
    return 2. - x[0];
  };

  auto eta = [](Eigen::Vector2d x) {
    return 0;
  };

  auto dir_sel = [](const Eigen::Vector2d& x) {
    return (x[0] - 2. > -1e-7 || x[0] + 2. < 1e-7);
  };

  if (argv[1] == std::string("0")) {
    std::cout << "Square" << std::endl;
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
    std::shared_ptr<lf::refinement::MeshHierarchy> multi_mesh_p =
        lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(mesh_p, L);
    mesh_p = multi_mesh_p->getMesh(L);
    auto fe_space =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_p);

    Eigen::VectorXd sol = transmission_fem::Solve(fe_space, dir_sel, inner_sel,
                                                  g, eta, epsilon1, epsilon2);

    lf::refinement::EntityCenterPositionSelector edge_sel_dir{dir_sel};
    auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(fe_space->Mesh(), 1)};

    // Print Dirichlet trace
    out << "Dirichlet trace" << std::endl;
    out << std::setw(25) << "x"
        << std::setw(25) << "y"
        << std::setw(25) << "u" << std::endl;
    lf::fe::MeshFunctionFE mf_sol(fe_space, sol);
    for (const lf::mesh::Entity* edge : mesh_p->Entities(1)) {
      if (bd_flags(*edge) || inner_bdry_sel(*edge)) {
        auto geo = edge->Geometry();
        Eigen::MatrixXd endpoints = lf::geometry::Corners(*geo);
        Eigen::VectorXd local_coord(1);
        local_coord << 0.0;
        Eigen::Vector2d global_coord = geo->Global(local_coord).col(0);
        double u = mf_sol(*edge, local_coord)[0];
        out << std::setw(25) << global_coord[0]
            << std::setw(25) << global_coord[1]
            << std::setw(25) << u << std::endl;
      }
    }

    // Print Neumann trace
    out << "Neumann trace" << std::endl;
    out << std::setw(25) << "x"
        << std::setw(25) << "y"
        << std::setw(25) << "psi" << std::endl;
    lf::fe::MeshFunctionGradFE mf_grad_sol(fe_space, sol);
    transmission_fem::MeshFunctionPWConstant
        mf_epsilon{epsilon1, epsilon2, inner_sel};
    Eigen::MatrixXd local = Eigen::MatrixXd::Zero(2, 1);
    lf::base::RefEl refEl = lf::base::RefEl::kTria();
    for (const lf::mesh::Entity *cell : mesh_p->Entities(0)) {
      for (int i = 0; i < 3; ++i) {
        const lf::mesh::Entity *edge = cell->SubEntities(1)[i];
        if (bd_flags(*edge) || inner_bdry_sel(*edge)) {
          double eps = mf_epsilon(*cell, local)[0];
          auto tria_geo = cell->Geometry();
          int orientation = tria_geo->Jacobian(local)
                                .determinant() > 0 ? 1 : -1;
          Eigen::MatrixXd corners = lf::geometry::Corners(*tria_geo);
          Eigen::Vector2d x0 =
              corners.col(refEl.SubSubEntity2SubEntity(1, i, 1, 0));
          Eigen::Vector2d x1 =
              corners.col(refEl.SubSubEntity2SubEntity(1, i, 1, 1));
          Eigen::Vector2d vec = x1 - x0;              
          Eigen::Vector2d normal(vec(1), -vec(0));
          normal *= orientation;
          normal.normalize();
          Eigen::Vector2d grad = mf_grad_sol(*cell, local)[0].col(0);
          double psi = grad.dot(normal);
          Eigen::Vector2d global_coord = 0.5 * (x0 + x1);
          out << std::setw(25) << global_coord[0]
              << std::setw(25) << global_coord[1];
          if (eps > 1.) {
            out << std::setw(25) << psi
                << std::setw(25) << 2 << std::endl;
          }
          else {
            out << std::setw(25) << -psi / epsilon2
                << std::setw(25) << 1 << std::endl;
          }
        }
      }
    }
  }

  else if (argv[1] == std::string("1")) {
    std::cout << "Kite" << std::endl;
    std::string mesh_file = "meshes/kite_sq" + 
                            std::to_string(L) + ".msh";
    auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
    const lf::io::GmshReader reader(std::move(mesh_factory), mesh_file);
    std::shared_ptr<const lf::mesh::Mesh> mesh_p = reader.mesh();
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

    Eigen::VectorXd sol = transmission_fem::Solve(fe_space, dir_sel, inner_sel,
                                                  g, eta, epsilon1, epsilon2);

    lf::refinement::EntityCenterPositionSelector edge_sel_dir{dir_sel};
    auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(fe_space->Mesh(), 1)};

    // Print Dirichlet trace
    out << "Dirichlet trace" << std::endl;
    out << std::setw(25) << "x"
        << std::setw(25) << "y"
        << std::setw(25) << "u" << std::endl;
    lf::fe::MeshFunctionFE mf_sol(fe_space, sol);
    for (const lf::mesh::Entity* edge : mesh_p->Entities(1)) {
      if (bd_flags(*edge) || inner_bdry_sel(*edge)) {
        auto geo = edge->Geometry();
        Eigen::MatrixXd endpoints = lf::geometry::Corners(*geo);
        Eigen::VectorXd local_coord(1);
        local_coord << 0.0;
        Eigen::Vector2d global_coord = geo->Global(local_coord).col(0);
        double u = mf_sol(*edge, local_coord)[0];
        out << std::setw(25) << global_coord[0]
            << std::setw(25) << global_coord[1]
            << std::setw(25) << u << std::endl;
      }
    }

    // Print Neumann trace
    out << "Neumann trace" << std::endl;
    out << std::setw(25) << "x"
        << std::setw(25) << "y"
        << std::setw(25) << "psi" << std::endl;
    lf::fe::MeshFunctionGradFE mf_grad_sol(fe_space, sol);
    transmission_fem::MeshFunctionPWConstant
        mf_epsilon{epsilon1, epsilon2, inner_sel};
    Eigen::MatrixXd local = Eigen::MatrixXd::Zero(2, 1);
    lf::base::RefEl refEl = lf::base::RefEl::kTria();
    for (const lf::mesh::Entity *cell : mesh_p->Entities(0)) {
      for (int i = 0; i < 3; ++i) {
        const lf::mesh::Entity *edge = cell->SubEntities(1)[i];
        if (bd_flags(*edge) || inner_bdry_sel(*edge)) {
          double eps = mf_epsilon(*cell, local)[0];
          auto tria_geo = cell->Geometry();
          int orientation = tria_geo->Jacobian(local)
                                .determinant() > 0 ? 1 : -1;
          Eigen::MatrixXd corners = lf::geometry::Corners(*tria_geo);
          Eigen::Vector2d x0 =
              corners.col(refEl.SubSubEntity2SubEntity(1, i, 1, 0));
          Eigen::Vector2d x1 =
              corners.col(refEl.SubSubEntity2SubEntity(1, i, 1, 1));
          Eigen::Vector2d vec = x1 - x0;              
          Eigen::Vector2d normal(vec(1), -vec(0));
          normal *= orientation;
          normal.normalize();
          Eigen::Vector2d grad = mf_grad_sol(*cell, local)[0].col(0);
          double psi = grad.dot(normal);
          Eigen::Vector2d global_coord = 0.5 * (x0 + x1);
          out << std::setw(25) << global_coord[0]
              << std::setw(25) << global_coord[1];
          if (eps > 1.) {
            out << std::setw(25) << psi
                << std::setw(25) << 2 << std::endl;
          }
          else {
            out << std::setw(25) << -psi / epsilon2
                << std::setw(25) << 1 << std::endl;
          }
        }
      }
    }
  }

  return 0;
}
