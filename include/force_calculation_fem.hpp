#ifndef FORCECALCULATIONFEMHPP
#define FORCECALCULATIONFEMHPP

#include "transmission_fem.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>

#include <lf/fe/fe.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>
#include <Eigen/Dense>

namespace transmission_fem {
Eigen::Vector2d CalculateForce(
    std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> fe_space,
    std::function<bool(Eigen::Vector2d)> dir_sel,
    std::function<double(Eigen::Vector2d)> g,
    std::function<double(Eigen::Vector2d)> eta,
    std::function<Eigen::Matrix2d(Eigen::Vector2d)> epsilon,
    std::function<Eigen::Vector2d(Eigen::Vector2d)> grad_w,
    std::ofstream &out) {

  Eigen::VectorXd sol = Solve(fe_space, dir_sel, g, eta, epsilon);

  /*
  auto mesh_p{fe_space->Mesh()};
  const lf::fe::MeshFunctionFE mf_sol(fe_space, sol);
  std::cout << "u" << std::endl;
  auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 1)};
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
  */

  const lf::mesh::Mesh& mesh{*(fe_space->Mesh())};
  lf::mesh::utils::MeshFunctionGlobal mf_epsilon{epsilon};
  lf::mesh::utils::MeshFunctionGlobal mf_grad_w{grad_w};
  lf::mesh::utils::MeshFunctionConstant mf_c{0.5};
  lf::fe::MeshFunctionGradFE mf_grad_sol(fe_space, sol);

  Eigen::Vector2d force = lf::fe::IntegrateMeshFunction(mesh,
      mf_epsilon *
          (mf_c * (lf::mesh::utils::squaredNorm(mf_grad_sol + mf_grad_w) -
                   lf::mesh::utils::squaredNorm(mf_grad_sol) -
                   lf::mesh::utils::squaredNorm(mf_grad_w)) *
               mf_grad_sol -
           mf_c * lf::mesh::utils::squaredNorm(mf_grad_sol) * mf_grad_w),
      2);

  std::cout << std::setw(50) << force[0]
            << std::setw(25) << force[1] << std::endl;

  out << std::setw(50) << force[0]
      << std::setw(25) << force[1] << std::endl;

  return force;
}
} // namespace transmission_fem

#endif // FORCECALCULATIONFEMHPP