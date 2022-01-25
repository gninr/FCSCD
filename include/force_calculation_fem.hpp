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
    std::function<bool(const lf::mesh::Entity&)> inner_bdry_sel,
    std::function<bool(const lf::mesh::Entity&)> inner_sel,
    std::function<double(Eigen::Vector2d)> g,
    std::function<double(Eigen::Vector2d)> eta,
    double epsilon1, double epsilon2,
    std::function<Eigen::Vector2d(Eigen::Vector2d)> grad_w,
    std::ofstream &out) {

  Eigen::VectorXd sol =
      Solve(fe_space, dir_sel, inner_sel, g, eta, epsilon1, epsilon2);

  const lf::mesh::Mesh& mesh{*(fe_space->Mesh())};
  MeshFunctionPWConstant mf_epsilon{epsilon1, epsilon2, inner_sel};
  lf::fe::MeshFunctionGradFE mf_grad_sol(fe_space, sol);

  // Volume formula
  lf::mesh::utils::MeshFunctionGlobal mf_grad_w{grad_w};
  lf::mesh::utils::MeshFunctionConstant mf_c{0.5};
  Eigen::Vector2d force_vol = lf::fe::IntegrateMeshFunction(mesh,
      mf_epsilon *
          (mf_grad_sol * lf::mesh::utils::transpose(mf_grad_sol) * mf_grad_w -
           mf_c * lf::mesh::utils::squaredNorm(mf_grad_sol) * mf_grad_w),
      2);

  // Stress tensor
  const lf::assemble::DofHandler& dofh = fe_space->LocGlobMap();
  Eigen::MatrixXd local = Eigen::MatrixXd::Zero(2, 1);
  lf::base::RefEl refEl = lf::base::RefEl::kTria();
  Eigen::Vector2d force_bdry = Eigen::Vector2d::Zero();
  for (const lf::mesh::Entity *cell : mesh.Entities(0)) {
    for (int i = 0; i < 3; ++i) {
      const lf::mesh::Entity *edge = cell->SubEntities(1)[i];
      if (inner_bdry_sel(*edge)) {
        auto tria_geo = cell->Geometry();
        int orientation = tria_geo->Jacobian(local)
                              .determinant() > 0 ? 1 : -1;
        Eigen::MatrixXd corners = lf::geometry::Corners(*tria_geo);
        Eigen::Vector2d vec =
            corners.col(refEl.SubSubEntity2SubEntity(1, i, 1, 1)) -
            corners.col(refEl.SubSubEntity2SubEntity(1, i, 1, 0));
        Eigen::Vector2d normal(vec(1), -vec(0));
        normal *= orientation;
        normal.normalize();
        Eigen::Vector2d grad = mf_grad_sol(*cell, local)[0].col(0);
        Eigen::Vector2d f_density = mf_epsilon(*cell, local)[0] *
            (grad.dot(normal) * grad - 0.5 * grad.squaredNorm() * normal);
        double length = lf::geometry::Volume(*(edge->Geometry()));
        /* 
        out << "\ncorner1: "
            << corners.col(refEl.SubSubEntity2SubEntity(1, i, 1, 1))
            << std::endl;
        out << "corner0: "
            << corners.col(refEl.SubSubEntity2SubEntity(1, i, 1, 0))
            << std::endl;
        out << "orientation: " << orientation << std::endl;
        out << "normal: " << normal << std::endl;
        out << "eps: " << mf_epsilon(*cell, local)[0] << std::endl;
        out << "length: " << length << std::endl;
        out << "grad: " << grad << std::endl;
        out << "grad.normal: " << grad.dot(normal) << std::endl;
        out << "grad.tangent: " << grad.dot(vec.normalized()) * orientation
            << std::endl;
        */
        force_bdry += f_density * length;
      }
    }
  }

  std::cout << std::setw(50) << force_vol[0]
            << std::setw(25) << force_vol[1]
            << std::setw(50) << force_bdry[0]
            << std::setw(25) << force_bdry[1] << std::endl;

  out << std::setw(50) << force_vol[0]
      << std::setw(25) << force_vol[1]
      << std::setw(50) << force_bdry[0]
      << std::setw(25) << force_bdry[1] << std::endl;

  return force_vol;
}
} // namespace transmission_fem

#endif // FORCECALCULATIONFEMHPP