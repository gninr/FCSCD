#ifndef FORCECALCULATIONBEMHPP
#define FORCECALCULATIONBEMHPP

#include "factors.hpp"
#include "integration.hpp"
#include "transmission_bem.hpp"
#include "velocity_fields.hpp"

#include <functional>
#include <iostream>
#include <math.h>

#include "abstract_bem_space.hpp"
#include "abstract_parametrized_curve.hpp"
#include "dirichlet.hpp"
#include "double_layer.hpp"
#include "gauleg.hpp"
#include "hypersingular.hpp"
#include "logweight_quadrature.hpp"
#include "parametrized_mesh.hpp"
#include "single_layer.hpp"
#include <Eigen/Dense>

namespace transmission_bem {
Eigen::VectorXd SolveAdjoint(
    const parametricbem2d::ParametrizedMesh &mesh,
    const parametricbem2d::AbstractBEMSpace &space_d,
    const parametricbem2d::AbstractBEMSpace &space_n,
    std::function<bool(Eigen::Vector2d)> dir_sel,
    std::function<double(Eigen::Vector2d)> g,
    std::function<double(Eigen::Vector2d)> eta,
    double epsilon1, double epsilon2, unsigned order) {
  // Compute Space Information
  Dims dims_d, dims_n;
  Indices ind_d, ind_n;
  ComputeDirSpaceInfo(mesh, space_d, dir_sel, &dims_d, &ind_d);
  ComputeNeuSpaceInfo(mesh, space_n, dir_sel, &dims_n, &ind_n);

  // Get Galerkin matrices
  Eigen::MatrixXd M = parametricbem2d::MassMatrix(
                          mesh, space_n, space_d, order);
  Eigen::MatrixXd V = parametricbem2d::single_layer::GalerkinMatrix(
                          mesh, space_n, order);
  Eigen::MatrixXd K = parametricbem2d::double_layer::GalerkinMatrix(
                          mesh, space_d, space_n, order);
  Eigen::MatrixXd W = parametricbem2d::hypersingular::GalerkinMatrix(
                          mesh, space_d, order);

  // Assemble LHS
  Eigen::MatrixXd lhs(dims_d.i + dims_n.i + dims_d.n + dims_n.d,
                      dims_d.i + dims_n.i + dims_d.n + dims_n.d);
  lhs << (epsilon1 / epsilon2 + 1) * Slice(W, ind_d.i, ind_d.i),
         2 * Slice(K, ind_n.i, ind_d.i).transpose(),
         Slice(W, ind_d.i, ind_d.n),
         Slice(K, ind_n.d, ind_d.i).transpose(),
         
         2 * Slice(K, ind_n.i, ind_d.i),
         -(epsilon2 / epsilon1 + 1) * Slice(V, ind_n.i, ind_n.i),
         Slice(K, ind_n.i, ind_d.n),
         -Slice(V, ind_n.i, ind_n.d),

         Slice(W, ind_d.n, ind_d.i),
         Slice(K, ind_n.i, ind_d.n).transpose(),
         Slice(W, ind_d.n, ind_d.n),
         Slice(K, ind_n.d, ind_d.n).transpose(),

         Slice(K, ind_n.d, ind_d.i),
         -Slice(V, ind_n.d, ind_n.i),
         Slice(K, ind_n.d, ind_d.n),
         -Slice(V, ind_n.d, ind_n.d);

  // Assemble RHS
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(lhs.rows());
  rhs.segment(dims_d.i + dims_n.i, dims_d.n) =
      0.5 * epsilon2 * Slice(M, ind_n.n, ind_d.n).transpose()
          * InterpolateNeuData(mesh, eta, space_n, ind_n);
  rhs.segment(dims_d.i + dims_n.i + dims_d.n, dims_n.d) =
      0.5 * epsilon2 * Slice(M, ind_n.d, ind_d.d)
          * InterpolateDirData(mesh, g, space_d, ind_d);

  // Solve LSE
  Eigen::HouseholderQR<Eigen::MatrixXd> dec(lhs);
  Eigen::VectorXd sol = dec.solve(rhs);

  
  std::cout << "\nrho_i" << std::endl;
  std::cout << sol.segment(0, dims_d.i) << std::endl;
  std::cout << "\npi_i" << std::endl;
  std::cout << sol.segment(dims_d.i, dims_n.i) << std::endl;
  std::cout << "\nrho" << std::endl;
  std::cout << sol.segment(dims_d.i + dims_n.i, dims_d.n) << std::endl;
  std::cout << "\npi" << std::endl;
  std::cout << sol.segment(dims_d.i + dims_n.i + dims_d.n, dims_n.d)
            << std::endl;
  

  /*
  Eigen::MatrixXd rhs_mat(dims_d.i + dims_n.i + dims_d.n + dims_n.d,
                          dims_d.d + dims_n.n);
  rhs_mat << -Slice(W, ind_d.i, ind_d.d),
             -Slice(K, ind_n.n, ind_d.i).transpose(),
             
             -Slice(K, ind_n.i, ind_d.d),
             Slice(V, ind_n.i, ind_n.n),

             -Slice(W, ind_d.n, ind_d.d),
             (0.5 * Slice(M, ind_n.n, ind_d.n)
                 - Slice(K, ind_n.n, ind_d.n)).transpose(),

             -0.5 * Slice(M, ind_n.d, ind_d.d) - Slice(K, ind_n.d, ind_d.d),
             Slice(V, ind_n.d, ind_n.n);
  Eigen::VectorXd rhs_vec(dims_d.d + dims_n.n);
  rhs_vec << InterpolateDirData(mesh, g, space_d, ind_d),
             InterpolateNeuData(mesh, eta, space_n, ind_n);
  std::cout << "l = " << -sol_vec.dot(rhs_mat * rhs_vec) << std::endl;
  Eigen::VectorXd state_sol_vec(dims_d.i + dims_n.i + dims_d.n + dims_n.d);
  state_sol_vec << state_sol.u_i, state_sol.psi_i, state_sol.u, state_sol.psi;
  std::cout << "J = " << -state_sol_vec.dot(rhs) << std::endl;
  */

  return sol;
}

double ComputeShapeDerivative(const parametricbem2d::ParametrizedMesh &mesh,
                              const parametricbem2d::AbstractBEMSpace &space_d,
                              const parametricbem2d::AbstractBEMSpace &space_n,
                              const AbstractVelocityField &nu,
                              std::function<bool(Eigen::Vector2d)> dir_sel,
                              std::function<double(Eigen::Vector2d)> g,
                              std::function<double(Eigen::Vector2d)> eta,
                              double epsilon1, double epsilon2,
                              const Eigen::VectorXd &state_sol,
                              const Eigen::VectorXd &adj_sol,
                              unsigned order) {
  // Compute Space Information
  Dims dims_d, dims_n;
  Indices ind_d, ind_n;
  ComputeDirSpaceInfo(mesh, space_d, dir_sel, &dims_d, &ind_d);
  ComputeNeuSpaceInfo(mesh, space_n, dir_sel, &dims_n, &ind_n);
  Eigen::VectorXd g_interp = InterpolateDirData(mesh, g, space_d, ind_d);
  Eigen::VectorXd eta_interp = InterpolateNeuData(mesh, eta, space_n, ind_n);

  double force = 0.0;

  Kernel1 kernel1;
  Kernel2 kernel2;
  LogKernel log_kernel;
  Factor1 F1;
  Factor2 F2;
  Factor3 F3;
  Factor4 F4;

  Eigen::MatrixXd V = ComputeMatrix(mesh, space_n, space_n,
      dims_n.all, dims_n.all, kernel1, F1, F1, nu, order);
  Eigen::MatrixXd K = ComputeMatrix(mesh, space_n, space_d,
      dims_n.all, dims_d.all, kernel2, F1, F1, nu, order);
  Eigen::MatrixXd W =
      ComputeMatrix(mesh, space_d, space_d, dims_d.all, dims_d.all,
                    kernel1, F2, F2, nu, order) +
      ComputeMatrix(mesh, space_d, space_d, dims_d.all, dims_d.all,
                    log_kernel, F3, F2, nu, order) +
      ComputeMatrix(mesh, space_d, space_d, dims_d.all, dims_d.all,
                    log_kernel, F4, F2, nu, order);

  Eigen::MatrixXd mat_a(dims_d.i + dims_n.i + dims_d.n + dims_n.d,
                        dims_d.i + dims_n.i + dims_d.n + dims_n.d);
  mat_a << (epsilon1 / epsilon2 + 1) * Slice(W, ind_d.i, ind_d.i),
           2 * Slice(K, ind_n.i, ind_d.i).transpose(),
           Slice(W, ind_d.i, ind_d.n),
           Slice(K, ind_n.d, ind_d.i).transpose(),
          
           2 * Slice(K, ind_n.i, ind_d.i),
           -(epsilon2 / epsilon1 + 1) * Slice(V, ind_n.i, ind_n.i),
           Slice(K, ind_n.i, ind_d.n),
           -Slice(V, ind_n.i, ind_n.d),

           Slice(W, ind_d.n, ind_d.i),
           Slice(K, ind_n.i, ind_d.n).transpose(),
           Eigen::MatrixXd::Zero(dims_d.n, dims_d.n + dims_n.d),

           Slice(K, ind_n.d, ind_d.i),
           -Slice(V, ind_n.d, ind_n.i),
           Eigen::MatrixXd::Zero(dims_n.d, dims_d.n + dims_n.d);
           
  Eigen::MatrixXd mat_b(dims_d.i + dims_n.i + dims_d.n + dims_n.d,
                        dims_n.n + dims_d.d);
  mat_b << -Slice(W, ind_d.i, ind_d.d),
           -Slice(K, ind_n.n, ind_d.i).transpose(),

           -Slice(K, ind_n.i, ind_d.d),
           Slice(V, ind_n.i, ind_n.n),

           Eigen::MatrixXd::Zero(dims_d.n + dims_n.d, dims_d.d + dims_n.n);

  Eigen::VectorXd vec(dims_d.d + dims_n.n);
  vec << InterpolateDirData(mesh, g, space_d, ind_d),
         InterpolateNeuData(mesh, eta, space_n, ind_n);

  return adj_sol.dot(mat_a * state_sol - mat_b * vec);
}

double CalculateForce(const parametricbem2d::ParametrizedMesh &mesh,
                      const parametricbem2d::AbstractBEMSpace &space_d,
                      const parametricbem2d::AbstractBEMSpace &space_n,
                      const AbstractVelocityField &nu,
                      std::function<bool(Eigen::Vector2d)> dir_sel,
                      std::function<double(Eigen::Vector2d)> g,
                      std::function<double(Eigen::Vector2d)> eta,
                      double epsilon1, double epsilon2, unsigned order) {
  // Get state solution
  Eigen::VectorXd state_sol = Solve(
      mesh, space_d, space_n, dir_sel, g, eta, epsilon1, epsilon2, order);
  // Get adjoint solution
  Eigen::VectorXd adj_sol = SolveAdjoint(mesh, space_d, space_n, dir_sel, g, eta,
      epsilon1, epsilon2, order);

  double force = ComputeShapeDerivative(mesh, space_d, space_n, nu, dir_sel,
      g, eta, epsilon1, epsilon2, state_sol, adj_sol, order);
      
  //std::cout << "force = " << force << std::endl;
  return force;
}
} // namespace transmission_bem

#endif // FORCECALCULATIONBEMHPP