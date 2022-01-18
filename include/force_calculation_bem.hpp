#ifndef FORCECALCULATIONBEMHPP
#define FORCECALCULATIONBEMHPP

#include "factors.hpp"
#include "integration.hpp"
#include "transmission_bem.hpp"
#include "velocity_fields.hpp"

#include <fstream>
#include <functional>
#include <iomanip>
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
void SolveStateAndAdjoint(
    const parametricbem2d::ParametrizedMesh &mesh,
    const parametricbem2d::AbstractBEMSpace &space_d,
    const parametricbem2d::AbstractBEMSpace &space_n,
    std::function<bool(Eigen::Vector2d)> dir_sel,
    std::function<double(Eigen::Vector2d)> g,
    std::function<double(Eigen::Vector2d)> eta,
    double epsilon1, double epsilon2, unsigned order,
    Eigen::VectorXd &state_sol, Eigen::VectorXd &adj_sol) {
  // Compute Space Information
  Dims dims_d, dims_n;
  Indices ind_d, ind_n;
  ComputeDirSpaceInfo(mesh, space_d, dir_sel, dims_d, ind_d);
  ComputeNeuSpaceInfo(mesh, space_n, dir_sel, dims_n, ind_n);

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

  // Assemble RHS of state problem
  Eigen::VectorXd g_interp = InterpolateDirData(mesh, g, space_d, ind_d);
  Eigen::VectorXd eta_interp = InterpolateNeuData(mesh, eta, space_n, ind_n);

  Eigen::MatrixXd state_rhs_mat(dims_d.i + dims_n.i + dims_d.n + dims_n.d,
                                dims_d.d + dims_n.n);
  state_rhs_mat << -Slice(W, ind_d.i, ind_d.d),
                   -Slice(K, ind_n.n, ind_d.i).transpose(),

                   -Slice(K, ind_n.i, ind_d.d),
                   Slice(V, ind_n.i, ind_n.n),

                   -Slice(W, ind_d.n, ind_d.d),
                   (0.5 * Slice(M, ind_n.n, ind_d.n)
                       - Slice(K, ind_n.n, ind_d.n)).transpose(),

                   -0.5 * Slice(M, ind_n.d, ind_d.d)
                       - Slice(K, ind_n.d, ind_d.d),
                   Slice(V, ind_n.d, ind_n.n);

  Eigen::VectorXd state_rhs_vec(dims_d.d + dims_n.n);
  state_rhs_vec << g_interp, eta_interp;

  // Assemble RHS of adjoint problem
  Eigen::VectorXd adj_rhs = Eigen::VectorXd::Zero(lhs.rows());
  adj_rhs.segment(dims_d.i + dims_n.i, dims_d.n) =
      0.5 * Slice(M, ind_n.n, ind_d.n).transpose() * eta_interp;
  adj_rhs.segment(dims_d.i + dims_n.i + dims_d.n, dims_n.d) =
      0.5 * Slice(M, ind_n.d, ind_d.d) * g_interp;

  // Solve LSE
  Eigen::HouseholderQR<Eigen::MatrixXd> dec(lhs);
  state_sol = dec.solve(state_rhs_mat * state_rhs_vec);
  adj_sol = dec.solve(adj_rhs);

  /*
  std::cout << "\nu_i" << std::endl;
  std::cout << state_sol.segment(0, dims_d.i) << std::endl;
  std::cout << "\npsi_i" << std::endl;
  std::cout << state_sol.segment(dims_d.i, dims_n.i) << std::endl;
  std::cout << "\nu" << std::endl;
  std::cout << state_sol.segment(dims_d.i + dims_n.i, dims_d.n) << std::endl;
  std::cout << "\npsi" << std::endl;
  std::cout << state_sol.tail(dims_n.d) << std::endl;

  std::cout << "\nrho_i" << std::endl;
  std::cout << adj_sol.segment(0, dims_d.i) << std::endl;
  std::cout << "\npi_i" << std::endl;
  std::cout << adj_sol.segment(dims_d.i, dims_n.i) << std::endl;
  std::cout << "\nrho" << std::endl;
  std::cout << adj_sol.segment(dims_d.i + dims_n.i, dims_d.n) << std::endl;
  std::cout << "\npi" << std::endl;
  std::cout << adj_sol.tail(dims_n.d) << std::endl;
  */  

  return;
}

double ComputeShapeDerivative(
    const parametricbem2d::ParametrizedMesh &mesh,
    const parametricbem2d::AbstractBEMSpace &space_d,
    const parametricbem2d::AbstractBEMSpace &space_n,
    const AbstractVelocityField &nu,
    std::function<bool(Eigen::Vector2d)> dir_sel,
    const DirData &g, const NeuData &eta,
    double epsilon1, double epsilon2,
    const Eigen::VectorXd &state_sol,
    const Eigen::VectorXd &adj_sol,
    unsigned order) {
  // Compute Space Information
  Dims dims_d, dims_n;
  Indices ind_d, ind_n;
  ComputeDirSpaceInfo(mesh, space_d, dir_sel, dims_d, ind_d);
  ComputeNeuSpaceInfo(mesh, space_n, dir_sel, dims_n, ind_n);
  Eigen::VectorXd g_interp = InterpolateDirData(mesh,
      [&](Eigen::VectorXd x) { return g(x); }, space_d, ind_d);
  Eigen::VectorXd eta_interp = InterpolateNeuData(mesh,
      [&](Eigen::VectorXd x) { return eta(x); }, space_n, ind_n);

  Kernel1 kernel1;
  Kernel2 kernel2;
  Kernel3 kernel3;
  LogKernel log_kernel;
  Factor1 F1;
  Factor2 F2;
  Factor3 F3(g);
  Factor4 F4(g);

  Eigen::MatrixXd V = ComputeMatrix(mesh, space_n, space_n,
      dims_n.all, dims_n.all, kernel1, F1, F1, nu, order);
  Eigen::MatrixXd K = ComputeMatrix(mesh, space_n, space_d,
      dims_n.all, dims_d.all, kernel2, F1, F1, nu, order);
  Eigen::MatrixXd W = ComputeMatrix(mesh, space_d, space_d,
      dims_d.all, dims_d.all, kernel1, F2, F2, nu, order);

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
           Slice(W, ind_d.n, ind_d.n),
           Slice(K, ind_n.d, ind_d.n).transpose(),

           Slice(K, ind_n.d, ind_d.i),
           -Slice(V, ind_n.d, ind_n.i),
           Slice(K, ind_n.d, ind_d.n),
           -Slice(V, ind_n.d, ind_n.d);

  Eigen::MatrixXd mat_b(dims_d.i + dims_n.i + dims_d.n + dims_n.d,
                        dims_d.d + dims_n.n);
  mat_b << -Slice(W, ind_d.i, ind_d.d),
           -Slice(K, ind_n.n, ind_d.i).transpose(),

           -Slice(K, ind_n.i, ind_d.d),
           Slice(V, ind_n.i, ind_n.n),

           -Slice(W, ind_d.n, ind_d.d),
           -Slice(K, ind_n.n, ind_d.n).transpose(),

           -Slice(K, ind_n.d, ind_d.d),
           Slice(V, ind_n.d, ind_n.n);

  Eigen::VectorXd vec(dims_d.d + dims_n.n);
  vec << g_interp, eta_interp;
  double res = adj_sol.dot(mat_a * state_sol - mat_b * vec);

  // Compute extra terms
  Eigen::VectorXd psi = state_sol.tail(dims_n.d);
  Eigen::VectorXd rho_i = adj_sol.segment(0, dims_d.i);
  Eigen::VectorXd pi_i = adj_sol.segment(dims_d.i, dims_n.i);
  Eigen::VectorXd rho = adj_sol.segment(dims_d.i + dims_n.i, dims_d.n);
  Eigen::VectorXd pi = adj_sol.tail(dims_n.d);

  Eigen::MatrixXd K_extra = ComputeMatrix(mesh, space_n, space_d,
      dims_n.all, dims_d.all, kernel3, F3, F1, nu, order);  
  Eigen::MatrixXd W_extra = ComputeMatrix(mesh, space_d, space_d,
      dims_d.all, dims_d.all, log_kernel, F4, F2, nu, order);
  res += (Slice(K_extra, ind_n.i, ind_d.d).transpose() * pi_i +
          Slice(K_extra, ind_n.d, ind_d.d).transpose() * pi +
          Slice(W_extra, ind_d.i, ind_d.d).transpose() * rho_i +
          Slice(W_extra, ind_d.n, ind_d.d).transpose() * rho).sum();

  Eigen::VectorXd vec_extra = EvaluateExtraTerm(mesh, space_d, space_n,
                                  dims_n.all, ind_n.d, F3, nu, order);
  res += vec_extra.dot(0.5 * pi + psi);

  res *= epsilon2;
  
  /*
  Eigen::VectorXd u_i = state_sol.segment(0, dims_d.i);
  Eigen::VectorXd psi_i = state_sol.segment(dims_d.i, dims_n.i);
  Eigen::VectorXd u = state_sol.segment(dims_d.i + dims_n.i, dims_d.n);
  Eigen::VectorXd psi = state_sol.tail(dims_n.d);

  Eigen::VectorXd rho_i = adj_sol.segment(0, dims_d.i);
  Eigen::VectorXd pi_i = adj_sol.segment(dims_d.i, dims_n.i);
  Eigen::VectorXd rho = adj_sol.segment(dims_d.i + dims_n.i, dims_d.n);
  Eigen::VectorXd pi = adj_sol.tail(dims_n.d);

  Eigen::MatrixXd res_a(4, 4);
  res_a << rho_i.dot((epsilon1 / epsilon2 + 1) *
                         Slice(W, ind_d.i, ind_d.i) * u_i),
           rho_i.dot(2 * Slice(K, ind_n.i, ind_d.i).transpose() * psi_i),
           rho_i.dot(Slice(W, ind_d.i, ind_d.n) * u),
           rho_i.dot(Slice(K, ind_n.d, ind_d.i).transpose() * psi),
           pi_i.dot(2 * Slice(K, ind_n.i, ind_d.i) * u_i),
           pi_i.dot(-(epsilon2 / epsilon1 + 1) *
                        Slice(V, ind_n.i, ind_n.i) * psi_i),
           pi_i.dot(Slice(K, ind_n.i, ind_d.n) * u),
           pi_i.dot(-Slice(V, ind_n.i, ind_n.d) * psi),
           rho.dot(Slice(W, ind_d.n, ind_d.i) * u_i),
           rho.dot(Slice(K, ind_n.i, ind_d.n).transpose() * psi_i),
           0, 0,
           pi.dot(Slice(K, ind_n.d, ind_d.i) * u_i),
           pi.dot(-Slice(V, ind_n.d, ind_n.i) * psi_i),
           0, 0;
  Eigen::MatrixXd res_b(4, 2);
  res_b << rho_i.dot(-Slice(W, ind_d.i, ind_d.d) * g_interp),
           rho_i.dot(-Slice(K, ind_n.n, ind_d.i).transpose() * eta_interp),
           pi_i.dot(-Slice(K, ind_n.i, ind_d.d) * g_interp),
           pi_i.dot(Slice(V, ind_n.i, ind_n.n) * eta_interp),
           0, 0, 0, 0;

  std::cout << "block result" << std::endl;
  std::cout << "mat_a:\n" << res_a << std::endl;
  std::cout << "mat_b:\n" << res_b << std::endl;
  std::cout << "sum row1: " << res_a.row(0).sum() - res_b.row(0).sum()
            << std::endl;
  std::cout << "sum row2: " << res_a.row(1).sum() - res_b.row(1).sum()
            << std::endl;
  std::cout << "sum col1: " << res_a.col(0).sum() << std::endl;
  std::cout << "sum col2: " << res_a.col(1).sum() << std::endl;
  */

  return res;
}

// Compute net force
Eigen::Vector2d CalculateForce(
    const parametricbem2d::ParametrizedMesh &mesh,
    const parametricbem2d::AbstractBEMSpace &space_d,
    const parametricbem2d::AbstractBEMSpace &space_n,
    std::function<bool(Eigen::Vector2d)> bdry_sel,
    std::function<bool(Eigen::Vector2d)> dir_sel,
    const DirData &g, const NeuData &eta,
    double epsilon1, double epsilon2, unsigned order,
    std::ofstream &out) {
  // Get state and adjoint solution
  Eigen::VectorXd state_sol, adj_sol;
  SolveStateAndAdjoint(
      mesh, space_d, space_n, dir_sel,
      [&](Eigen::VectorXd x) { return g(x); },
      [&](Eigen::VectorXd x) { return eta(x); },
      epsilon1, epsilon2, order, state_sol, adj_sol);

  // Reuse state & adjoint solution for net force computation
  NuConstant nu_x(Eigen::Vector2d(1., 0.), bdry_sel);
  NuConstant nu_y(Eigen::Vector2d(0., 1.), bdry_sel);
  double Fx = ComputeShapeDerivative(mesh, space_d, space_n, nu_x, dir_sel,
      g, eta, epsilon1, epsilon2, state_sol, adj_sol, order);
  double Fy = ComputeShapeDerivative(mesh, space_d, space_n, nu_y, dir_sel,
      g, eta, epsilon1, epsilon2, state_sol, adj_sol, order);

  std::cout << std::setw(25) << Fx
              << std::setw(25) << Fy << std::endl;

  out << std::setw(25) << Fx
      << std::setw(25) << Fy << std::endl;

  return Eigen::Vector2d(Fx, Fy);
}

// Compute force with arbitrary velocity field
double CalculateForce(
    const parametricbem2d::ParametrizedMesh &mesh,
    const parametricbem2d::AbstractBEMSpace &space_d,
    const parametricbem2d::AbstractBEMSpace &space_n,
    std::function<bool(Eigen::Vector2d)> bdry_sel,
    std::function<bool(Eigen::Vector2d)> dir_sel,
    const DirData &g, const NeuData &eta,
    double epsilon1, double epsilon2, unsigned order,
    const AbstractVelocityField &nu) {
  // Get state and adjoint solution
  Eigen::VectorXd state_sol, adj_sol;
  SolveStateAndAdjoint(
      mesh, space_d, space_n, dir_sel,
      [&](Eigen::VectorXd x) { return g(x); },
      [&](Eigen::VectorXd x) { return eta(x); },
      epsilon1, epsilon2, order, state_sol, adj_sol);

  double force = ComputeShapeDerivative(mesh, space_d, space_n, nu, dir_sel,
        g, eta, epsilon1, epsilon2, state_sol, adj_sol, order);

  //std::cout << "force = " << force << std::endl;
  return force;
}
} // namespace transmission_bem

#endif // FORCECALCULATIONBEMHPP