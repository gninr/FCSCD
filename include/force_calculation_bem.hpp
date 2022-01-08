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

double c = 1. / (2. * M_PI);

Solution SolveAdjoint(const parametricbem2d::ParametrizedMesh &mesh,
                      const parametricbem2d::AbstractBEMSpace &space_d,
                      const parametricbem2d::AbstractBEMSpace &space_n,
                      std::function<bool(Eigen::Vector2d)> dir_sel,
                      std::function<double(Eigen::Vector2d)> g,
                      std::function<double(Eigen::Vector2d)> eta,
                      double epsilon1, double epsilon2, unsigned order,
                      Solution state_sol) {
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
  Eigen::VectorXd rhs(lhs.rows());
  rhs << (epsilon1 - epsilon2) / 2
             * Slice(M, ind_n.i, ind_d.i).transpose() * state_sol.psi_i,
         (epsilon1 - epsilon2) / 2
             * Slice(M, ind_n.i, ind_d.i) * state_sol.u_i,
         -epsilon2 / 2 * Slice(M, ind_n.n, ind_d.n).transpose()
             * InterpolateNeuData(mesh, eta, space_n, ind_n),
         -epsilon2 / 2 * Slice(M, ind_n.d, ind_d.d)
             * InterpolateDirData(mesh, g, space_d, ind_d);

  // Solve LSE
  Eigen::HouseholderQR<Eigen::MatrixXd> dec(lhs);
  Eigen::VectorXd sol_vec = dec.solve(rhs);

  // Construct solution
  Solution sol;
  sol.u_i = sol_vec.segment(0, dims_d.i);
  sol.psi_i = sol_vec.segment(dims_d.i, dims_n.i);
  sol.u = sol_vec.segment(dims_d.i + dims_n.i, dims_d.n);
  sol.psi = sol_vec.segment(dims_d.i + dims_n.i + dims_d.n, dims_n.d);
  return sol;
}

Eigen::MatrixXd ComputeMatrixJ(
    const parametricbem2d::ParametrizedMesh &mesh,
    const parametricbem2d::AbstractBEMSpace &space_x,
    const parametricbem2d::AbstractBEMSpace &space_y,
    const Dims &dims_x, const Dims &dims_y,
    const Indices &ind_x, const Indices &ind_y,
    const AbstractVelocityField &nu, unsigned order) {
  parametricbem2d::PanelVector panels = mesh.getPanels();
  unsigned Q_x = space_x.getQ();
  unsigned Q_y = space_y.getQ();
  unsigned numpanels_i = mesh.getSplit();
  QuadRule GaussQR = getGaussQR(order);
  // Initialize matrix
  Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(dims_x.all, dims_y.all);

  for (unsigned i = 0; i < numpanels_i; ++i) {
    parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
    for (unsigned I = 0; I < Q_x; ++I) {
      for (unsigned J = 0; J < Q_y; ++J) {
        auto integrand = [&](double s) {
          Eigen::Vector2d x = pi(s);
          Eigen::Vector2d tangent = pi.Derivative(s);
          Eigen::Vector2d normal(tangent(1), -tangent(0));
          normal /= normal.norm();
          return space_x.evaluateShapeFunction(I, s) *
                 space_y.evaluateShapeFunction(J, s) *
                 (normal.dot(nu.grad(x) * normal) - nu.div(x)) *
                 pi.Derivative(s).norm();
        };
        // Local to global mapping of elements
        unsigned II = space_x.LocGlobMap2(I + 1, i + 1, mesh) - 1;
        unsigned JJ = space_y.LocGlobMap2(J + 1, i + 1, mesh) - 1;
        mat(II, JJ) +=
            parametricbem2d::ComputeIntegral(integrand, -1, 1, GaussQR);
      }
    }
  }

  return Slice(mat, ind_x.i, ind_y.i);
}

double ComputeShapeDerivative(const parametricbem2d::ParametrizedMesh &mesh,
                              const parametricbem2d::AbstractBEMSpace &space_d,
                              const parametricbem2d::AbstractBEMSpace &space_n,
                              const AbstractVelocityField &nu,
                              std::function<bool(Eigen::Vector2d)> dir_sel,
                              std::function<double(Eigen::Vector2d)> g,
                              std::function<double(Eigen::Vector2d)> eta,
                              double epsilon1, double epsilon2,
                              const Solution &state_sol,
                              const Solution &adj_sol,
                              unsigned order) {
  // Compute Space Information
  Dims dims_d, dims_n;
  Indices ind_d, ind_n;
  ComputeDirSpaceInfo(mesh, space_d, dir_sel, &dims_d, &ind_d);
  ComputeNeuSpaceInfo(mesh, space_n, dir_sel, &dims_n, &ind_n);
  Eigen::VectorXd g_interp = InterpolateDirData(mesh, g, space_d, ind_d);
  Eigen::VectorXd eta_interp = InterpolateNeuData(mesh, eta, space_n, ind_n);

  double force = 0.0;

  double a_V_ii = 0, a_V_iD = 0, a_V_Di = 0,
         a_K_ii_1 = 0, a_K_ii_2 = 0, a_K_iD_1 = 0, a_K_iD_2 = 0, a_K_Ni_1 = 0, a_K_Ni_2 = 0,
         a_W_ii = 0, a_W_iN = 0, a_W_Ni = 0,
         b_V_Ni = 0, b_K_Di = 0, b_K_iN = 0, b_W_Di = 0,
         J = 0;
  {
    Kernel1 kernel;
    Factor1 F;
    Eigen::MatrixXd mat = ComputeSingularMatrix(mesh, space_n, space_n,
        dims_n, dims_n, ind_n, ind_n, kernel, F, F, nu, order);
    // a_V_ii(psi_i, pi_i)
    a_V_ii += -c * state_sol.psi_i.dot(mat * adj_sol.psi_i);
  }

  {
    Kernel2 kernel;
    Factor1 F;
    Eigen::MatrixXd mat = ComputeGeneralMatrix(mesh, space_n, space_n,
        dims_n, dims_n, ind_n, ind_n, kernel, F, F, nu, order);
    // a_V_iD(psi_i, pi)
    a_V_iD += c * adj_sol.psi.dot(
                      mat.block(0, 0, dims_n.d, dims_n.i) * state_sol.psi_i);
    // a_V_Di(psi, pi_i)
    a_V_Di += c * state_sol.psi.dot(
                      mat.block(0, 0, dims_n.d, dims_n.i) * adj_sol.psi_i);
    // b_V_Ni(eta, pi_i)
    b_V_Ni +=
        c * eta_interp.dot(
                mat.block(dims_n.d, 0, dims_n.n, dims_n.i) * adj_sol.psi_i);
  }

  {
    Kernel3 kernel;
    Factor1 F;
    Eigen::MatrixXd mat = ComputeSingularMatrix(mesh, space_n, space_d,
        dims_n, dims_d, ind_n, ind_d, kernel, F, F, nu, order);
    // a_K_ii(u_i, pi_i)
    a_K_ii_1 += c * adj_sol.psi_i.dot(mat * state_sol.u_i);
    // a_K_ii(rho_i, psi_i)
    a_K_ii_2 += c * state_sol.psi_i.dot(mat * adj_sol.u_i);
  }

  {
    Kernel4 kernel;
    Factor1 F;
    Eigen::MatrixXd mat = ComputeGeneralMatrix(mesh, space_n, space_d,
        dims_n, dims_d, ind_n, ind_d, kernel, F, F, nu, order);
    // a_K_iD(rho_i, psi)
    a_K_iD_1 += c * state_sol.psi.dot(
                        mat.block(0, 0, dims_n.d, dims_d.i) * adj_sol.u_i);
    // a_K_iD(u_i, pi)
    a_K_iD_2 += c * adj_sol.psi.dot(
                        mat.block(0, 0, dims_n.d, dims_d.i) * state_sol.u_i);
    // b_K_iN(rho_i, eta)
    b_K_iN += c * eta_interp.dot(
                      mat.block(dims_n.d, 0, dims_n.n, dims_d.i) * adj_sol.u_i);
  }

  {
    Kernel5 kernel;
    Factor1 F;
    Eigen::MatrixXd mat = ComputeGeneralMatrix(mesh, space_d, space_n,
        dims_d, dims_n, ind_d, ind_n, kernel, F, F, nu, order);
    // a_K_Ni(u, pi_i)
    a_K_Ni_1 += c * state_sol.u.dot(
                        mat.block(dims_d.d, 0, dims_d.n, dims_n.i) *
                            adj_sol.psi_i);
    // a_K_Ni(rho, psi_i)
    a_K_Ni_2 += c * adj_sol.u.dot(
                        mat.block(dims_d.d, 0, dims_d.n, dims_n.i) *
                            state_sol.psi_i);
    // b_K_Di(g, pi_i)
    b_K_Di +=
        c * g_interp.dot(
                mat.block(0, 0, dims_d.d, dims_n.i) * state_sol.psi_i);
  }

  {
    Kernel1 kernel;
    Factor2 F;
    Eigen::MatrixXd mat = ComputeSingularMatrix(mesh, space_d, space_d,
        dims_d, dims_d, ind_d, ind_d, kernel, F, F, nu, order);
    // a_W_ii(u_i, rho_i) term1
    a_W_ii += -c * adj_sol.u_i.dot(mat * state_sol.u_i);
  }

  {
    LogKernel kernel;
    Factor3 F;
    Factor2 G;
    Eigen::MatrixXd mat = ComputeSingularMatrix(mesh, space_d, space_d,
        dims_d, dims_d, ind_d, ind_d, kernel, F, G, nu, order);
    // a_W_ii(u_i, rho_i) term2
    a_W_ii += c * adj_sol.u_i.dot(mat * state_sol.u_i);
    // a_W_ii(u_i, rho_i) term4
    a_W_ii += c * state_sol.u_i.dot(mat * adj_sol.u_i);
  }

  {
    LogKernel kernel;
    Factor4 F;
    Factor2 G;
    Eigen::MatrixXd mat = ComputeSingularMatrix(mesh, space_d, space_d,
        dims_d, dims_d, ind_d, ind_d, kernel, F, G, nu, order);
    // a_W_ii(u_i, rho_i) term3
    a_W_ii += c * adj_sol.u_i.dot(mat * state_sol.u_i);
    // a_W_ii(u_i, rho_i) term5
    a_W_ii += c * state_sol.u_i.dot(mat * adj_sol.u_i);
  }

  {
    Kernel1 kernel;
    Factor2 F;
    Eigen::MatrixXd mat = ComputeGeneralMatrix(mesh, space_d, space_d,
        dims_d, dims_d, ind_d, ind_d, kernel, F, F, nu, order);
    // a_W_iN(u_i, rho) term1
    a_W_iN += -c * adj_sol.u.dot(
                       mat.block(dims_d.d, 0, dims_d.n, dims_d.i) *
                           state_sol.u_i);
    // a_W_Ni(u, rho_i) term1
    a_W_Ni += -c * state_sol.u.dot(
                       mat.block(dims_d.d, 0, dims_d.n, dims_d.i) *
                           adj_sol.u_i);
    // b_W_Di(g, rho_i) term1
    b_W_Di += -c * g_interp.dot(
                       mat.block(0, 0, dims_d.d, dims_d.i) * adj_sol.u_i);
  }

  {
    LogKernel kernel;
    Factor3 F;
    Factor2 G;
    Eigen::MatrixXd mat = ComputeGeneralMatrix(mesh, space_d, space_d,
        dims_d, dims_d, ind_d, ind_d, kernel, F, G, nu, order);
    // a_W_iN(u_i, rho) term2
    a_W_iN += c * adj_sol.u.dot(
                       mat.block(dims_d.d, 0, dims_d.n, dims_d.i) *
                           state_sol.u_i);
    // a_W_Ni(u, rho_i) term2
    a_W_Ni += c * state_sol.u.dot(
                      mat.block(dims_d.d, 0, dims_d.n, dims_d.i) *
                          adj_sol.u_i);
    // b_W_Di(g, rho_i) term2
    b_W_Di += -c * g_interp.dot(
                       mat.block(0, 0, dims_d.d, dims_d.i) * adj_sol.u_i);
  }

  {
    LogKernel kernel;
    Factor4 F;
    Factor2 G;
    Eigen::MatrixXd mat = ComputeGeneralMatrix(mesh, space_d, space_d,
        dims_d, dims_d, ind_d, ind_d, kernel, F, G, nu, order);
    // a_W_iN(u_i, rho) term3
    a_W_iN += c * adj_sol.u.dot(
                      mat.block(dims_d.d, 0, dims_d.n, dims_d.i) *
                          state_sol.u_i);
    // a_W_Ni(u, rho_i) term3
    a_W_Ni += c * state_sol.u.dot(
                      mat.block(dims_d.d, 0, dims_d.n, dims_d.i) *
                          adj_sol.u_i);
    // b_W_Di(g, rho_i) term3
    b_W_Di += -c * g_interp.dot(
                       mat.block(0, 0, dims_d.d, dims_d.i) * adj_sol.u_i);
  }

  {
    Eigen::MatrixXd mat = ComputeMatrixJ(mesh, space_d, space_n,
        dims_d, dims_n, ind_d, ind_n, nu, order);
    // J(u_i, psi_i, u, psi)
    J += 0.5 * (epsilon2 - epsilon1) *
             state_sol.u_i.dot(mat * adj_sol.psi_i);
  }

  force = J
      + (epsilon1 / epsilon2 + 1) * a_W_ii + 2 * a_K_ii_2 + a_W_Ni + a_K_iD_1
      + b_W_Di + b_K_iN
      + 2 * a_K_ii_1 - (epsilon2 / epsilon1 + 1) * a_V_ii + a_K_Ni_1 - a_V_Di
      + b_K_Di - b_V_Ni
      + a_W_iN + a_K_Ni_2
      + a_K_iD_2 - a_V_iD;

  return force;
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
  Solution state_sol = Solve(
      mesh, space_d, space_n, dir_sel, g, eta, epsilon1, epsilon2, order);
  // Get adjoint solution
  Solution adj_sol = SolveAdjoint(mesh, space_d, space_n, dir_sel, g, eta,
      epsilon1, epsilon2, order, state_sol);

  /*
  std::cout << "\nu_i" << std::endl;
  std::cout << state_sol.u_i << std::endl;
  std::cout << "\npsi_i" << std::endl;
  std::cout << state_sol.psi_i << std::endl;
  std::cout << "\nu" << std::endl;
  std::cout << state_sol.u << std::endl;
  std::cout << "\npsi" << std::endl;
  std::cout << state_sol.psi << std::endl;

  std::cout << "\nrho_i" << std::endl;
  std::cout << adj_sol.u_i << std::endl;
  std::cout << "\npi_i" << std::endl;
  std::cout << adj_sol.psi_i << std::endl;
  std::cout << "\nrho" << std::endl;
  std::cout << adj_sol.u << std::endl;
  std::cout << "\npi" << std::endl;
  std::cout << adj_sol.psi << std::endl;
  */
  double force = ComputeShapeDerivative(mesh, space_d, space_n, nu, dir_sel,
      g, eta, epsilon1, epsilon1, state_sol, adj_sol, order);
      
  //std::cout << "force = " << force << std::endl;
  return force;
}
} // namespace transmission_bem

#endif // FORCECALCULATIONBEMHPP