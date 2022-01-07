#ifndef FORCECALCULATIONBEMHPP
#define FORCECALCULATIONBEMHPP

#include "factors.hpp"
#include "transmission_bem.hpp"
#include "velocity_fields.hpp"

#include <functional>
#include <iostream>
#include <limits>
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

namespace transmission_bem {

double sqrt_epsilon = std::sqrt(std::numeric_limits<double>::epsilon());
double tol = std::numeric_limits<double>::epsilon();
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
             * Slice(M, ind_d.i, ind_n.i).transpose() * state_sol.psi_i,
         (epsilon1 - epsilon2) / 2
             * Slice(M, ind_n.i, ind_d.i) * state_sol.u_i,
         -epsilon2 / 2 * Slice(M, ind_n.n, ind_d.n).transpose()
             * InterpolateNeuData(mesh, eta, space_n, ind_n),
         -epsilon2 / 2 * Slice(M, ind_n.d, ind_d.d)
             * InterpolateDirData(mesh, g, space_d, ind_d);

  // Solve LSE
  Eigen::VectorXd sol_vec = lhs.lu().solve(rhs);

  // Construct solution
  Solution sol;
  sol.u_i = sol_vec.segment(0, dims_d.i);
  sol.psi_i = sol_vec.segment(dims_d.i, dims_n.i);
  sol.u = sol_vec.segment(dims_d.i + dims_n.i, dims_d.n);
  sol.psi = sol_vec.segment(dims_d.i + dims_n.i + dims_d.n, dims_n.d);
  return sol;
}

Eigen::MatrixXd IntegralCoinciding(
    const parametricbem2d::AbstractParametrizedCurve &pi,
    const parametricbem2d::AbstractBEMSpace &space,
    const parametricbem2d::AbstractBEMSpace &space_p,
    const AbstractSingularKernel &kernel,
    const AbstractFactor &F, const AbstractFactor &G,
    const AbstractVelocityField &nu, const QuadRule &GaussQR) {
  unsigned N = GaussQR.n;
  unsigned Q = space.getQ();
  unsigned Q_p = space_p.getQ();
  Eigen::MatrixXd mat(Q, Q_p);

  for (unsigned i = 0; i < Q; ++i) {
    for (unsigned j = 0; j < Q_p; ++j) {
      // Local integrand
      auto integrand = [&](double s, double t) {
        double non_singular =
                   F(pi, space_p, nu, j, t) * G(pi, space, nu, i, s);
        double singular;
        // Direct evaluation when away from singularity
        if (fabs(s - t) > sqrt_epsilon) {
          singular = kernel(pi, pi, nu, s, t);
        }
        // Stable evaluation near singularity using Tayler expansion
        else {
          std::cout << "stable_st" << std::endl;
          singular = kernel.stable_st(pi, nu, (s + t) / 2.);
        }
        return singular * non_singular;
      };

      double local_integral = 0;
      for (unsigned k = 0; k < N; ++k) {
        for (unsigned l = 0; l < N; ++l) {
          local_integral += GaussQR.w(k) * GaussQR.w(l) *
                            integrand(GaussQR.x(k), GaussQR.x(l));
        }
      }
      mat(i, j) = local_integral;
    }
  }

  return mat;
}

Eigen::MatrixXd IntegralAdjacent(
    const parametricbem2d::AbstractParametrizedCurve &pi,
    const parametricbem2d::AbstractParametrizedCurve &pi_p,
    const parametricbem2d::AbstractBEMSpace &space,
    const parametricbem2d::AbstractBEMSpace &space_p,
    const AbstractSingularKernel &kernel,
    const AbstractFactor &F, const AbstractFactor &G,
    const AbstractVelocityField &nu, const QuadRule &GaussQR) {
  unsigned N = GaussQR.n;
  unsigned Q = space.getQ();
  unsigned Q_p = space_p.getQ();
  Eigen::MatrixXd mat(Q, Q_p);

  // Ensure common point between panels corresponds to parameter 0 in both
  // scaled parametrizations
  bool swap = (pi(1) - pi_p(-1)).norm() / 100. > tol;
  double length_pi = 2 * pi.Derivative(swap ? -1 : 1).norm();
  double length_pi_p = 2 * pi_p.Derivative(swap ? 1 : -1).norm();

  for (unsigned i = 0; i < Q; ++i) {
    for (unsigned j = 0; j < Q_p; ++j) {
      // Local integrand in polar coordinates
      auto integrand = [&](double r, double phi) {
        double s_pr = r * cos(phi);
        double t_pr = r * sin(phi);
        double s = swap ? 2 * s_pr / length_pi - 1
                        : 1 - 2 * s_pr / length_pi;
        double t = swap ? 1 - 2 * t_pr / length_pi_p
                        : 2 * t_pr / length_pi_p - 1;
        double non_singular =
            F(pi_p, space_p, nu, j, t) * G(pi, space, nu, i, s) *
                4 / length_pi / length_pi_p;
        double singular;
        // Direct evaluation away from singularity
        if (r > sqrt_epsilon) {
          singular = r * kernel(pi, pi_p, nu, s, t);
        }
        // Stable evaluation near singularity using Tayler expansion
        else {
          double s0 = swap ? -1 : 1;
          double t0 = swap ? 1 : -1;
          std::cout << "stable_pr" << std::endl;
          singular = kernel.stable_pr(pi, pi_p, nu, length_pi, length_pi_p,
                                      r, phi, s0, t0);
        }
        return singular * non_singular;
      };

      // Split integral into two parts
      double alpha = atan(length_pi_p / length_pi);
      double i1 = 0., i2 = 0.;
      // part 1 (phi from 0 to alpha)
      for (unsigned k = 0; k < N; ++k) {
        double phi = alpha / 2 * (1 + GaussQR.x(k));
        // Upper limit for inner integral
        double rmax = length_pi / cos(phi);
        // Evaluate inner integral
        double inner = 0.;
        for (unsigned l = 0; l < N; ++l) {
          double r = rmax / 2 * (1 + GaussQR.x(l));
          inner += GaussQR.w(l) * integrand(r, phi);
        }
        inner *= rmax / 2;
        i1 += GaussQR.w(k) * inner * alpha / 2;
      }
      // part 2 (phi from alpha to pi/2)
      for (unsigned k = 0; k < N; ++k) {
        double phi =
            GaussQR.x(k) * (M_PI / 2. - alpha) / 2. + (M_PI / 2. + alpha) / 2.;
        // Upper limit for inner integral
        double rmax = length_pi_p / sin(phi);
        // Evaluate inner integral
        double inner = 0.;
        for (unsigned l = 0; l < N; ++l) {
          double r = rmax / 2 * (1 + GaussQR.x(l));
          inner += GaussQR.w(l) * integrand(r, phi);
        }
        inner *= rmax / 2;
        i2 += GaussQR.w(k) * inner * (M_PI / 2. - alpha) / 2.;
      }
      mat(i, j) = i1 + i2;
    }
  }

  return mat;
}

Eigen::MatrixXd IntegralGeneral(
    const parametricbem2d::AbstractParametrizedCurve &pi,
    const parametricbem2d::AbstractParametrizedCurve &pi_p,
    const parametricbem2d::AbstractBEMSpace &space,
    const parametricbem2d::AbstractBEMSpace &space_p,
    const AbstractKernel &kernel,
    const AbstractFactor &F, const AbstractFactor &G,
    const AbstractVelocityField &nu, const QuadRule &GaussQR) {
  unsigned N = GaussQR.n;
  unsigned Q = space.getQ();
  unsigned Q_p = space_p.getQ();
  Eigen::MatrixXd mat(Q, Q_p);

  for (unsigned i = 0; i < Q; ++i) {
    for (unsigned j = 0; j < Q_p; ++j) {
      // Local integrand
      auto integrand = [&](double s, double t) {
        return kernel(pi, pi_p, nu, s, t) *
                   F(pi_p, space, nu, j, t) * G(pi, space, nu, i, s);
      };

      double local_integral = 0;
      for (unsigned k = 0; k < N; ++k) {
        for (unsigned l = 0; l < N; ++l) {
          local_integral += GaussQR.w(k) * GaussQR.w(l) *
                            integrand(GaussQR.x(k), GaussQR.x(l));
        }
      }
      mat(i, j) = local_integral;
    }
  }

  return mat;
}

// Compute matrix with elements generated by global shape functions
// associated with panels at interface
Eigen::MatrixXd ComputeSingularMatrix(
    const parametricbem2d::ParametrizedMesh &mesh,
    const parametricbem2d::AbstractBEMSpace &space,
    const parametricbem2d::AbstractBEMSpace &space_p,
    const Dims &dims, const Dims &dims_p,
    const AbstractSingularKernel &kernel,
    const AbstractFactor &F, const AbstractFactor &G,
    const AbstractVelocityField &nu, unsigned order) {
  parametricbem2d::PanelVector panels = mesh.getPanels();
  unsigned Q = space.getQ();
  unsigned Q_p = space_p.getQ();
  unsigned numpanels_i = mesh.getSplit();
  QuadRule GaussQR = getGaussQR(order);
  // Initialize matrix
  Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(dims.i, dims_p.i);

  for (unsigned i = 0; i < numpanels_i; ++i) {
    for (unsigned j = 0; j < numpanels_i; ++j) {
      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      parametricbem2d::AbstractParametrizedCurve &pi_p = *panels[j];

      // Evaluate interaction matrix
      Eigen::MatrixXd panel_mat(Q, Q_p);
      if (i == j)
        panel_mat = IntegralCoinciding(pi, space, space_p,
                                       kernel, F, G, nu, GaussQR);
      else if ((pi(1) - pi_p(-1)).norm() / 100. < tol ||
               (pi(-1) - pi_p(1)).norm() / 100. < tol)
        panel_mat = IntegralAdjacent(pi, pi_p, space, space_p,
                                     kernel, F, G, nu, GaussQR);
      else
        panel_mat = IntegralGeneral(pi, pi_p, space, space_p,
                                    kernel, F, G, nu, GaussQR);

      // Local to global mapping of elements in interaction matrix
      for (unsigned I = 0; I < Q; ++I) {
        for (unsigned J = 0; J < Q; ++J) {
          unsigned II = space.LocGlobMap2(I + 1, i + 1, mesh) - 1;
          unsigned JJ = space.LocGlobMap2(J + 1, j + 1, mesh) - 1;
          mat(II, JJ) += panel_mat(I, J);
        }
      }
    }
  }

  return mat;
}

// Compute matrix with elements generated by global shape functions
// associated with panels at interface (row) and on boundary (col)
Eigen::MatrixXd ComputeGeneralMatrix(
    const parametricbem2d::ParametrizedMesh &mesh,
    const parametricbem2d::AbstractBEMSpace &space,
    const parametricbem2d::AbstractBEMSpace &space_p,
    const Dims &dims, const Dims &dims_p,
    const AbstractSingularKernel &kernel,
    const AbstractFactor &F, const AbstractFactor &G,
    const AbstractVelocityField &nu, unsigned order) {
  parametricbem2d::PanelVector panels = mesh.getPanels();
  unsigned Q = space.getQ();
  unsigned Q_p = space_p.getQ();
  unsigned numpanels = mesh.getNumPanels();
  unsigned numpanels_i = mesh.getSplit();
  QuadRule GaussQR = getGaussQR(order);
  // Initialize matrix
  Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(dims.i, dims_p.d + dims_p.n);
  
  for (unsigned i = 0; i < numpanels_i; ++i) {
    for (unsigned j = numpanels_i; j < numpanels; ++j) {
      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      parametricbem2d::AbstractParametrizedCurve &pi_p = *panels[j];

      // Evaluate interaction matrix
      Eigen::MatrixXd panel_mat(Q, Q_p);
      panel_mat = IntegralGeneral(pi, pi_p, space, space_p,
                                  kernel, F, G, nu, GaussQR);

      // Local to global mapping of elements in interaction matrix
      for (unsigned I = 0; I < Q; ++I) {
        for (unsigned J = 0; J < Q_p; ++J) {
          unsigned II = space.LocGlobMap2(I + 1, i + 1, mesh) - 1;
          unsigned JJ = space.LocGlobMap2(J + 1, j + 1, mesh) - 1;
          mat(II, JJ) += panel_mat(I, J);
        }
      }
    }
  }

  return mat;
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
  Kernel1 kernel;
  Factor1 F;

  Eigen::MatrixXd mat = ComputeSingularMatrix(
    mesh, space_n, space_n, dims_n, dims_n, kernel, F, F, nu, order);
  return -c * state_sol.psi_i.dot(mat * adj_sol.psi_i);
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

  double force = ComputeShapeDerivative(mesh, space_d, space_n, nu, dir_sel,
      g, eta, epsilon1, epsilon1, state_sol, adj_sol, order);
      
  std::cout << "force = " << force << std::endl;
  return force;
}
} // namespace transmission_bem

#endif // FORCECALCULATIONBEMHPP