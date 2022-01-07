#ifndef FORCECALCULATIONBEMHPP
#define FORCECALCULATIONBEMHPP

#include "factors.hpp"
#include "transmission_bem.hpp"
#include "velocity_fields.hpp"

#include <functional>
#include <iostream>
#include <limits>
#include <math.h>
#include <type_traits>

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

// Integrate over coinciding panels
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

  // Log kernel
  if (typeid(kernel) == typeid(LogKernel)) {
    for (unsigned i = 0; i < Q; ++i) {
      for (unsigned j = 0; j < Q_p; ++j) {
        // Local integrand
        auto integrand1 = [&](double s, double t) {
          double non_singular =
                    F(pi, space_p, nu, j, t) * G(pi, space, nu, i, s);
          double singular;
          // Direct evaluation when away from singularity
          if (fabs(s - t) > sqrt_epsilon) {
            singular = (pi(s) - pi(t)).squaredNorm() / (s - t) / (s - t);
          }
          // Stable evaluation near singularity using Tayler expansion
          else {
            //std::cout << "stable_st" << std::endl;
            singular = pi.Derivative(0.5 * (s + t)).squaredNorm();
          }
          return 0.5 * log(singular) * non_singular;
        };

        // Split integral into two parts
        double i1 = 0., i2 = 0.;

        for (unsigned k = 0; k < N; ++k) {
          for (unsigned l = 0; l < N; ++l) {
            i1 += GaussQR.w(k) * GaussQR.w(l) *
                      integrand1(GaussQR.x(k), GaussQR.x(l));
          }
        }

        // Local integrand in transformed coordinates
        auto integrand2 = [&](double w, double z) {
          return F(pi, space_p, nu, j, 0.5 * (w - z)) *
                     G(pi, space, nu, i, 0.5 * (w + z)) +
                 F(pi, space_p, nu, j, 0.5 * (w + z)) *
                     G(pi, space, nu, i, 0.5 * (w - z));
        };

        auto inner2_z = [&](double z) {
          auto integrand2_w = [&](double w) { return integrand2(w, z); };
          return parametricbem2d::ComputeIntegral(
                     integrand2_w, -2 + z, 2 - z, GaussQR);
        };

        i2 = parametricbem2d::ComputeLoogIntegral(inner2_z, 2, GaussQR);

        mat(i, j) = i1 + 0.5 * i2;
      }
    }
  }

  // Other kernels
  else {
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
            //std::cout << "stable_st" << std::endl;
            singular = kernel.stable_st(pi, nu, s, t);
          }
          return singular * non_singular;
        };

        double integral = 0;
        for (unsigned k = 0; k < N; ++k) {
          for (unsigned l = 0; l < N; ++l) {
            integral += GaussQR.w(k) * GaussQR.w(l) *
                            integrand(GaussQR.x(k), GaussQR.x(l));
          }
        }

        mat(i, j) = integral;
      }
    }
  }

  return mat;
}

// Integrate over adjoint panels
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

  // Log kernel
  if (typeid(kernel) == typeid(LogKernel)) {
    for (unsigned i = 0; i < Q; ++i) {
      for (unsigned j = 0; j < Q_p; ++j) {
        // Integral 1

        // Local integrand in polar coordinates
        auto integrand1 = [&](double r, double phi) {
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
            singular = (pi(s) - pi_p(t)).squaredNorm() / r / r;
          }
          // Stable evaluation near singularity using Tayler expansion
          else {
            //std::cout << "stable_pr" << std::endl;
            singular =
                1 + sin(2 * phi) * pi.Derivative(s).dot(pi_p.Derivative(t)) *
                        4 / length_pi / length_pi_p;
          }
          return r * log(singular) * non_singular;
        };
        
        // Split integral1 into two parts
        double alpha = atan(length_pi_p / length_pi);
        double i11 = 0., i12 = 0.;
        // part 1 (phi from 0 to alpha)
        for (unsigned k = 0; k < N; ++k) {
          double phi = alpha / 2 * (1 + GaussQR.x(k));
          // Upper limit for inner integral
          double rmax = length_pi / cos(phi);
          // Evaluate inner integral
          double inner = 0.;
          for (unsigned l = 0; l < N; ++l) {
            double r = rmax / 2 * (1 + GaussQR.x(l));
            inner += GaussQR.w(l) * integrand1(r, phi);
          }
          inner *= rmax / 2;
          i11 += GaussQR.w(k) * inner * alpha / 2;
        }
        // part 2 (phi from alpha to pi/2)
        for (unsigned k = 0; k < N; ++k) {
          double phi = GaussQR.x(k) * (M_PI / 2. - alpha) / 2. +
                           (M_PI / 2. + alpha) / 2.;
          // Upper limit for inner integral
          double rmax = length_pi_p / sin(phi);
          // Evaluate inner integral
          double inner = 0.;
          for (unsigned l = 0; l < N; ++l) {
            double r = rmax / 2 * (1 + GaussQR.x(l));
            inner += GaussQR.w(l) * integrand1(r, phi);
          }
          inner *= rmax / 2;
          i12 += GaussQR.w(k) * inner * (M_PI / 2. - alpha) / 2.;
        }

        // Integral 1

        auto integrand2 = [&](double r, double phi) {
          return r * F(pi_p, space_p, nu, j, r * sin(phi)) *
                     G(pi, space, nu, i, r * cos(phi)) *
                 4 / length_pi / length_pi_p;
        };

        // Split integral2 into two parts
        double i21 = 0., i22 = 0.;
        // part 1 (phi from 0 to alpha)
        auto inner21 = [&](double phi) {
          auto in = [&](double r) { return integrand2(r, phi); };
          return parametricbem2d::ComputeLoogIntegral(
                     in, length_pi / cos(phi), GaussQR);
        };
        i21 = parametricbem2d::ComputeIntegral(
                  inner21, 0, alpha, GaussQR);
        // part 2 (phi from alpha to pi/2)
        auto inner22 = [&](double phi) {
          auto in = [&](double r) { return integrand2(r, phi); };
          return parametricbem2d::ComputeLoogIntegral(
                     in, length_pi_p / sin(phi), GaussQR);
        };
        i22 = parametricbem2d::ComputeIntegral(
                  inner22, alpha, M_PI / 2, GaussQR);

        mat(i, j) = 0.5 * (i11 + i12) + (i21 + i22);
      }
    }
  }

  // Other kernels
  else {
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
            //std::cout << "stable_pr" << std::endl;
            singular = kernel.stable_pr(pi, pi_p, nu, length_pi, length_pi_p,
                                        r, phi, s, t);
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
          double phi = GaussQR.x(k) * (M_PI / 2. - alpha) / 2. +
                           (M_PI / 2. + alpha) / 2.;
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
  }

  return mat;
}

// Integrate over disjoint panels
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
                   F(pi_p, space_p, nu, j, t) * G(pi, space, nu, i, s);
      };

      double integral = 0;
      for (unsigned k = 0; k < N; ++k) {
        for (unsigned l = 0; l < N; ++l) {
          integral += GaussQR.w(k) * GaussQR.w(l) *
                          integrand(GaussQR.x(k), GaussQR.x(l));
        }
      }
      mat(i, j) = integral;
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
    const Indices &ind, const Indices &ind_p,
    const AbstractSingularKernel &kernel,
    const AbstractFactor &F, const AbstractFactor &G,
    const AbstractVelocityField &nu, unsigned order) {
  parametricbem2d::PanelVector panels = mesh.getPanels();
  unsigned Q = space.getQ();
  unsigned Q_p = space_p.getQ();
  unsigned numpanels_i = mesh.getSplit();
  QuadRule GaussQR = getGaussQR(order);
  // Initialize matrix
  Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(dims.all, dims_p.all);

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
        for (unsigned J = 0; J < Q_p; ++J) {
          unsigned II = space.LocGlobMap2(I + 1, i + 1, mesh) - 1;
          unsigned JJ = space_p.LocGlobMap2(J + 1, j + 1, mesh) - 1;
          mat(II, JJ) += panel_mat(I, J);
        }
      }
    }
  }

  return Slice(mat, ind.i, ind_p.i);
}

// Compute matrix with elements generated by global shape functions
// associated with panels on boundary (row) and at interface (col)
Eigen::MatrixXd ComputeGeneralMatrix(
    const parametricbem2d::ParametrizedMesh &mesh,
    const parametricbem2d::AbstractBEMSpace &space,
    const parametricbem2d::AbstractBEMSpace &space_p,
    const Dims &dims, const Dims &dims_p,
    const Indices &ind, const Indices &ind_p,
    const AbstractKernel &kernel,
    const AbstractFactor &F, const AbstractFactor &G,
    const AbstractVelocityField &nu, unsigned order) {
  parametricbem2d::PanelVector panels = mesh.getPanels();
  unsigned Q = space.getQ();
  unsigned Q_p = space_p.getQ();
  unsigned numpanels = mesh.getNumPanels();
  unsigned numpanels_i = mesh.getSplit();
  QuadRule GaussQR = getGaussQR(order);
  // Initialize matrix
  Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(dims.all, dims_p.all);

  for (unsigned i = numpanels_i; i < numpanels; ++i) {
    for (unsigned j = 0; j < numpanels_i; ++j) {
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
          unsigned JJ = space_p.LocGlobMap2(J + 1, j + 1, mesh) - 1;
          mat(II, JJ) += panel_mat(I, J);
        }
      }
    }
  }

  Eigen::MatrixXd res(dims.d + dims.n, dims_p.i);
  res << Slice(mat, ind.d, ind_p.i), Slice(mat, ind.n, ind_p.i);
  return res;
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

  {
    Kernel1 kernel;
    Factor1 F;
    Eigen::MatrixXd mat = ComputeSingularMatrix(mesh, space_n, space_n,
        dims_n, dims_n, ind_n, ind_n, kernel, F, F, nu, order);
    // a_V_ii(psi_i, pi_i)
    force += -c * state_sol.psi_i.dot(mat * adj_sol.psi_i);
  }

  {
    Kernel2 kernel;
    Factor1 F;
    Eigen::MatrixXd mat = ComputeGeneralMatrix(mesh, space_n, space_n,
        dims_n, dims_n, ind_n, ind_n, kernel, F, F, nu, order);
    // a_V_iD(psi_i, pi)
    force += c * adj_sol.psi.dot(
                     mat.block(0, 0, dims_n.d, dims_n.i) * state_sol.psi_i);
    // a_V_Di(psi, pi_i)
    force += c * state_sol.psi.dot(
                     mat.block(0, 0, dims_n.d, dims_n.i) * adj_sol.psi_i);
    // b_V_Ni(eta, pi_i)
    force +=
        c * eta_interp.dot(
                mat.block(dims_n.d, 0, dims_n.n, dims_n.i) * adj_sol.psi_i);
  }

  {
    Kernel3 kernel;
    Factor1 F;
    Eigen::MatrixXd mat = ComputeSingularMatrix(mesh, space_n, space_d,
        dims_n, dims_d, ind_n, ind_d, kernel, F, F, nu, order);
    // a_K_ii(u_i, pi_i)
    force += c * adj_sol.psi_i.dot(mat * state_sol.u_i);
    // a_K_ii(rho_i, psi_i)
    force += c * state_sol.psi_i.dot(mat * adj_sol.u_i);
  }

  {
    Kernel4 kernel;
    Factor1 F;
    Eigen::MatrixXd mat = ComputeGeneralMatrix(mesh, space_n, space_d,
        dims_n, dims_d, ind_n, ind_d, kernel, F, F, nu, order);
    // a_K_iD(u_i, pi)
    force += c * adj_sol.psi.dot(
                     mat.block(0, 0, dims_n.d, dims_d.i) * state_sol.u_i);
    // a_K_iD(rho_i, psi)
    force += c * state_sol.psi.dot(
                     mat.block(0, 0, dims_n.d, dims_n.i) * adj_sol.u_i);
    // b_K_iN(rho_i, eta)
    force += c * eta_interp.dot(
                     mat.block(dims_n.d, 0, dims_n.n, dims_n.i) * adj_sol.u_i);
  }

  {
    Kernel5 kernel;
    Factor1 F;
    Eigen::MatrixXd mat = ComputeGeneralMatrix(mesh, space_d, space_n,
        dims_d, dims_n, ind_d, ind_n, kernel, F, F, nu, order);
    // a_K_Ni(u, pi_i)
    force += c * state_sol.u.dot(
                     mat.block(dims_d.d, 0, dims_d.n, dims_n.i) *
                         adj_sol.psi_i);
    // a_K_Ni(rho, psi_i)
    force += c * adj_sol.u.dot(
                     mat.block(dims_d.d, 0, dims_d.n, dims_n.i) *
                         state_sol.psi_i);
    // b_K_Di(g, pi_i)
    force +=
        c * g_interp.dot(
                mat.block(0, 0, dims_d.d, dims_n.i) * state_sol.psi_i);
  }

  {
    Kernel1 kernel;
    Factor2 F;
    Eigen::MatrixXd mat = ComputeSingularMatrix(mesh, space_d, space_d,
        dims_d, dims_d, ind_d, ind_d, kernel, F, F, nu, order);
    // a_W_ii(u_i, rho_i) term1
    force += -c * adj_sol.u_i.dot(mat * state_sol.u_i);
  }

  {
    LogKernel kernel;
    Factor3 F;
    Factor2 G;
    Eigen::MatrixXd mat = ComputeSingularMatrix(mesh, space_d, space_d,
        dims_d, dims_d, ind_d, ind_d, kernel, F, G, nu, order);
    // a_W_ii(u_i, rho_i) term2
    force += c * adj_sol.u_i.dot(mat * state_sol.u_i);
    // a_W_ii(u_i, rho_i) term4
    force += c * state_sol.u_i.dot(mat * adj_sol.u_i);
  }

  {
    LogKernel kernel;
    Factor4 F;
    Factor2 G;
    Eigen::MatrixXd mat = ComputeSingularMatrix(mesh, space_d, space_d,
        dims_d, dims_d, ind_d, ind_d, kernel, F, G, nu, order);
    // a_W_ii(u_i, rho_i) term3
    force += c * adj_sol.u_i.dot(mat * state_sol.u_i);
    // a_W_ii(u_i, rho_i) term5
    force += c * state_sol.u_i.dot(mat * adj_sol.u_i);
  }

  {
    Kernel1 kernel;
    Factor2 F;
    Eigen::MatrixXd mat = ComputeGeneralMatrix(mesh, space_d, space_d,
        dims_d, dims_d, ind_d, ind_d, kernel, F, F, nu, order);
    // a_W_iN(u_i, rho) term1
    force += -c * adj_sol.u.dot(
                      mat.block(dims_d.d, 0, dims_d.n, dims_d.i) *
                          state_sol.u_i);
    // a_W_Ni(u, rho_i) term1
    force += -c * state_sol.u.dot(
                      mat.block(dims_d.d, 0, dims_d.n, dims_d.i) *
                          adj_sol.u_i);
    // b_W_Di(g, rho_i) term1
    force += -c * g_interp.dot(
                      mat.block(0, 0, dims_d.d, dims_d.i) * adj_sol.u_i);
  }

  {
    LogKernel kernel;
    Factor3 F;
    Factor2 G;
    Eigen::MatrixXd mat = ComputeGeneralMatrix(mesh, space_d, space_d,
        dims_d, dims_d, ind_d, ind_d, kernel, F, G, nu, order);
    // a_W_iN(u_i, rho) term2
    force += c * adj_sol.u.dot(
                      mat.block(dims_d.d, 0, dims_d.n, dims_d.i) *
                          state_sol.u_i);
    // a_W_Ni(u, rho_i) term2
    force += c * state_sol.u.dot(
                      mat.block(dims_d.d, 0, dims_d.n, dims_d.i) *
                          adj_sol.u_i);
    // b_W_Di(g, rho_i) term2
    force += -c * g_interp.dot(
                      mat.block(0, 0, dims_d.d, dims_d.i) * adj_sol.u_i);
  }

  {
    LogKernel kernel;
    Factor4 F;
    Factor2 G;
    Eigen::MatrixXd mat = ComputeGeneralMatrix(mesh, space_d, space_d,
        dims_d, dims_d, ind_d, ind_d, kernel, F, G, nu, order);
    // a_W_iN(u_i, rho) term3
    force += c * adj_sol.u.dot(
                      mat.block(dims_d.d, 0, dims_d.n, dims_d.i) *
                          state_sol.u_i);
    // a_W_Ni(u, rho_i) term3
    force += c * state_sol.u.dot(
                      mat.block(dims_d.d, 0, dims_d.n, dims_d.i) *
                          adj_sol.u_i);
    // b_W_Di(g, rho_i) term3
    force += -c * g_interp.dot(
                      mat.block(0, 0, dims_d.d, dims_d.i) * adj_sol.u_i);
  }

  {
    Eigen::MatrixXd mat = ComputeMatrixJ(mesh, space_d, space_n,
        dims_d, dims_n, ind_d, ind_n, nu, order);
    // J(u_i, psi_i, u, psi)
    force += 0.5 * (epsilon2 - epsilon1) *
                 state_sol.u_i.dot(mat * adj_sol.psi_i);
  }

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