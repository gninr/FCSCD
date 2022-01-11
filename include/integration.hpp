#ifndef INTEGRATIONHPP
#define INTEGRATIONHPP

#include "factors.hpp"
#include "transmission_bem.hpp"
#include "velocity_fields.hpp"

#include <iostream>
#include <limits>
#include <math.h>

#include "abstract_bem_space.hpp"
#include "abstract_parametrized_curve.hpp"
#include "gauleg.hpp"
#include "logweight_quadrature.hpp"
#include "parametrized_mesh.hpp"
#include <Eigen/Dense>

namespace transmission_bem {
//double c = -1. / (2. * M_PI); // defined in factors.hpp
double sqrt_epsilon = 1e5 * std::sqrt(std::numeric_limits<double>::epsilon());
double tol = std::numeric_limits<double>::epsilon();

// Integrate over coinciding panels
Eigen::MatrixXd IntegralCoinciding(
    const parametricbem2d::AbstractParametrizedCurve &pi,
    const parametricbem2d::AbstractBEMSpace &space,
    const parametricbem2d::AbstractBEMSpace &space_p,
    const AbstractKernel &kernel,
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
          return log(singular) * non_singular;
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

        mat(i, j) = c * 0.5 * (i1 + i2); // c = -1 / (2 * M_PI)
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
    const AbstractKernel &kernel,
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

        // Integral 2

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

        mat(i, j) = c * (0.5 * (i11 + i12) + (i21 + i22));
      }
    }
  }

  // Other kernels
  else {
    double scale = swap ? 2 / length_pi : -2 / length_pi;
    double scale_p = swap ? -2 / length_pi_p : 2 / length_pi_p;
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
            double s0 = swap ? -1 : 1;
            double t0 = swap ? 1 : -1;
            singular = kernel.stable_pr(pi, pi_p, nu, scale, scale_p,
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

Eigen::MatrixXd ComputeMatrix(
    const parametricbem2d::ParametrizedMesh &mesh,
    const parametricbem2d::AbstractBEMSpace &space,
    const parametricbem2d::AbstractBEMSpace &space_p,
    unsigned dim, unsigned dim_p,
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
  Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(dim, dim_p);

  // Elements generated by global shape functions associated with panels
  // at interface
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

  // Elements generated by global shape functions associated with panels
  // on boundary (row) and at interface (col)
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
          unsigned JJ = space_p.LocGlobMap2(J + 1, j + 1, mesh) - 1;
          mat(II, JJ) += panel_mat(I, J);
        }
      }
    }
  }

  return mat;
}
} // namespace transission_bem

#endif // INTEGRATIONHPP