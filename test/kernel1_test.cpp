#include "factors.hpp"
#include "force_calculation_bem.hpp"
#include "velocity_fields.hpp"

#include <iostream>

#include "abstract_parametrized_curve.hpp"
#include "discontinuous_space.hpp"
#include "integral_gauss.hpp"
#include "parametrized_circular_arc.hpp"
#include "parametrized_line.hpp"
#include "parametrized_mesh.hpp"
#include <Eigen/Dense>

double sqrt_epsilon = transmission_bem::sqrt_epsilon;
double tol = transmission_bem::tol;

// snippet from FCSC
Eigen::MatrixXd EvaluateSecond(const parametricbem2d::ParametrizedMesh &mesh,
                      const transmission_bem::Dims &dims,
                      const transmission_bem::Indices &ind,
                      const AbstractVelocityField &nu, unsigned order) {
  // The BEM space used for solving the state and adjoint problems
  parametricbem2d::DiscontinuousSpace<0> space;
  // Getting the panels
  parametricbem2d::PanelVector panels = mesh.getPanels();
  // Number of reference shape functions for the space
  unsigned q = space.getQ();
  // Getting the number of panels
  unsigned numpanels_i = mesh.getSplit();
  // Initializing the R_{ij} matrix
  Eigen::MatrixXd R = Eigen::MatrixXd::Constant(dims.i, dims.i, 0);

  // Looping over the panels and evaluating the local integrals
  for (unsigned i = 0; i < numpanels_i; ++i) {
    for (unsigned j = 0; j < numpanels_i; ++j) {
      // The panels pi and pi' for which the local integral has to be
      // evaluated
      parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
      parametricbem2d::AbstractParametrizedCurve &pi_p = *panels[j];

      // Going over reference shape functions
      for (unsigned k = 0; k < q; ++k) {
        for (unsigned l = 0; l < q; ++l) {
          double local_integral = 0;
          // coinciding panels case
          if (i == j) {
            auto integrand = [&](double t, double s) {
              double non_singular = space.evaluateShapeFunction(k, t) *
                                    space.evaluateShapeFunction(l, s) *
                                    pi.Derivative(t).norm() *
                                    pi.Derivative(s).norm();
              double singular;
              // Direct evaluation when away from singularity
              if (fabs(s - t) > sqrt_epsilon) {

                singular = (pi(s) - pi(t)).dot(nu(pi(s)) - nu(pi(t))) /
                           (pi(s) - pi(t)).squaredNorm();

              }
              // stable evaluation near singularity using Taylor expansion
              else {
                singular = pi.Derivative((s + t) / 2.)
                               .dot(nu.grad(pi((s + t) / 2.)).transpose() *
                                    pi.Derivative((s + t) / 2.)) /
                           (pi.Derivative((s + t) / 2.)).squaredNorm();
              }

              return singular * non_singular;
            };
            // Function for upper limit of the integral
            auto ul = [&](double x) { return 1.; };
            // Function for lower limit of the integral
            auto ll = [&](double x) { return -1.; };
            local_integral = parametricbem2d::ComputeDoubleIntegral(
                integrand, -1., 1., ll, ul, order);
          }

          // Adjacent panels case
          else if ((pi(1) - pi_p(-1)).norm() / 100. < tol ||
                   (pi(-1) - pi_p(1)).norm() / 100. < tol) {
            // Swap is used to check whether pi(1) = pi'(-1) or pi(-1) =
            // pi'(1)
            bool swap = (pi(1) - pi_p(-1)).norm() / 100. > tol;
            // Panel lengths for local arclength parametrization
            double length_pi =
                2 * pi.Derivative(swap ? -1 : 1)
                        .norm(); // Length for panel pi to ensure norm of
                                 // arclength parametrization is 1 at the
                                 // common point
            double length_pi_p =
                2 * pi_p.Derivative(swap ? 1 : -1)
                        .norm(); // Length for panel pi_p to ensure norm of
                                 // arclength parametrization is 1 at the
                                 // common point

            // Local integrand in polar coordinates
            auto integrand = [&](double phi, double r) {
              // Converting polar coordinates to local arclength coordinates
              double s_pr = r * cos(phi);
              double t_pr = r * sin(phi);
              // Converting local arclength coordinates to reference interval
              // coordinates
              double s = swap ? 1 - 2 * s_pr / length_pi_p
                              : 2 * s_pr / length_pi_p - 1;
              double t =
                  swap ? 2 * t_pr / length_pi - 1 : 1 - 2 * t_pr / length_pi;
              // reference interval coordinates corresponding to zeros in
              // arclength coordinates
              double s0 = swap ? 1 : -1; // different implementation
              double t0 = swap ? -1 : 1;

              double non_singular =
                  space.evaluateShapeFunction(k, t) *
                  space.evaluateShapeFunction(l, s) * pi.Derivative(t).norm() *
                  pi_p.Derivative(s).norm() * (4 / length_pi / length_pi_p);
              double singular;
              // Direct evaluation away from the singularity
              if (r > sqrt_epsilon) {
                singular = (pi_p(s) - pi(t)).dot(nu(pi_p(s)) - nu(pi(t))) /
                           (pi_p(s) - pi(t)).squaredNorm();

              }
              // Stable evaluation near singularity using Taylor expansion
              else {
                singular =
                    (cos(phi) * pi_p.Derivative(s0) * 2 / length_pi_p +
                     sin(phi) * pi.Derivative(t0) * 2 / length_pi)
                        .dot(nu.grad(pi(t0)).transpose() *
                             (cos(phi) * pi_p.Derivative(s0) * 2 / length_pi_p +
                              sin(phi) * pi.Derivative(t0) * 2 / length_pi)) /
                    (1 + sin(2 * phi) *
                             pi.Derivative(t0).dot(pi_p.Derivative(s0)) * 4 /
                             length_pi / length_pi_p);
              }
              // Including the Jacobian of transformation 'r'
              return r * singular * non_singular;
            };
            // Getting the split point for integral over the angle in polar
            // coordinates
            double alpha = std::atan(length_pi / length_pi_p);
            // Defining upper and lower limits of inner integrals
            auto ll = [&](double phi) { return 0; };
            auto ul1 = [&](double phi) { return length_pi_p / cos(phi); };
            auto ul2 = [&](double phi) { return length_pi / sin(phi); };
            // Computing the local integral
            local_integral = parametricbem2d::ComputeDoubleIntegral(
                integrand, 0, alpha, ll, ul1, order);
            local_integral += parametricbem2d::ComputeDoubleIntegral(
                integrand, alpha, M_PI / 2., ll, ul2, order);
          }

          // General case
          else {
            // Local integral
            auto integrand = [&](double t, double s) {
              return space.evaluateShapeFunction(k, t) *
                     space.evaluateShapeFunction(l, s) *
                     pi.Derivative(t).norm() * pi_p.Derivative(s).norm() *
                     (pi_p(s) - pi(t)).dot(nu(pi_p(s)) - nu(pi(t))) /
                     (pi_p(s) - pi(t)).squaredNorm();
            };
            // Function for upper limit of the integral
            auto ul = [&](double x) { return 1.; };
            // Function for lower limit of the integral
            auto ll = [&](double x) { return -1.; };
            local_integral = parametricbem2d::ComputeDoubleIntegral(
                integrand, -1, 1, ll, ul, order);
          }

          // Local to global mapping
          unsigned II = space.LocGlobMap2(k + 1, i + 1, mesh) - 1;
          unsigned JJ = space.LocGlobMap2(l + 1, j + 1, mesh) - 1;
          R(II, JJ) += local_integral;
        }
      }
    }
  }
  // with linear velocity, R matrix should contain the multiples of panel
  // lengths std::cout << "R matrix \n" << R << std::endl;
  return transmission_bem::Slice(R, ind.i, ind.i);;
}

int main() {
  std::cout << "Test kernel 1" << std::endl;
  std::cout << "####################################" << std::endl;
  // Gauss quadrature order
  unsigned order = 16;
  std::cout << "Gauss Quadrature used with order = " << order << std::endl;

  // Inner square vertices
  Eigen::Vector2d NE(1, 1);
  Eigen::Vector2d NW(0, 1);
  Eigen::Vector2d SE(1, 0);
  Eigen::Vector2d SW(0, 0);
  // Inner square edges
  parametricbem2d::ParametrizedLine ir(NE, SE); // right
  parametricbem2d::ParametrizedLine it(NW, NE); // top
  parametricbem2d::ParametrizedLine il(SW, NW); // left
  parametricbem2d::ParametrizedLine ib(SE, SW); // bottom

  // Outer square vertices
  Eigen::Vector2d NEo(1.1, 1.1);
  Eigen::Vector2d NWo(-1.1, 1.1);
  Eigen::Vector2d SEo(1.1, -1.1);
  Eigen::Vector2d SWo(-1.1, -1.1);
  // Outer square edges
  parametricbem2d::ParametrizedLine Or(SEo, NEo); // right
  parametricbem2d::ParametrizedLine ot(NEo, NWo); // top
  parametricbem2d::ParametrizedLine ol(NWo, SWo); // left
  parametricbem2d::ParametrizedLine ob(SWo, SEo); // bottom
  
  unsigned nsplit = 32;

  // Panels for the edges of the inner square
  parametricbem2d::PanelVector panels_ir(ir.split(nsplit));
  parametricbem2d::PanelVector panels_ib(ib.split(nsplit));
  parametricbem2d::PanelVector panels_il(il.split(nsplit));
  parametricbem2d::PanelVector panels_it(it.split(nsplit));

  // Panels for the edges of outer square
  parametricbem2d::PanelVector panels_or(Or.split(nsplit));
  parametricbem2d::PanelVector panels_ot(ot.split(2*nsplit));
  parametricbem2d::PanelVector panels_ol(ol.split(2*nsplit));
  parametricbem2d::PanelVector panels_ob(ob.split(3*nsplit));

  // Creating the ParametricMesh object
  parametricbem2d::PanelVector panels;

  /*
  panels.insert(panels.end(), panels_ir.begin(), panels_ir.end());
  panels.insert(panels.end(), panels_ib.begin(), panels_ib.end());
  panels.insert(panels.end(), panels_il.begin(), panels_il.end());
  panels.insert(panels.end(), panels_it.begin(), panels_it.end());
  */
  Eigen::Vector2d center(0.5, 0.5);
  double r = 0.5;
  parametricbem2d::ParametrizedCircularArc icirc(center, r, 0., 2. * M_PI);
  parametricbem2d::PanelVector panels_i(icirc.split(4*nsplit));
  panels.insert(panels.end(), panels_i.begin(), panels_i.end());

  panels.insert(panels.end(), panels_or.begin(), panels_or.end());
  panels.insert(panels.end(), panels_ot.begin(), panels_ot.end());
  panels.insert(panels.end(), panels_ol.begin(), panels_ol.end());
  panels.insert(panels.end(), panels_ob.begin(), panels_ob.end());

  parametricbem2d::ParametrizedMesh mesh(panels);

  auto dir_sel = [](const Eigen::Vector2d& x) {
    return (x[0] - 1.1 > -1e-7 || x[0] + 1.1 < 1e-7);
  };

  parametricbem2d::DiscontinuousSpace<0> space_n;
  NuRadial nu;

  // Compute Space Information
  transmission_bem::Dims dims_n;
  transmission_bem::Indices ind_n;
  transmission_bem::ComputeNeuSpaceInfo(
      mesh, space_n, dir_sel, &dims_n, &ind_n);

  transmission_bem::Kernel1 kernel;
  transmission_bem::Factor1 F;
  Eigen::MatrixXd mat = ComputeSingularMatrix(mesh, space_n, space_n,
      dims_n, dims_n, ind_n, ind_n, kernel, F, F, nu, order);
  Eigen::MatrixXd mat_ref = EvaluateSecond(mesh, dims_n, ind_n, nu, order);

  // std::cout << "mat =\n" << mat << std::endl;
  // std::cout << "mat_ref =\n" << mat_ref << std::endl;
  std::cout << "mat.norm = " << mat.norm() << std::endl;
  std::cout << "diff = " << (mat - mat_ref).norm() << std::endl;

  return 0;
}