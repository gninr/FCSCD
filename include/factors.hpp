#ifndef FACTORSHPP
#define FACTORSHPP

#include "velocity_fields.hpp"

#include <functional>
#include <math.h>

#include "abstract_bem_space.hpp"
#include "abstract_parametrized_curve.hpp"
#include <Eigen/Dense>

namespace transmission_bem {
class AbstractKernel {
public:
  // Directe evaluation
  virtual double operator()(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double s, double t) const = 0;
};

class AbstractSingularKernel : public AbstractKernel {
public:
  // Stable evaluation near singularity on identical panels
  virtual double stable_st(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const AbstractVelocityField &nu,
      double mid) const = 0;

  // Stable evaluation near singularity on adjoint panels
  // in polar coordinates including Jacobian 'r'
  virtual double stable_pr(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double length_pi, double length_pi_p,
      double r, double phi, double s0, double t0) const = 0;
};

class Kernel1 : public AbstractSingularKernel {
public:
  double operator()(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double s, double t) const {
    Eigen::Vector2d x = pi(s);
    Eigen::Vector2d y = pi_p(t);
    return (x - y).dot(nu(x) - nu(y)) / (x - y).squaredNorm();
  }

  double stable_st(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const AbstractVelocityField &nu,
      double mid) const {
    Eigen::Vector2d gamma_dot = pi.Derivative(mid);
    return gamma_dot.dot(nu.grad(pi(mid)) * gamma_dot) /
           gamma_dot.squaredNorm();
  }

  double stable_pr(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double length_pi, double length_pi_p,
      double r, double phi, double s0, double t0) const {
    Eigen::Vector2d gamma_dot = pi.Derivative(s0);
    Eigen::Vector2d gamma_dot_p = pi_p.Derivative(t0);
    Eigen::Vector2d a = cos(phi) * gamma_dot * 2 / length_pi +
                        sin(phi) * gamma_dot_p * 2 / length_pi_p;
    double b = 1 + sin(2 * phi) * gamma_dot.dot(gamma_dot_p) *
                       4 / length_pi / length_pi_p;
    return r * a.dot(nu.grad(pi(s0)) * a) / b;
  }
};

class AbstractFactor {
public:
  virtual double operator()(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractBEMSpace &space,
      const AbstractVelocityField &nu, unsigned q, double s) const = 0;
};

class Factor1 : public AbstractFactor {
public:
  double operator()(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractBEMSpace &space,
      const AbstractVelocityField &nu, unsigned q, double s) const {
    return space.evaluateShapeFunction(q, s) * pi.Derivative(s).norm();
  }
};


} // namespace transmission_bem

#endif // FACTORSHPP