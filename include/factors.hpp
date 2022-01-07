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
  // Stable evaluation near singularity on coinciding panels
  virtual double stable_st(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const AbstractVelocityField &nu,
      double s, double t) const = 0;

  // Stable evaluation near singularity on adjoint panels
  // in polar coordinates including Jacobian 'r'
  virtual double stable_pr(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double length_pi, double length_pi_p,
      double r, double phi, double s, double t) const = 0;
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
      double s, double t) const {
    double mid = 0.5 * (s + t);
    Eigen::Vector2d gamma_dot = pi.Derivative(mid);
    return gamma_dot.dot(nu.grad(pi(mid)) * gamma_dot) /
           gamma_dot.squaredNorm();
  }

  double stable_pr(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double length_pi, double length_pi_p,
      double r, double phi, double s, double t) const {
    Eigen::Vector2d gamma_dot = pi.Derivative(s);
    Eigen::Vector2d gamma_dot_p = pi_p.Derivative(t);
    Eigen::Vector2d a = cos(phi) * gamma_dot * 2 / length_pi +
                        sin(phi) * gamma_dot_p * 2 / length_pi_p;
    double b = 1 + sin(2 * phi) * gamma_dot.dot(gamma_dot_p) *
                       4 / length_pi / length_pi_p;
    return r * a.dot(nu.grad(pi(s)) * a) / b;
  }
};

class Kernel2 : public AbstractKernel {
public:
  double operator()(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double s, double t) const {
    Eigen::Vector2d x = pi(s);
    Eigen::Vector2d y = pi_p(t);
    return (x - y).dot(nu(y)) / (x - y).squaredNorm();
  }
};

class Kernel3 : public AbstractSingularKernel {
public:
  double operator()(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double s, double t) const {
    Eigen::Vector2d x = pi(s);
    Eigen::Vector2d y = pi_p(t);
    Eigen::Vector2d a = x - y;

    Eigen::Vector2d tangent = pi_p.Derivative(t);
    Eigen::Vector2d normal(tangent(1), -tangent(0));
    normal /= normal.norm();

    double res = normal.dot(nu(x) - nu(y))
                 - 2 * a.dot(normal) * a.dot(nu(x) - nu(y)) / a.squaredNorm()
                 - a.dot(nu.grad(y).transpose() * normal)
                 + a.dot(normal) * normal.dot(nu.grad(y).transpose() * normal);
    return res / a.squaredNorm();
  }

  double stable_st(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const AbstractVelocityField &nu,
      double s, double t) const {
    double mid = 0.5 * (s + t);
    Eigen::Vector2d c = pi(mid);
    Eigen::Vector2d gamma_dot = pi.Derivative(mid);
    Eigen::Vector2d gamma_ddot = pi.DoubleDerivative(mid);

    Eigen::Vector2d tangent = pi.Derivative(t);
    Eigen::Vector2d normal(tangent(1), -tangent(0));
    normal /= normal.norm();

    Eigen::Vector2d v;
    v << (nu.dgrad1(c) * gamma_dot).dot(gamma_dot),
         (nu.dgrad2(c) * gamma_dot).dot(gamma_dot);

    double res = 0.5 * normal.dot(v)
                 - gamma_ddot.dot(normal)
                     * gamma_dot.dot(nu.grad(c) * gamma_dot)
                     / gamma_dot.squaredNorm()
                 + 0.5 * gamma_ddot.dot(normal)
                     * normal.dot(nu.grad(c) * normal);
    return res / gamma_dot.squaredNorm();
  }

  double stable_pr(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double length_pi, double length_pi_p,
      double r, double phi, double s, double t) const {
    Eigen::Vector2d y = pi_p(t);
    Eigen::Vector2d gamma_dot = pi.Derivative(s);
    Eigen::Vector2d gamma_dot_p = pi_p.Derivative(t);
    Eigen::Vector2d a = cos(phi) * gamma_dot * 2 / length_pi +
                        sin(phi) * gamma_dot_p * 2 / length_pi_p;
    double b = 1 + sin(2 * phi) * gamma_dot.dot(gamma_dot_p) *
                       4 / length_pi / length_pi_p;

    Eigen::Vector2d tangent = pi_p.Derivative(t);
    Eigen::Vector2d normal(tangent(1), -tangent(0));
    normal /= normal.norm();

    Eigen::Vector2d v;
    v << (nu.dgrad1(y) * a).dot(a),
         (nu.dgrad2(y) * a).dot(a);

    double res = 0.5 * r * normal.dot(v)
                 - 2 * a.dot(normal) * a.dot(nu.grad(y) * a) / b
                 + a.dot(normal) * normal.dot(nu.grad(y) * normal);
    return res / b;
  }
};

class Kernel4 : public AbstractKernel {
public:
  double operator()(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double s, double t) const {
    Eigen::Vector2d x = pi(s);
    Eigen::Vector2d y = pi_p(t);
    Eigen::Vector2d a = x - y;

    Eigen::Vector2d tangent = pi_p.Derivative(t);
    Eigen::Vector2d normal(tangent(1), -tangent(0));
    normal /= normal.norm();

    double res = - normal.dot(nu(y))
                 + 2 * a.dot(normal) * a.dot(nu(y)) / a.squaredNorm()
                 - a.dot(nu.grad(y).transpose() * normal)
                 + a.dot(normal) * normal.dot(nu.grad(y).transpose() * normal);
    return res / a.squaredNorm();
  }
};

class Kernel5 : public AbstractKernel {
public:
  double operator()(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double s, double t) const {
    Eigen::Vector2d x = pi(s);
    Eigen::Vector2d y = pi_p(t);
    Eigen::Vector2d a = x - y;

    Eigen::Vector2d tangent = pi.Derivative(s);
    Eigen::Vector2d normal(tangent(1), -tangent(0));
    normal /= normal.norm();

    double res = normal.dot(nu(y))
                 - 2 * a.dot(normal) * a.dot(nu(y)) / a.squaredNorm();
    return res / a.squaredNorm();
  }
};

class LogKernel : public AbstractSingularKernel {
public:
  double operator()(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double s, double t) const {
    Eigen::Vector2d x = pi(s);
    Eigen::Vector2d y = pi_p(t);
    return log((x - y).norm());
  }

  // Dummy implementation; treated in IntegralCoinciding
  double stable_st(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const AbstractVelocityField &nu,
      double s, double t) const {
    return 0;
  }

  // Dummy implementation; treated in IntegralAdjacent
  double stable_pr(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double length_pi, double length_pi_p,
      double r, double phi, double s, double t) const {
    return 0;
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

class Factor2 : public AbstractFactor {
public:
  double operator()(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractBEMSpace &space,
      const AbstractVelocityField &nu, unsigned q, double s) const {
    return space.evaluateShapeFunctionDot(q, s);
  }
};

class Factor3 : public AbstractFactor {
public:
  double operator()(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractBEMSpace &space,
      const AbstractVelocityField &nu, unsigned q, double s) const {
    Eigen::Vector2d x = pi(s);

    Eigen::Vector2d tangent = pi.Derivative(s);
    Eigen::Vector2d normal(tangent(1), -tangent(0));
    normal /= normal.norm();

    double res = nu.div(x) - normal.dot(nu.grad(x).transpose() * normal);
    return res * space.evaluateShapeFunctionDot(q, s);
  }
};

class Factor4 : public AbstractFactor {
public:
  double operator()(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractBEMSpace &space,
      const AbstractVelocityField &nu, unsigned q, double s) const {
    Eigen::Vector2d x = pi(s);
    Eigen::Vector2d gamma_dot = pi.Derivative(s);
    Eigen::Vector2d gamma_ddot = pi.DoubleDerivative(s);

    Eigen::Vector2d v;
    v << (nu.dgrad1(x) * gamma_dot).dot(gamma_dot),
         (nu.dgrad2(x) * gamma_dot).dot(gamma_dot);

    double res = - 2 * nu(x).dot(gamma_dot) * gamma_dot.dot(gamma_ddot)
                       / gamma_dot.squaredNorm()
                 + (nu.grad(x) * gamma_dot).dot(gamma_ddot)
                 + gamma_dot.dot(v + nu.grad(x) * gamma_ddot);
    return res * space.evaluateShapeFunction(q, s) / gamma_dot.squaredNorm();
  }
};

} // namespace transmission_bem

#endif // FACTORSHPP