#ifndef FACTORSHPP
#define FACTORSHPP

#include "velocity_fields.hpp"

#include <functional>
#include <math.h>

#include "abstract_bem_space.hpp"
#include "abstract_parametrized_curve.hpp"
#include <Eigen/Dense>

namespace transmission_bem {
class DirData {
public:
  virtual double operator()(Eigen::Vector2d x) const = 0;

  virtual Eigen::Vector2d grad(Eigen::Vector2d x) const = 0;

  virtual Eigen::Matrix2d hess(Eigen::Vector2d x) const = 0;
};

class G_CONST : public DirData {
public:
  G_CONST(std::function<double(Eigen::Vector2d)> g) : g_(g) {}

  double operator()(Eigen::Vector2d x) const {
    return g_(x);
  }

  Eigen::Vector2d grad(Eigen::Vector2d x) const {
    return Eigen::Vector2d::Zero();
  }

  Eigen::Matrix2d hess(Eigen::Vector2d x) const {
    return Eigen::Matrix2d::Zero();
  }

private:
  std::function<double(Eigen::Vector2d)> g_;
};

class NeuData {
public:
  virtual double operator()(Eigen::Vector2d x) const = 0;
};

class ETA_CONST : public NeuData {
public:
  ETA_CONST(std::function<double(Eigen::Vector2d)> eta) : eta_(eta) {}
  
  double operator()(Eigen::Vector2d x) const {
    return eta_(x);
  }

private:
  std::function<double(Eigen::Vector2d)> eta_;
};

double c = -1. / (2. * M_PI);

class AbstractKernel {
public:
  // Directe evaluation
  virtual double operator()(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double s, double t) const = 0;

  // Stable evaluation near singularity on coinciding panels
  virtual double stable_st(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const AbstractVelocityField &nu,
      double s0, double t0) const = 0;

  // Stable evaluation near singularity on adjoint panels
  // in polar coordinates including Jacobian 'r'
  virtual double stable_pr(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double scale, double scale_p,
      double r, double phi, double s, double t) const = 0;
};

class Kernel1 : public AbstractKernel {
public:
  double operator()(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double s, double t) const {
    Eigen::Vector2d x = pi(s);
    Eigen::Vector2d y = pi_p(t);
    return c * (x - y).dot(nu(x) - nu(y)) / (x - y).squaredNorm();
  }

  double stable_st(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const AbstractVelocityField &nu,
      double s, double t) const {
    double mid = 0.5 * (s + t);
    Eigen::Vector2d gamma_dot = pi.Derivative(mid);
    return c * gamma_dot.dot(nu.grad(pi(mid)) * gamma_dot) /
           gamma_dot.squaredNorm();
  }

  double stable_pr(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double scale, double scale_p,
      double r, double phi, double s0, double t0) const {
    Eigen::Vector2d gamma_dot = pi.Derivative(s0);
    Eigen::Vector2d gamma_dot_p = pi_p.Derivative(t0);
    Eigen::Vector2d a = cos(phi) * gamma_dot * scale -
                        sin(phi) * gamma_dot_p * scale_p;
    double b = 1 - sin(2 * phi) * gamma_dot.dot(gamma_dot_p) *
                       scale * scale_p;
    return c * r * a.dot(nu.grad(pi(s0)) * a) / b;
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
    Eigen::Vector2d a = x - y;

    Eigen::Vector2d tangent = pi_p.Derivative(t);
    Eigen::Vector2d normal(tangent(1), -tangent(0));
    normal /= normal.norm();

    double res = 2 * a.dot(normal) * a.dot(nu(x) - nu(y)) / a.squaredNorm()
                 - normal.dot(nu(x) - nu(y)) 
                 - a.dot(nu.div(y) * normal - nu.grad(y).transpose() * normal);
    return c * res / a.squaredNorm();
  }

  double stable_st(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const AbstractVelocityField &nu,
      double s, double t) const {
    Eigen::Vector2d y = pi(t);
    Eigen::Vector2d gamma_dot = pi.Derivative(t);
    Eigen::Vector2d gamma_ddot = pi.DoubleDerivative(t);

    Eigen::Vector2d tangent = gamma_dot;
    Eigen::Vector2d normal(tangent(1), -tangent(0));
    normal /= normal.norm();

    Eigen::Vector2d v;
    v << (nu.dgrad1(y) * gamma_dot).dot(gamma_dot),
         (nu.dgrad2(y) * gamma_dot).dot(gamma_dot);

    double res = gamma_ddot.dot(normal)
                     * gamma_dot.dot(nu.grad(y) * gamma_dot)
                     / gamma_dot.squaredNorm()
                 - 0.5 * normal.dot(v)
                 - 0.5 * gamma_ddot.dot(normal) * nu.div(y);
    return c * res / gamma_dot.squaredNorm();
  }

  double stable_pr(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double scale, double scale_p,
      double r, double phi, double s0, double t0) const {
    Eigen::Vector2d y = pi_p(t0);
    Eigen::Vector2d gamma_dot = pi.Derivative(s0);
    Eigen::Vector2d gamma_dot_p = pi_p.Derivative(t0);
    Eigen::Vector2d a = cos(phi) * gamma_dot * scale -
                        sin(phi) * gamma_dot_p * scale_p;
    double b = 1 - sin(2 * phi) * gamma_dot.dot(gamma_dot_p) *
                       scale * scale_p;

    Eigen::Vector2d tangent = gamma_dot_p;
    Eigen::Vector2d normal(tangent(1), -tangent(0));
    normal /= normal.norm();

    Eigen::Vector2d v;
    v << (nu.dgrad1(y) * a).dot(a),
         (nu.dgrad2(y) * a).dot(a);

    double res = 2 * a.dot(normal) * a.dot(nu.grad(y) * a) / b
                 - a.dot(normal) * nu.div(y);
    return c * res / b;
  }
};

class Kernel3 : public AbstractKernel {
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

    return -c * a.dot(normal) / a.squaredNorm();
  }

  double stable_st(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const AbstractVelocityField &nu,
      double s, double t) const {
    Eigen::Vector2d gamma_dot = pi.Derivative(t);
    Eigen::Vector2d gamma_ddot = pi.DoubleDerivative(t);

    Eigen::Vector2d tangent = gamma_dot;
    Eigen::Vector2d normal(tangent(1), -tangent(0));
    normal /= normal.norm();

    return -c * 0.5 * gamma_ddot.dot(normal) / gamma_dot.squaredNorm();
  }

  double stable_pr(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double scale, double scale_p,
      double r, double phi, double s0, double t0) const {
    Eigen::Vector2d gamma_dot = pi.Derivative(s0);
    Eigen::Vector2d gamma_dot_p = pi_p.Derivative(t0);
    Eigen::Vector2d a = cos(phi) * gamma_dot * scale -
                        sin(phi) * gamma_dot_p * scale_p;
    double b = 1 - sin(2 * phi) * gamma_dot.dot(gamma_dot_p) *
                       scale * scale_p;

    Eigen::Vector2d tangent = gamma_dot_p;
    Eigen::Vector2d normal(tangent(1), -tangent(0));
    normal /= normal.norm();

    return -c * a.dot(normal) / b;
  }
};

class LogKernel : public AbstractKernel {
public:
  double operator()(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double s, double t) const {
    Eigen::Vector2d x = pi(s);
    Eigen::Vector2d y = pi_p(t);
    return c * log((x - y).norm());
  }

  // Dummy implementation; treated in IntegralCoinciding
  double stable_st(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const AbstractVelocityField &nu,
      double s0, double t0) const {
    throw std::logic_error(
        "Stable evaluation of log kernel should not be called!");
    return 0;
  }

  // Dummy implementation; treated in IntegralAdjacent
  double stable_pr(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractParametrizedCurve &pi_p,
      const AbstractVelocityField &nu,
      double scale, double scale_p,
      double r, double phi, double s, double t) const {
    throw std::logic_error(
        "Stable evaluation of log kernel should not be called!");
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
  Factor3(const DirData &g) {
    grad_g_ = [&](Eigen::VectorXd x) { return g.grad(x); };
  }

  double operator()(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractBEMSpace &space,
      const AbstractVelocityField &nu, unsigned q, double s) const {
    Eigen::Vector2d x = pi(s);
    return grad_g_(x).dot(nu(x)) * pi.Derivative(s).norm();
  }

private:
  std::function<Eigen::Vector2d(Eigen::Vector2d)> grad_g_;
};

class Factor4 : public AbstractFactor {
public:
  Factor4(const DirData &g) {
    grad_g_ = [&](Eigen::VectorXd x) { return g.grad(x); };
    hess_g_ = [&](Eigen::VectorXd x) { return g.hess(x); };
  }

  double operator()(
      const parametricbem2d::AbstractParametrizedCurve &pi,
      const parametricbem2d::AbstractBEMSpace &space,
      const AbstractVelocityField &nu, unsigned q, double s) const {
    Eigen::Vector2d x = pi(s);
    Eigen::Vector2d grad =
        nu.grad(x).transpose() * grad_g_(x) + hess_g_(x).transpose() * nu(x);
    return  grad.dot(pi.Derivative(s));
  }

private:
  std::function<Eigen::Vector2d(Eigen::Vector2d)> grad_g_;
  std::function<Eigen::Matrix2d(Eigen::Vector2d)> hess_g_;
};
} // namespace transmission_bem

#endif // FACTORSHPP