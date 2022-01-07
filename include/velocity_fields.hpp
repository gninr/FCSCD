#ifndef VELOCITYFIELDSHPP
#define VELOCITYFIELDSHPP

#include <Eigen/Dense>

class AbstractVelocityField {
public:
  virtual Eigen::Vector2d operator()(const Eigen::Vector2d &x) const = 0;

  virtual Eigen::Matrix2d grad(const Eigen::Vector2d &x) const = 0;

  virtual double div(const Eigen::Vector2d &x) const = 0;

  virtual Eigen::Matrix2d dgrad1(const Eigen::Vector2d &x) const = 0;

  virtual Eigen::Matrix2d dgrad2(const Eigen::Vector2d &x) const = 0;
};

class NuConstant : public AbstractVelocityField {
public:
  NuConstant(const Eigen::Vector2d &direction) {
    direction_ = direction.normalized();
  }

  Eigen::Vector2d operator()(const Eigen::Vector2d &x) const {
    return direction_;
  }

  Eigen::Matrix2d grad(const Eigen::Vector2d &x) const {
    return Eigen::Matrix2d::Zero();
  }

  double div(const Eigen::Vector2d &x) const {
    return 0;
  }

  Eigen::Matrix2d dgrad1(const Eigen::Vector2d &x) const {
    return Eigen::Matrix2d::Zero();
  }

  Eigen::Matrix2d dgrad2(const Eigen::Vector2d &x) const {
    return Eigen::Matrix2d::Zero();
  }

private:
  Eigen::Vector2d direction_;
};

class NuRotation : public AbstractVelocityField {
public:
  NuRotation(const Eigen::Vector2d &center) : center_(center) {};

  Eigen::Vector2d operator()(const Eigen::Vector2d &x) const {
    Eigen::Vector2d xnew = x - center_;
    return Eigen::Vector2d(-xnew[1], xnew[0]);
  }

  Eigen::Matrix2d grad(const Eigen::Vector2d &x) const {
    Eigen::Matrix2d M;
    M << 0, -1, 1, 0;
    return M;
  }

  double div(const Eigen::Vector2d &x) const {
    return 0;
  }

  Eigen::Matrix2d dgrad1(const Eigen::Vector2d &x) const {
    Eigen::Matrix2d M;
    M << 0, 0, 0, 0;
    return M;
  }

  Eigen::Matrix2d dgrad2(const Eigen::Vector2d &x) const {
    Eigen::Matrix2d M;
    M << 0, 0, 0, 0;
    return M;
  }
  
private:
  Eigen::Vector2d center_;
};

class NuRadial : public AbstractVelocityField {
public:
  Eigen::Vector2d operator()(const Eigen::Vector2d &X) const {
    double R = 2;
    double x = X(0);
    double y = X(1);
    Eigen::Vector2d out;
    if (x * x + y * y < 1.1 * 1.1 * R * R) // Inner circle
      out << x + x * x + y * y - R * R, y + x * x + y * y - R * R;
    else // Outer circle
      out << x + x * x + y * y - 4 * R * 4 * R,
          y + x * x + y * y - 4 * R * 4 * R;
    return out;
  }

  Eigen::Matrix2d grad(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Matrix2d M;
    M << 1 + 2 * x, 2 * x, 2 * y, 1 + 2 * y;
    return M;
  }

  double div(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    return 2 * (1 + x + y);
  }
  // Second order derivative of the velocity field's first component
  Eigen::Matrix2d dgrad1(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Matrix2d M;
    M << 2, 0, 0, 2;
    return M;
  }
  // Second order derivative of the velocity field's second component
  Eigen::Matrix2d dgrad2(const Eigen::Vector2d &X) const {
    double x = X(0);
    double y = X(1);
    Eigen::Matrix2d M;
    M << 2, 0, 0, 2;
    return M;
  }
};

#endif // VELOCITYFIELDS