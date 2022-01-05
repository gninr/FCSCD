#ifndef TRANSMISSIONBEMHPP
#define TRANSMISSIONBEMHPP

#include "abstract_bem_space.hpp"
#include "continuous_space.hpp"
#include "dirichlet.hpp"
#include "discontinuous_space.hpp"
#include "double_layer.hpp"
#include "hypersingular.hpp"
#include "parametrized_mesh.hpp"
#include "single_layer.hpp"
#include <Eigen/Dense>

namespace transmission_bem {

struct Solution {
  Eigen::VectorXd u_i, psi_i, u, psi;
};

Eigen::VectorXd slice(Eigen::VectorXd v, Eigen::ArrayXi ind) {
  unsigned n = ind.size();
  Eigen::VectorXd res(n);
  for (unsigned i = 0; i < n; ++i) {
    res[i] = v[ind[i]];
  }
  return res;
}

Eigen::MatrixXd slice(Eigen::MatrixXd A,
                      Eigen::ArrayXi ind_row,
                      Eigen::ArrayXi ind_col) {
  unsigned num_rows = ind_row.size();
  unsigned num_cols = ind_col.size();
  Eigen::MatrixXd res(num_rows, num_cols);
  for (unsigned i = 0; i < num_rows; ++i) {
    res.row(i) = slice(A.row(ind_row[i]), ind_col);
  }
  return res;
}

Solution solve(const parametricbem2d::ParametrizedMesh &mesh,
                      // Dirichlet trace space
                      parametricbem2d::AbstractBEMSpace &space_d,
                      // Neumann trace space
                      parametricbem2d::AbstractBEMSpace &space_n,
                      std::function<double(Eigen::Vector2d)> dir_sel,
                      std::function<double(Eigen::Vector2d)> g,
                      std::function<double(Eigen::Vector2d)> eta,
                      double epsilon1, double epsilon2, unsigned order) {      
  // Get panels
  parametricbem2d::PanelVector panels = mesh.getPanels();
  // Count number of panels
  unsigned numpanels = mesh.getNumPanels();
  unsigned numpanels_i = mesh.getSplit();

  // Preprocessing of Dirichlet trace space

  // Number of reference shape functions
  unsigned q_d = space_d.getQ();
  // Space dimension
  unsigned dim_d = space_d.getSpaceDim(numpanels);
  // Mark global shape functions associated with different type of panels
  Eigen::ArrayXi space_d_sel_i(Eigen::VectorXi::Zero(dim_d));
  Eigen::ArrayXi space_d_sel_d(Eigen::VectorXi::Zero(dim_d));
  for (unsigned i = 0; i < numpanels; ++i) {
    // Panels at interface
    if (i < numpanels_i) {
      for (unsigned k = 0; k < q_d; ++k) {
        unsigned II = space_d.LocGlobMap2(k + 1, i + 1, mesh) - 1;
        space_d_sel_i[II] = 1;
      }
    }
    // Panels on boundary
    else {
      parametricbem2d::AbstractParametrizedCurve &gamma_pi = *panels[i];
      // Panels with Dirichlet boundary condition
      if (dir_sel(gamma_pi(-1)) && dir_sel(gamma_pi(1))) {
        for (unsigned k = 0; k < q_d; ++k) {
          unsigned II = space_d.LocGlobMap2(k + 1, i + 1, mesh) - 1;
          space_d_sel_d[II] = 1;
        }
      }
    }
  }
  // Number of global shape functions associated with panels at interface
  unsigned dim_d_i = space_d_sel_i.sum();
  // Number of global shape functions associated with panels with different
  // boundary conditions
  unsigned dim_d_d = space_d_sel_d.sum();
  unsigned dim_d_n = dim_d - dim_d_i - dim_d_d;
  // Indices of global shape functions associated with different type of panels
  Eigen::ArrayXi ind_d_i(dim_d_i), ind_d_d(dim_d_d), ind_d_n(dim_d_n);
  {
    unsigned curr_i = 0, curr_n = 0, curr_d = 0;
    for (unsigned i = 0; i < dim_d; ++i) {
      if (space_d_sel_i[i]) {
        ind_d_i[curr_i++] = i;
      }
      else if (space_d_sel_d[i]) {
        ind_d_d[curr_d++] = i;
      }
      else {
        ind_d_n[curr_n++] = i;
      }
    }
  }

  // Preprocessing of Neumann trace space

  // Number of reference shape functions
  unsigned q_n = space_n.getQ();
  // Space dimension
  unsigned dim_n = space_n.getSpaceDim(numpanels);
  // Mark global shape functions associated with different type of panels
  Eigen::ArrayXi space_n_sel_i = Eigen::VectorXi::Zero(dim_n);
  Eigen::ArrayXi space_n_sel_n = Eigen::VectorXi::Zero(dim_n);
  for (unsigned i = 0; i < numpanels; ++i) {
    // Panels at interface
    if (i < numpanels_i) {
      for (unsigned k = 0; k < q_n; ++k) {
        unsigned II = space_n.LocGlobMap2(k + 1, i + 1, mesh) - 1;
        space_n_sel_i[II] = 1;
      }
    }
    // Panels on boundary
    else {
      parametricbem2d::AbstractParametrizedCurve &gamma_pi = *panels[i];
      // Panels with Neumann boundary condition
      if (!dir_sel(gamma_pi(-1)) || !dir_sel(gamma_pi(1))) {
        for (unsigned k = 0; k < q_n; ++k) {
          unsigned II = space_n.LocGlobMap2(k + 1, i + 1, mesh) - 1;
          space_n_sel_n[II] = 1;
        }
      }
    }
  }
  // Number of global shape functions associated with panels at interface
  unsigned dim_n_i = space_n_sel_i.sum();
  // Number of global shape functions associated with panels with different
  // boundary conditions
  unsigned dim_n_n = space_n_sel_n.sum();
  unsigned dim_n_d = dim_n - dim_n_i - dim_n_n;
  // Indices of global shape functions associated with different type of panels
  Eigen::ArrayXi ind_n_i(dim_n_i), ind_n_n(dim_n_n), ind_n_d(dim_n_d);
  {
    unsigned curr_i = 0, curr_n = 0, curr_d = 0;
    for (unsigned i = 0; i < dim_n; ++i) {
      if (space_n_sel_i[i]) {
        ind_n_i[curr_i++] = i;
      }
      else if (space_n_sel_n[i]) {
        ind_n_n[curr_n++] = i;
      }
      else {
        ind_n_d[curr_d++] = i;
      }
    }
  }

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
  Eigen::MatrixXd lhs(dim_n_i + dim_n_d + dim_d_i + dim_d_n,
                      dim_d_i + dim_n_i + dim_d_n + dim_n_d);
  lhs << slice(K, ind_n_i, ind_d_i),
         -(epsilon2 / epsilon1 + 1) * slice(V, ind_n_i, ind_n_i),
         slice(K, ind_n_i, ind_d_n),
         -slice(V, ind_n_i, ind_n_d),
         slice(K, ind_n_d, ind_d_i),
         -slice(V, ind_n_d, ind_n_i),
         slice(K, ind_n_d, ind_d_n),
         -slice(V, ind_n_d, ind_n_d),
         (epsilon1 / epsilon2 + 1) * slice(W, ind_d_i, ind_d_i),
         2 * slice(K, ind_n_i, ind_d_i).transpose(),
         slice(W, ind_d_i, ind_d_n),
         slice(K, ind_n_d, ind_d_i).transpose(),
         slice(W, ind_d_n, ind_d_i),
         slice(K, ind_n_i, ind_d_n).transpose(),
         slice(W, ind_d_n, ind_d_n),
         slice(K, ind_n_d, ind_d_n).transpose();

  // Get interpolants of boundary data
  Eigen::VectorXd g_interp = space_d.Interpolate(
      [&](double x, double y) { return g(Eigen::Vector2d(x, y)); },
      mesh);
  Eigen::VectorXd eta_interp = space_n.Interpolate(
      [&](double x, double y) { return eta(Eigen::Vector2d(x, y)); },
      mesh);

  // Assemble RHS
  Eigen::MatrixXd rhs_mat(dim_n_i + dim_n_d + dim_d_i + dim_d_n,
                          dim_d_d + dim_n_n);
  rhs_mat << -slice(K, ind_n_i, ind_d_d),
             slice(V, ind_n_i, ind_n_n),
             -0.5 * slice(M, ind_n_d, ind_d_d) - slice(K, ind_n_d, ind_d_d),
             slice(V, ind_n_d, ind_n_n),
             -slice(W, ind_d_i, ind_d_d),
             -slice(K, ind_n_n, ind_d_i).transpose(),
             -slice(W, ind_d_n, ind_d_d),
             (0.5 * slice(M, ind_n_n, ind_d_n)
                - slice(K, ind_n_n, ind_d_n)).transpose();

  Eigen::VectorXd rhs_vec(dim_d_d + dim_n_n);
  rhs_vec << slice(g_interp, ind_d_d), slice(eta_interp, ind_n_n);

  // Solve LSE
  Eigen::VectorXd sol_vec = lhs.lu().solve(rhs_mat * rhs_vec);

  // Construct solution
  Solution sol;
  sol.u_i = sol_vec.segment(0, dim_d_i);
  sol.psi_i = sol_vec.segment(dim_d_i, dim_n_i);
  sol.u = sol_vec.segment(dim_d_i + dim_n_i, dim_d_n);
  sol.psi = sol_vec.segment(dim_d_i + dim_n_i + dim_d_n, dim_n_d);
  return sol;
}
} // namespace transmission_bem

#endif // TRANSMISSIONBEMHPP