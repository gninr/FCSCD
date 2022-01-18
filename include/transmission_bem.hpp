#ifndef TRANSMISSIONBEMHPP
#define TRANSMISSIONBEMHPP

#include <functional>

#include "abstract_bem_space.hpp"
#include "dirichlet.hpp"
#include "double_layer.hpp"
#include "hypersingular.hpp"
#include "parametrized_mesh.hpp"
#include "single_layer.hpp"
#include <Eigen/Dense>

namespace transmission_bem {
Eigen::VectorXd Slice(const Eigen::VectorXd &v, const Eigen::ArrayXi &ind) {
  unsigned n = ind.size();
  Eigen::VectorXd res(n);
  for (unsigned i = 0; i < n; ++i) {
    res[i] = v[ind[i]];
  }
  return res;
}

Eigen::MatrixXd Slice(const Eigen::MatrixXd &A,
                      const Eigen::ArrayXi &ind_row,
                      const Eigen::ArrayXi &ind_col) {
  unsigned num_rows = ind_row.size();
  unsigned num_cols = ind_col.size();
  Eigen::MatrixXd res(num_rows, num_cols);
  for (unsigned i = 0; i < num_rows; ++i) {
    res.row(i) = Slice(A.row(ind_row[i]), ind_col);
  }
  return res;
}

// Number of global shape functions associated with different type of panels
struct Dims {
  unsigned all;
  unsigned i; // at interface
  unsigned d; // on boundary with Dirichlet condition
  unsigned n; // on boundary with Neumann condition
};

// Indices of global shape functions associated with different type of panels
struct Indices {
  Eigen::ArrayXi i; // at interface
  Eigen::ArrayXi d; // on boundary with Dirichlet condition
  Eigen::ArrayXi n; // on boundary with Neumann condition
};

// Preprocessing of Dirichlet trace space
void ComputeDirSpaceInfo(const parametricbem2d::ParametrizedMesh &mesh,
                         const parametricbem2d::AbstractBEMSpace &space,
                         std::function<bool(Eigen::Vector2d)> dir_sel,
                         Dims &dims, Indices &ind) {
  // Get panels
  parametricbem2d::PanelVector panels = mesh.getPanels();
  // Count number of panels
  unsigned numpanels = mesh.getNumPanels();
  unsigned numpanels_i = mesh.getSplit();
  // Number of reference shape functions
  unsigned q = space.getQ();
  // Space dimension
  unsigned dim = space.getSpaceDim(numpanels);
  dims.all = dim;

  // Mark global shape functions associated with different type of panels
  Eigen::ArrayXi space_sel_i(Eigen::VectorXi::Zero(dim));
  Eigen::ArrayXi space_sel_d(Eigen::VectorXi::Zero(dim));
  // Panels at interface
  for (unsigned i = 0; i < numpanels_i; ++i) {
    for (unsigned k = 0; k < q; ++k) {
      unsigned II = space.LocGlobMap2(k + 1, i + 1, mesh) - 1;
      space_sel_i[II] = 1;
    }
  }
  // Panels on boundary
  for (unsigned i = numpanels_i; i < numpanels; ++i) {
    parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
    // Panels with Dirichlet boundary condition
    if (dir_sel(pi(-1)) && dir_sel(pi(1))) {
      for (unsigned k = 0; k < q; ++k) {
        unsigned II = space.LocGlobMap2(k + 1, i + 1, mesh) - 1;
        space_sel_d[II] = 1;
      }
    }
  }

  // Compute number of global shape functions associated with
  // different type of panels
  dims.i = space_sel_i.sum();
  dims.d = space_sel_d.sum();
  dims.n = dim - dims.i - dims.d;
  // Compute indices of global shape functions associated with
  // different type of panels
  ind.i = Eigen::ArrayXi(dims.i);
  ind.d = Eigen::ArrayXi(dims.d);
  ind.n = Eigen::ArrayXi(dims.n);
  unsigned curr_i = 0, curr_n = 0, curr_d = 0;
  for (unsigned i = 0; i < dim; ++i) {
    if (space_sel_i[i]) {
      ind.i[curr_i++] = i;
    }
    else if (space_sel_d[i]) {
      ind.d[curr_d++] = i;
    }
    else {
      ind.n[curr_n++] = i;
    }
  }

  return;
}

// Preprocessing of Neumann trace space
void ComputeNeuSpaceInfo(const parametricbem2d::ParametrizedMesh &mesh,
                         const parametricbem2d::AbstractBEMSpace &space,
                         std::function<bool(Eigen::Vector2d)> dir_sel,
                         Dims &dims, Indices &ind) {
  // Get panels
  parametricbem2d::PanelVector panels = mesh.getPanels();
  // Count number of panels
  unsigned numpanels = mesh.getNumPanels();
  unsigned numpanels_i = mesh.getSplit();
  // Number of reference shape functions
  unsigned q = space.getQ();
  // Space dimension
  unsigned dim = space.getSpaceDim(numpanels);
  dims.all = dim;

  // Mark global shape functions associated with different type of panels
  Eigen::ArrayXi space_sel_i = Eigen::VectorXi::Zero(dim);
  Eigen::ArrayXi space_sel_n = Eigen::VectorXi::Zero(dim);
  // Panels at interface
  for (unsigned i = 0; i < numpanels_i; ++i) {
    for (unsigned k = 0; k < q; ++k) {
      unsigned II = space.LocGlobMap2(k + 1, i + 1, mesh) - 1;
      space_sel_i[II] = 1;
    }
  }
  // Panels on boundary
  for (unsigned i = numpanels_i; i < numpanels; ++i) {
    parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
    // Panels with Neumann boundary condition
    if (!dir_sel(pi(-1)) || !dir_sel(pi(1))) {
      for (unsigned k = 0; k < q; ++k) {
        unsigned II = space.LocGlobMap2(k + 1, i + 1, mesh) - 1;
        space_sel_n[II] = 1;
      }
    }
  }

  // Compute number of global shape functions associated with
  // different type of panels
  dims.i = space_sel_i.sum();
  dims.n = space_sel_n.sum();
  dims.d = dim - dims.i - dims.n;
  // Compute indices of global shape functions associated with
  // different type of panels
  ind.i = Eigen::ArrayXi(dims.i);
  ind.n = Eigen::ArrayXi(dims.n);
  ind.d = Eigen::ArrayXi(dims.d);
  unsigned curr_i = 0, curr_n = 0, curr_d = 0;
  for (unsigned i = 0; i < dim; ++i) {
    if (space_sel_i[i]) {
      ind.i[curr_i++] = i;
    }
    else if (space_sel_n[i]) {
      ind.n[curr_n++] = i;
    }
    else {
      ind.d[curr_d++] = i;
    }
  }

  return;
}

// Interpolate Dirichlet data in Dirichlet trace space
Eigen::VectorXd InterpolateDirData(
    const parametricbem2d::ParametrizedMesh &mesh,
    std::function<double(Eigen::Vector2d)> g,
    const parametricbem2d::AbstractBEMSpace &space,
    Indices ind) {
  Eigen::VectorXd g_interp = space.Interpolate(
      [&](double x, double y) { return g(Eigen::Vector2d(x, y)); },
      mesh);
  return Slice(g_interp, ind.d);
}

// Interpolate Neumann data in Neumann trace space
Eigen::VectorXd InterpolateNeuData(
    const parametricbem2d::ParametrizedMesh &mesh,
    std::function<double(Eigen::Vector2d)> eta,
    const parametricbem2d::AbstractBEMSpace &space,
    Indices ind) {
  Eigen::VectorXd eta_interp = space.Interpolate(
      [&](double x, double y) { return eta(Eigen::Vector2d(x, y)); },
      mesh);
  return Slice(eta_interp, ind.n);
}

Eigen::VectorXd Solve(
    const parametricbem2d::ParametrizedMesh &mesh,
    // Dirichlet trace space
    const parametricbem2d::AbstractBEMSpace &space_d,
    // Neumann trace space
    const parametricbem2d::AbstractBEMSpace &space_n,
    std::function<bool(Eigen::Vector2d)> dir_sel,
    std::function<double(Eigen::Vector2d)> g,
    std::function<double(Eigen::Vector2d)> eta,
    double epsilon1, double epsilon2, unsigned order) {      
  // Compute Space Information
  Dims dims_d, dims_n;
  Indices ind_d, ind_n;
  ComputeDirSpaceInfo(mesh, space_d, dir_sel, dims_d, ind_d);
  ComputeNeuSpaceInfo(mesh, space_n, dir_sel, dims_n, ind_n);

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
  Eigen::MatrixXd rhs_mat(dims_d.i + dims_n.i + dims_d.n + dims_n.d,
                          dims_d.d + dims_n.n);
  rhs_mat << -Slice(W, ind_d.i, ind_d.d),
             -Slice(K, ind_n.n, ind_d.i).transpose(),
             
             -Slice(K, ind_n.i, ind_d.d),
             Slice(V, ind_n.i, ind_n.n),

             -Slice(W, ind_d.n, ind_d.d),
             (0.5 * Slice(M, ind_n.n, ind_d.n)
                 - Slice(K, ind_n.n, ind_d.n)).transpose(),

             -0.5 * Slice(M, ind_n.d, ind_d.d) - Slice(K, ind_n.d, ind_d.d),
             Slice(V, ind_n.d, ind_n.n);

  Eigen::VectorXd rhs_vec(dims_d.d + dims_n.n);
  rhs_vec << InterpolateDirData(mesh, g, space_d, ind_d),
             InterpolateNeuData(mesh, eta, space_n, ind_n);

  // Solve LSE
  Eigen::HouseholderQR<Eigen::MatrixXd> dec(lhs);
  Eigen::VectorXd sol = dec.solve(rhs_mat * rhs_vec);

  std::cout << "\nu_i" << std::endl;
  std::cout << sol.segment(0, dims_d.i) << std::endl;
  std::cout << "\npsi_i" << std::endl;
  std::cout << sol.segment(dims_d.i, dims_n.i) << std::endl;
  std::cout << "\nu" << std::endl;
  std::cout << sol.segment(dims_d.i + dims_n.i, dims_d.n) << std::endl;
  std::cout << "\npsi" << std::endl;
  std::cout << sol.tail(dims_n.d) << std::endl;

  return sol;
}
} // namespace transmission_bem

#endif // TRANSMISSIONBEMHPP