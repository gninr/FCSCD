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
Eigen::VectorXd solve(const parametricbem2d::ParametrizedMesh &mesh,
                      unsigned nd, // number of panels on Dirichlet boundary
                      std::function<double(double, double)> g,
                      std::function<double(double, double)> eta,
                      double epsilon1, double epsilon2, unsigned order) {
  
  parametricbem2d::ContinuousSpace<1> trial_space; // in H^{1/2}
  parametricbem2d::DiscontinuousSpace<0> test_space; // in H^{/2}
  parametricbem2d::ContinuousSpace<1> g_interp_space;
  parametricbem2d::DiscontinuousSpace<0> eta_interp_space;
  
  Eigen::MatrixXd M = parametricbem2d::MassMatrix(
                          mesh, test_space, trial_space, order);
  Eigen::MatrixXd V = parametricbem2d::single_layer::GalerkinMatrix(
                          mesh, test_space, order);
  Eigen::MatrixXd K = parametricbem2d::double_layer::GalerkinMatrix(
                          mesh, trial_space, test_space, order);
  Eigen::MatrixXd W = parametricbem2d::hypersingular::GalerkinMatrix(
                          mesh, trial_space, order);
  Eigen::VectorXd g_interp = g_interp_space.Interpolate(g, mesh);
  Eigen::VectorXd eta_interp = eta_interp_space.Interpolate(eta, mesh);

  // Number of panels on each boundary
  unsigned n, ni, nn;
  n = mesh.getNumPanels();
  ni = mesh.getSplit();
  nn = n - ni - nd;
  // Dimension of spaces
  // TODO

  // Assemble matrix blocks
  auto block_ii = [&](Eigen::MatrixXd A) {
    Eigen::MatrixXd A_ii(ni, ni);
    A_ii << A.block(0, 0, ni, ni);
    return A_ii;
  };
  auto block_id = [&](Eigen::MatrixXd A) {
    Eigen::MatrixXd A_id(ni, nd);
    A_id << A.block(0, ni, ni, nd/2),
            A.block(0, ni+nd/2+nn/2, ni, nd/2);
    return A_id;
  };
  auto block_in = [&](Eigen::MatrixXd A) {
    Eigen::MatrixXd A_in(ni, nn);
    A_in << A.block(0, ni+nd/2, ni, nn/2),
            A.block(0, ni+nd+nn/2, ni, nn/2);
    return A_in;
  };
  auto block_di = [&](Eigen::MatrixXd A) {
    Eigen::MatrixXd A_di(nd, ni);
    A_di << A.block(ni, 0, nd/2, ni),
            A.block(ni+nd/2+nn/2, 0, nd/2, ni);
    return A_di;
  };
  auto block_dd = [&](Eigen::MatrixXd A) {
    Eigen::MatrixXd A_dd(nd, nd);
    A_dd << A.block(ni, ni, nd/2, nd/2),
            A.block(ni, ni+nd/2+nn/2, nd/2, nd/2),
            A.block(ni+nd/2+nn/2, ni, nd/2, nd/2),
            A.block(ni+nd/2+nn/2, ni+nd/2+nn/2, nd/2, nd/2);
    return A_dd;
  };
  auto block_dn = [&](Eigen::MatrixXd A) {
    Eigen::MatrixXd A_dn(nd, nn);
    A_dn << A.block(ni, ni+nd/2, nd/2, nn/2),
            A.block(ni, ni+nd+nn/2, nd/2, nn/2),
            A.block(ni+nd/2+nn/2, ni+nd/2, nd/2, nn/2),
            A.block(ni+nd/2+nn/2, ni+nd+nn/2, nd/2, nn/2);
    return A_dn;
  };
  auto block_ni = [&](Eigen::MatrixXd A) {
    Eigen::MatrixXd A_ni(nn, ni);
    A_ni << A.block(ni+nd/2, 0, nn/2, ni),
            A.block(ni+nd+nn/2, 0, nn/2, ni);
    return A_ni;
  };
  auto block_nd = [&](Eigen::MatrixXd A) {
    Eigen::MatrixXd A_nd(nn, nd);
    A_nd << A.block(ni+nd/2, ni, nn/2, nd/2),
            A.block(ni+nd/2, ni+nd/2+nn/2, nn/2, nd/2),
            A.block(ni+nd+nn/2, ni, nn/2, nd/2),
            A.block(ni+nd+nn/2, ni+nd/2+nn/2, nn/2, nd/2);
    return A_nd;
  };
  auto block_nn = [&](Eigen::MatrixXd A) {
    Eigen::MatrixXd A_nn(nn, nn);
    A_nn << A.block(ni+nd/2, ni+nd/2, nn/2, nn/2),
            A.block(ni+nd/2, ni+nd+nn/2, nn/2, nn/2),
            A.block(ni+nd+nn/2, ni+nd/2, nn/2, nn/2),
            A.block(ni+nd+nn/2, ni+nd+nn/2, nn/2, nn/2);
    return A_nn;
  };

  // Assemble boundary data
  Eigen::VectorXd g_N(nd);
  g_N << g_interp.segment(ni, nd/2),
         g_interp.segment(ni+nd/2+nn/2, nd/2);

  Eigen::VectorXd eta_N(nn);
  eta_N << eta_interp.segment(ni+nd/2, nn/2),
           eta_interp.segment(ni+nd+nn/2, nn/2);

  // Assemble Galerkin matrix
  Eigen::MatrixXd mat(ni*2+nd+nn, ni*2+nd+nn);
  mat << 2. * block_ii(K),
         -(1. + epsilon2/epsilon1) * block_ii(V),
         0.5 * block_in(M) + block_in(K),
         -block_id(V),
         0.5 * block_di(M) + block_di(K),
         -block_di(V),
         0.5 * block_dn(M) + block_dn(K),
         -block_dd(V),
         (epsilon1/epsilon2 + 1.) * block_ii(W),
         2. * block_ii(K).transpose(),
         block_in(W),
         (-0.5 * block_di(M) + block_di(K)).transpose(),
         -block_ni(W),
         (0.5 * block_in(M) - block_in(K)).transpose(),
         -block_nn(W),
         (0.5 * block_dn(M) - block_dn(K)).transpose();

  // Assemble rhs vector
  Eigen::VectorXd rhs(ni*2+nd+nn);
  rhs << -(0.5 * block_id(M) + block_id(K)) * g_N + block_in(V) * eta_N,
         -(0.5 * block_dd(M) + block_dd(K)) * g_N + block_dn(V) * eta_N,
         -block_id(W) * g_N + (0.5 * block_ni(M) - block_ni(K)).transpose() * eta_N,
         block_nd(W) * g_N - (0.5 * block_nn(M) - block_nn(K)).transpose() * eta_N;

  Eigen::VectorXd sol = mat.lu().solve(rhs);
  return sol;
}
} // namespace transmission_bem

#endif // TRANSMISSIONBEMHPP