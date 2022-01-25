#ifndef TRANSMISSIONFEMHPP
#define TRANSMISSIONFEMHPP

#include <lf/assemble/assemble.h>
#include <lf/fe/fe.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/mesh/utils/utils.h>
#include <lf/refinement/refinement.h>
#include <lf/uscalfe/uscalfe.h>

namespace transmission_fem {

class MeshFunctionPWConstant {
  using POSPRED = std::function<bool(const lf::mesh::Entity&)>;
public:
  MeshFunctionPWConstant(double epsilon1, double epsilon2, POSPRED inner_sel) :
      epsilon1_(epsilon1), epsilon2_(epsilon2), inner_sel_(inner_sel) {}

  std::vector<double> operator()(const lf::mesh::Entity& e,
                                 const Eigen::MatrixXd& local) const {
    std::vector<double> res;
    res.reserve(local.cols());
    double eps = inner_sel_(e) ? epsilon1_ : epsilon2_;
    for (unsigned i = 0; i < local.cols(); ++i) {
      res.push_back(eps);
    }
    return res;
  }
private:
  double epsilon1_, epsilon2_;
  POSPRED inner_sel_;
};

Eigen::VectorXd Solve(
    std::shared_ptr<lf::uscalfe::UniformScalarFESpace<double>> fe_space,
    std::function<bool(Eigen::Vector2d)> dir_sel,
    std::function<bool(const lf::mesh::Entity&)> inner_sel,
    std::function<double(Eigen::Vector2d)> g,
    std::function<double(Eigen::Vector2d)> eta,
    double epsilon1, double epsilon2) {

  using size_type = lf::base::size_type;
  using glb_idx_t = lf::assemble::glb_idx_t;

  // Boundary conditions
  lf::mesh::utils::MeshFunctionGlobal mf_g{g};
  lf::mesh::utils::MeshFunctionGlobal mf_eta{eta};

  // Coefficient
  MeshFunctionPWConstant mf_epsilon{epsilon1, epsilon2, inner_sel};
  lf::mesh::utils::MeshFunctionConstant<double> mf_zero{0.};
  
  // Selectors for boundary conditions
  lf::refinement::EntityCenterPositionSelector edge_sel_dir{dir_sel};

  // Set up FE space
  const lf::mesh::Mesh& mesh{*(fe_space->Mesh())};
  const lf::assemble::DofHandler& dofh{fe_space->LocGlobMap()};
  auto bd_flags{lf::mesh::utils::flagEntitiesOnBoundary(fe_space->Mesh(), 1)};
  const size_type N_dofs(dofh.NumDofs());

  // Assemble Galerkin matrix
  lf::assemble::COOMatrix<double> mat(N_dofs, N_dofs);
  lf::uscalfe::ReactionDiffusionElementMatrixProvider<
      double, decltype(mf_epsilon), decltype(mf_zero)>
      elmat_builder(fe_space, mf_epsilon, mf_zero);
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, elmat_builder, mat);

  // Assemble rhs vector
  Eigen::VectorXd rhs(N_dofs);
  rhs.setZero();
  auto edge_sel_neu = [&](const lf::mesh::Entity& edge) {
    return (bd_flags(edge) && !edge_sel_dir(edge));
  };
  lf::uscalfe::ScalarLoadEdgeVectorProvider<double, decltype(mf_eta),
                                            decltype(edge_sel_neu)>
      elvec_builder_neu(fe_space, mf_eta, edge_sel_neu);
  AssembleVectorLocally(1, dofh, elvec_builder_neu, rhs);

  // Fix components according to Dirichlet boundary conditions
  auto ess_bdc_flags_values{lf::fe::InitEssentialConditionFromFunction(
      *fe_space,
      [&](const lf::mesh::Entity& edge) {
        return (bd_flags(edge) && edge_sel_dir(edge));
      },
      mf_g)};
  lf::assemble::FixFlaggedSolutionComponents<double>(
      [&](glb_idx_t gdof_idx) {
        return ess_bdc_flags_values[gdof_idx];
      },
      mat, rhs);

  // Solve LSE
  Eigen::SparseMatrix<double> mat_crs = mat.makeSparse();
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(mat_crs);
  Eigen::VectorXd sol = solver.solve(rhs);

  return sol;
}
} // namespace transmission_fem

#endif // TRANSMISSIONFEMHPP