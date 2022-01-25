#include "transmission_bem.hpp"

#include <cmath>
#include <iostream>

#include "continuous_space.hpp"
#include "discontinuous_space.hpp"
#include "abstract_parametrized_curve.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include "parametrized_mesh.hpp"
#include <Eigen/Dense>

Eigen::Vector2d exkite(double t) {
  return Eigen::Vector2d(0.3+.35 * std::cos(t) + .1625 * std::cos(2 * t),
                         0.5+.35 * std::sin(t));
}

Eigen::Vector2d exdkite(double t) {
  return Eigen::Vector2d(-.35 * std::sin(t) - 2 * .1625 * std::sin(2 * t),
                         .35 * std::cos(t));
}

Eigen::VectorXd get_kite_params(unsigned N) {
  /*
  // Calculating the length of the kite
  unsigned N_length = 500; // No. of points used in the calculation
  Eigen::VectorXd pts_length = Eigen::VectorXd::LinSpaced(N_length,0,2*M_PI);
  double L = 0;
  for (unsigned i = 0 ; i < N_length-1 ; ++i)
    L += (exkite(pts_length(i)) - exkite(pts_length(i+1))).norm();

  std::cout << "found length of the kite: " << L << std::endl;
  */
  double L = 2.46756;
  // Solving the equation for Phi using explicit timestepping
  unsigned k = 20; // multiplicity factor
  double h = L/N/k; // step size
  Eigen::VectorXd phi_full = Eigen::VectorXd::Constant(N*k,0);
  Eigen::VectorXd phi = Eigen::VectorXd::Constant(N,0);
  for (unsigned i = 1 ; i < N*k ; ++i)
    phi_full(i) = phi_full(i-1) + h /(exdkite(phi_full(i-1))).norm();

  for (unsigned i = 0 ; i < N ; ++i)
    phi(i) = phi_full(i*k);

  return phi;
}

int main(int argc, char *argv[]) {
  std::ofstream out("transmission_bem.txt");
  out.precision(std::numeric_limits<double>::digits10);

  // Gauss quadrature order
  unsigned order = 8;

  double epsilon1 = 1., epsilon2 = 100.;
  unsigned nsplit = 16;

  // Outer square vertices
  Eigen::Vector2d NEo(2., 2.);
  Eigen::Vector2d NWo(-2., 2.);
  Eigen::Vector2d SEo(2., -2.);
  Eigen::Vector2d SWo(-2., -2.);
  // Outer square edges
  parametricbem2d::ParametrizedLine Or(SEo, NEo); // right
  parametricbem2d::ParametrizedLine ot(NEo, NWo); // top
  parametricbem2d::ParametrizedLine ol(NWo, SWo); // left
  parametricbem2d::ParametrizedLine ob(SWo, SEo); // bottom

  // Panels for the edges of outer square
  parametricbem2d::PanelVector panels_or(Or.split(4*nsplit));
  parametricbem2d::PanelVector panels_ot(ot.split(4*nsplit));
  parametricbem2d::PanelVector panels_ol(ol.split(4*nsplit));
  parametricbem2d::PanelVector panels_ob(ob.split(4*nsplit));

  // Creating the ParametricMesh object
  parametricbem2d::PanelVector panels;

  if (argv[1] == std::string("0")) {
    std::cout << "Square" << std::endl;
      
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

    // Panels for the edges of the inner square
    parametricbem2d::PanelVector panels_ir(ir.split(nsplit));
    parametricbem2d::PanelVector panels_ib(ib.split(nsplit));
    parametricbem2d::PanelVector panels_il(il.split(nsplit));
    parametricbem2d::PanelVector panels_it(it.split(nsplit));

    panels.insert(panels.end(), panels_ir.begin(), panels_ir.end());
    panels.insert(panels.end(), panels_ib.begin(), panels_ib.end());
    panels.insert(panels.end(), panels_il.begin(), panels_il.end());
    panels.insert(panels.end(), panels_it.begin(), panels_it.end());
  }

  else if (argv[1] == std::string("1")) {
    std::cout << "Kite" << std::endl;

    parametricbem2d::PanelVector panels_kite;
    double lkite = 2.46756; // length of kite
    unsigned N = ceil(lkite * nsplit);
    Eigen::VectorXd meshpts = get_kite_params(N);
    Eigen::VectorXd tempp(N+1);
    tempp << meshpts, 2*M_PI;
    tempp = -tempp + Eigen::VectorXd::Constant(N+1,2*M_PI); // clockwise
    //std::cout << "temp: " << tempp.transpose() << std::endl;
    // Defining the kite domain
    Eigen::MatrixXd cos_list_o(2, 2);
    cos_list_o << .35, .1625, 0, 0;
    Eigen::MatrixXd sin_list_o(2, 2);
    sin_list_o << 0, 0, .35, 0;
    for (unsigned i = 0 ; i < N ; ++i) {
      panels_kite.push_back(
          std::make_shared<parametricbem2d::ParametrizedFourierSum>(
              Eigen::Vector2d(0.3, 0.5), cos_list_o, sin_list_o, 
              tempp(i), tempp(i+1)));
    }

    panels.insert(panels.end(), panels_kite.begin(), panels_kite.end());
  }

  panels.insert(panels.end(), panels_or.begin(), panels_or.end());
  panels.insert(panels.end(), panels_ot.begin(), panels_ot.end());
  panels.insert(panels.end(), panels_ol.begin(), panels_ol.end());
  panels.insert(panels.end(), panels_ob.begin(), panels_ob.end());

  parametricbem2d::ParametrizedMesh mesh(panels);

  auto g = [](Eigen::Vector2d x) {
    return 2. - x[0];
  };

  auto eta = [](Eigen::Vector2d x) {
    return 0;
  };

  auto dir_sel = [](const Eigen::Vector2d& x) {
    return (x[0] - 2. > -1e-7 || x[0] + 2. < 1e-7);
  };

  parametricbem2d::ContinuousSpace<1> space_d;
  parametricbem2d::DiscontinuousSpace<0> space_n;
  Eigen::VectorXd sol = transmission_bem::Solve(
      mesh, space_d, space_n, dir_sel, g, eta, epsilon1, epsilon2, order);

  transmission_bem::Dims dims_d, dims_n;
  transmission_bem::Indices ind_d, ind_n;
  transmission_bem::ComputeDirSpaceInfo(mesh, space_d, dir_sel, dims_d, ind_d);
  transmission_bem::ComputeNeuSpaceInfo(mesh, space_n, dir_sel, dims_n, ind_n);

  /*
  std::cout << "\nu_i" << std::endl;
  std::cout << sol.segment(0, dims_d.i) << std::endl;
  std::cout << "\npsi_i" << std::endl;
  std::cout << sol.segment(dims_d.i, dims_n.i) << std::endl;
  std::cout << "\nu" << std::endl;
  std::cout << sol.segment(dims_d.i + dims_n.i, dims_d.n) << std::endl;
  std::cout << "\npsi" << std::endl;
  std::cout << sol.tail(dims_n.d) << std::endl;
  */

  unsigned numpanels = mesh.getNumPanels();

  // Print Dirichlet trace
  out << "Dirichlet trace" << std::endl;
  out << std::setw(25) << "x"
      << std::setw(25) << "y"
      << std::setw(25) << "u" << std::endl;
  unsigned Q_d = space_d.getQ();
  Eigen::VectorXd u_i = sol.segment(0, dims_d.i);
  Eigen::VectorXd u = sol.segment(dims_d.i + dims_n.i, dims_d.n);
  Eigen::VectorXd g_interp =
      transmission_bem::InterpolateDirData(mesh, g, space_d, ind_d);
  Eigen::VectorXd u_all = Eigen::VectorXd::Zero(dims_d.all);
  for (unsigned k = 0; k < dims_d.i; ++k) {
    u_all[ind_d.i[k]] = u_i[k];
  }
  for (unsigned k = 0; k < dims_d.d; ++k) {
    u_all[ind_d.d[k]] = g_interp[k];
  }
  for (unsigned k = 0; k < dims_d.n; ++k) {
    u_all[ind_d.n[k]] = u[k];
  }
  for (unsigned i = 0; i < numpanels; ++i) {
    parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
    double local_coord = 0.;
    Eigen::Vector2d global_coord = pi(local_coord);
    double res = 0.;
    for (unsigned I = 0; I < Q_d; ++I) {
      unsigned II = space_d.LocGlobMap2(I + 1, i + 1, mesh) - 1;
      res += u_all[II] * space_d.evaluateShapeFunction(I, local_coord);
    }
    out << std::setw(25) << global_coord[0]
        << std::setw(25) << global_coord[1]
        << std::setw(25) << res << std::endl;
  }

  // Print Neumann trace
  out << "Neumann trace" << std::endl;
  out << std::setw(25) << "x"
      << std::setw(25) << "y"
      << std::setw(25) << "psi" << std::endl;
  unsigned Q_n = space_n.getQ();
  Eigen::VectorXd psi_i = sol.segment(dims_d.i, dims_n.i);
  Eigen::VectorXd psi = sol.tail(dims_n.d);
  Eigen::VectorXd eta_interp =
      transmission_bem::InterpolateNeuData(mesh, eta, space_n, ind_n);
  Eigen::VectorXd psi_all = Eigen::VectorXd::Zero(dims_n.all);
  for (unsigned k = 0; k < dims_n.i; ++k) {
    psi_all[ind_n.i[k]] = psi_i[k];
  }
  for (unsigned k = 0; k < dims_n.d; ++k) {
    psi_all[ind_n.d[k]] = psi[k];
  }
  for (unsigned k = 0; k < dims_n.n; ++k) {
    psi_all[ind_n.n[k]] = eta_interp[k];
  }
  for (unsigned i = 0; i < numpanels; ++i) {
    parametricbem2d::AbstractParametrizedCurve &pi = *panels[i];
    double local_coord = 0.;
    Eigen::Vector2d global_coord = pi(local_coord);
    double res = 0.;
    for (unsigned I = 0; I < Q_n; ++I) {
      unsigned II = space_n.LocGlobMap2(I + 1, i + 1, mesh) - 1;
      res += psi_all[II] * space_n.evaluateShapeFunction(I, local_coord);
    }
    out << std::setw(25) << global_coord[0]
        << std::setw(25) << global_coord[1]
        << std::setw(25) << res << std::endl;
  }

  return 0;
}
