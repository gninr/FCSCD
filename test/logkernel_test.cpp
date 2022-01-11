#include "factors.hpp"
#include "force_calculation_bem.hpp"
#include "velocity_fields.hpp"

#include <iostream>
#include <math.h>

#include "abstract_parametrized_curve.hpp"
#include "discontinuous_space.hpp"
#include "parametrized_circular_arc.hpp"
#include "parametrized_line.hpp"
#include "parametrized_mesh.hpp"
#include <Eigen/Dense>

int main() {
  std::cout << "Test log kernel" << std::endl;
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
  
  unsigned nsplit = 16;

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
  transmission_bem::NuRadial nu;

  // Compute Space Information
  transmission_bem::Dims dims_n;
  transmission_bem::Indices ind_n;
  transmission_bem::ComputeNeuSpaceInfo(
      mesh, space_n, dir_sel, &dims_n, &ind_n);

  transmission_bem::LogKernel kernel;
  transmission_bem::Factor1 F;
  Eigen::MatrixXd mat = transmission_bem::Slice(
      transmission_bem::ComputeMatrix(mesh, space_n, space_n,
          dims_n.all, dims_n.all, kernel, F, F, nu, order),
      ind_n.i, ind_n.i);
  Eigen::MatrixXd mat_ref =
      parametricbem2d::single_layer::GalerkinMatrix(mesh, space_n, order);
  Eigen::MatrixXd mat_ref_ii =
      transmission_bem::Slice(mat_ref, ind_n.i, ind_n.i);

  //std::cout << "mat =\n" << mat << std::endl;
  //std::cout << "mat_ref_ii =\n" << mat_ref_ii << std::endl;
  std::cout << "mat.norm = " << mat.norm() << std::endl;
  std::cout << "diff = " << (mat - mat_ref_ii).norm() << std::endl;

  return 0;
}