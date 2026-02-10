#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

Eigen::VectorXi find_boundary_indices(const Eigen::MatrixXd& V);

void remain_dof(
    const Eigen::MatrixXd& Vert,
    const Eigen::SparseMatrix<double>& K_global,
    const Eigen::VectorXd& F_global,
    Eigen::SparseMatrix<double>& K_in,
    Eigen::VectorXd& F_in,
    Eigen::VectorXi& in_indices);