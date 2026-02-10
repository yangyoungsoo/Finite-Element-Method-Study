#pragma once

#include <Eigen/Sparse>
#include <Eigen/Dense>

Eigen::VectorXd solve_CG(
    const Eigen::SparseMatrix<double>& K_in,
    const Eigen::VectorXd& F_in
);