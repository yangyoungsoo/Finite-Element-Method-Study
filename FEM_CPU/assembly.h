#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

void assemble_system(
    const std::vector<Eigen::Triplet<double>>& tripletList,
    Eigen::SparseMatrix<double>& K_global,
    int num_nodes
);