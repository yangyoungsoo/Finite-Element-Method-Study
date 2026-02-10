#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <functional> // std::function을 위해 필요

using SourceFunc = std::function<double(double, double, double)>;

void make_local_K(
    const Eigen::MatrixXi& Tet,
    const Eigen::Matrix<double, 4, 3>& GradPhiHat,
    const std::vector<Eigen::Matrix3d>& JACOBIAN_inverse,
    const std::vector<double>& DETERMINANT,
    std::vector<Eigen::Triplet<double>>& tripletList,
    int num_elements);

void make_global_F(
    const Eigen::MatrixXi& Tet,
    const Eigen::MatrixXd& Vert,
    const std::vector<Eigen::Matrix3d>& JACOBIAN,
    const std::vector<double>& DETERMINANT,
    Eigen::VectorXd& F_global,
    int num_elements,
    SourceFunc src_func);