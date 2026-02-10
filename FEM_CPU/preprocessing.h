#pragma once

#include <Eigen/Dense>
#include <vector>

void preprocessing(
    const Eigen::MatrixXi& Tet,
    const Eigen::MatrixXd& Vert,
    std::vector<Eigen::Matrix3d>& JACOBIAN,
    std::vector<Eigen::Matrix3d>& JACOBIAN_inverse,
    std::vector<double>& DETERMINANT,
    int num_element,
    int num_nodes);