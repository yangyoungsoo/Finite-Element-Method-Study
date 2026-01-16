#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "read_txt.h"

#include <cmath> 

struct local_stiffness_result {
    Eigen::Matrix <double, 4, 4> Ke;
    Eigen::Matrix <double, 3, 3> J;
    double det;
};

local_stiffness_result local_stiffness(const Eigen::MatrixXd& Vert, int n1, int n2, int n3, int n4) {

    Eigen::Matrix<double, 4, 3> GradPhiHat;
    GradPhiHat << -1, -1, -1,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1;

    auto node1 = Vert.row(n1);
    auto node2 = Vert.row(n2);
    auto node3 = Vert.row(n3);
    auto node4 = Vert.row(n4);

    double x1 = node1(0); double y1 = node1(1); double z1 = node1(2);
    double x2 = node2(0); double y2 = node2(1); double z2 = node2(2);
    double x3 = node3(0); double y3 = node3(1); double z3 = node3(2);
    double x4 = node4(0); double y4 = node4(1); double z4 = node4(2);

    Eigen::Matrix  <double, 3, 3> J;
    J << x2 - x1, x3 - x1, x4 - x1,
        y2 - y1, y3 - y1, y4 - y1,
        z2 - z1, z3 - z1, z4 - z1;

    double det = J.determinant();
    Eigen::Matrix3d Jinv = J.inverse();

    Eigen::Matrix <double, 4, 3> G = GradPhiHat * Jinv;

    Eigen::Matrix <double, 4, 4> Ke = (std::abs(det) / 6.0) * (G * G.transpose());

    return { Ke, J, det };
}
