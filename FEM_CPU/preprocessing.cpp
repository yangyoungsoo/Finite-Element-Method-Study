#include "preprocessing.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

void preprocessing(
    const Eigen::MatrixXi& Tet,
    const Eigen::MatrixXd& Vert,
    std::vector<Eigen::Matrix3d>& JACOBIAN,
    std::vector<Eigen::Matrix3d>& JACOBIAN_inverse,
    std::vector<double>& DETERMINANT,
    int num_elements,
    int num_nodes
) {
    auto start_preprocessing = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_elements; ++i) {
        auto r = Tet.row(i);
        int n1 = r(0), n2 = r(1), n3 = r(2), n4 = r(3);

        const double x1 = Vert(n1, 0), y1 = Vert(n1, 1), z1 = Vert(n1, 2);
        const double x2 = Vert(n2, 0), y2 = Vert(n2, 1), z2 = Vert(n2, 2);
        const double x3 = Vert(n3, 0), y3 = Vert(n3, 1), z3 = Vert(n3, 2);
        const double x4 = Vert(n4, 0), y4 = Vert(n4, 1), z4 = Vert(n4, 2);

        const double x21 = x2 - x1, y21 = y2 - y1, z21 = z2 - z1;
        const double x31 = x3 - x1, y31 = y3 - y1, z31 = z3 - z1;
        const double x41 = x4 - x1, y41 = y4 - y1, z41 = z4 - z1;

        Eigen::Matrix<double, 3, 3> J;
        J << x21, x31, x41,
            y21, y31, y41,
            z21, z31, z41;

        const double det =
            x21 * (y31 * z41 - y41 * z31)
            - x31 * (y21 * z41 - y41 * z21)
            + x41 * (y21 * z31 - y31 * z21);

        const double J11 = y31 * z41 - y41 * z31, J12 = x41 * z31 - x31 * z41, J13 = x31 * y41 - x41 * y31;
        const double J21 = y41 * z21 - y21 * z41, J22 = x21 * z41 - x41 * z21, J23 = x41 * y21 - x21 * y41;
        const double J31 = y21 * z31 - y31 * z21, J32 = x31 * z21 - x21 * z31, J33 = x21 * y31 - x31 * y21;

        Eigen::Matrix<double, 3, 3> Jinv;
        Jinv << J11, J12, J13,
            J21, J22, J23,
            J31, J32, J33;

        Jinv = (1 / det) * Jinv;

        JACOBIAN[i] = J;    
        JACOBIAN_inverse[i] = Jinv;
        DETERMINANT[i] = det;
    }

    auto end_preprocessing = std::chrono::high_resolution_clock::now();
    auto dura_preprocessing = std::chrono::duration_cast<std::chrono::microseconds>(end_preprocessing - start_preprocessing);

    std::cout << "Duration for Preprocessing: " << dura_preprocessing.count() * 1e-6 << "seconds" << std::endl;

}