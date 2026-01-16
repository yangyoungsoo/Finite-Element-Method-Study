#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "read_txt.h"
#include "Stiffness.h"
#include "GQ.h"

#include <cmath> 

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double source_function(double x, double y) {
    return 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
}

int main() {

    std::string tet_path = "tet.txt";
    std::string vert_path = "vert.txt";

    Eigen::MatrixXi Tet = read_Tet(tet_path);
    Eigen::MatrixXd Vert = read_Vert(vert_path);

    std::cout << "Tetrahedron: " << Tet.rows() << std::endl;

    int num_nodes = Vert.rows(); int Tet_size = Tet.rows();

    // Sparse Matrix
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList; // Triplet(r, c, v)

    tripletList.reserve(Tet_size * 16);

    Eigen::VectorXd F_global(num_nodes);
    F_global.setZero();

    int quadrature_order = 4;
    Eigen::MatrixXd quad_points = GQ::TriGaussPoints3D(quadrature_order);

    int num_qp = quad_points.rows(); 

    for (int i = 0; i < Tet_size; i++) {
        auto r = Tet.row(i);
        int n1 = r(0), n2 = r(1), n3 = r(2), n4 = r(3);
        int nodes[4] = { n1, n2, n3, n4 };

        auto res = local_stiffness(Vert, n1, n2, n3, n4);

        Eigen::Vector3d node1_pos = Vert.row(n1);
        Eigen::Vector4d local_F = Eigen::Vector4d::Zero();

        for (int q = 0; q < num_qp; q++) {

            double xi = quad_points(q, 0);
            double eta = quad_points(q, 1);
            double zeta = quad_points(q, 2);
            double weight = quad_points(q, 3);

            Eigen::Vector3d xi_vec(xi, eta, zeta);
            Eigen::Vector3d phys_pos = node1_pos + res.J * xi_vec;

            double N1 = 1.0 - xi - eta - zeta;

            double f_val = source_function(phys_pos(0), phys_pos(1));

            double coeff = weight * f_val * std::abs(res.det);

            local_F(0) += coeff * N1;
            local_F(1) += coeff * xi;  
            local_F(2) += coeff * eta;  
            local_F(3) += coeff * zeta;
        }

        for (int k = 0; k < 4; k++) {
            F_global(nodes[k]) += local_F(k);
        }

        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                tripletList.push_back(T(nodes[row], nodes[col], res.Ke(row, col)));
            }
        }
    }

    // sparse matrix
    Eigen::SparseMatrix<double> K_global(num_nodes, num_nodes);
    K_global.setFromTriplets(tripletList.begin(), tripletList.end());

    int limit = 10;

    std::cout << "\n--- F_global [0 ~ " << limit << "] ---" << std::endl;
    for (int i = 0; i <= limit && i < F_global.size(); ++i) {
        std::cout << "F(" << i << "): " << F_global(i) << std::endl;
    }

    std::cout << "\n--- K_global (Row, Col): Value [0 ~ " << limit << "] ---" << std::endl;

    for (int k = 0; k < K_global.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(K_global, k); it; ++it) {

            if (it.row() <= limit && it.col() <= limit) {
                std::cout << "(" << it.row() << "," << it.col() << "): " << it.value() << std::endl;
            }

        }
    }

    return 0;
}
