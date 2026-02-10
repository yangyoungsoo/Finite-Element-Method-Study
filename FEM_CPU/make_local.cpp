#include "make_local.h"
#include "GQ.h"         

#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Sparse>
#include <chrono>

void make_local_K(
	const Eigen::MatrixXi& Tet,
	const Eigen::Matrix<double, 4, 3>& GradPhiHat,
	const std::vector<Eigen::Matrix3d>& JACOBIAN_inverse,
	const std::vector<double>& DETERMINANT,
	std::vector<Eigen::Triplet<double>>& tripletList,
	int num_elements) {

    auto start_make_local_K = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < num_elements; ++i) {
		Eigen::Matrix3d invJ = JACOBIAN_inverse[i];
		Eigen::Matrix<double, 4, 3>G;
		G = GradPhiHat * invJ;

        Eigen::Matrix<double, 4, 4> GGT = G * G.transpose();

        // 3. 강성 행렬 계산 (부피 적분)
        // |det|/6 이 사면체의 부피입니다.
        double vol = std::abs(DETERMINANT[i]) / 6.0;
        Eigen::Matrix<double, 4, 4> Ke = vol * GGT;

		auto r = Tet.row(i);
		int n1 = r(0), n2 = r(1), n3 = r(2), n4 = r(3);
		int nodes[4] = { n1, n2, n3, n4 };

		for (int row = 0; row < 4; row++) {
			for (int col = 0; col < 4; col++) {
				tripletList.push_back(Eigen::Triplet<double>(nodes[row], nodes[col], Ke(row, col)));
			}
		}
	}
    auto end_make_local_K = std::chrono::high_resolution_clock::now();
    auto dura_make_local_K = std::chrono::duration_cast<std::chrono::microseconds>(end_make_local_K - start_make_local_K);

    std::cout << "Duration for Local Stiffness: " << dura_make_local_K.count() * 1e-6 << "seconds" << std::endl;

}

void make_global_F(
    const Eigen::MatrixXi& Tet,
    const Eigen::MatrixXd& Vert, 
    const std::vector<Eigen::Matrix3d>& JACOBIAN, 
    const std::vector<double>& DETERMINANT,
    Eigen::VectorXd& F_global,          
    int num_elements,
    SourceFunc src_func
) {
    Eigen::MatrixXd quad_points = TriGaussPoints3D(2);
    int num_qp = quad_points.rows();

    auto start_make_global_F = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_elements; ++i) {
        auto r = Tet.row(i);
        int n1 = r(0), n2 = r(1), n3 = r(2), n4 = r(3);
        int nodes[4] = { n1, n2, n3, n4 };

        Eigen::Vector3d node1_pos = Vert.row(n1);
        Eigen::Vector4d local_F = Eigen::Vector4d::Zero();
        double abs_det = std::abs(DETERMINANT[i]);

        for (int q = 0; q < num_qp; q++) {
            double xi = quad_points(q, 0);
            double eta = quad_points(q, 1);
            double zeta = quad_points(q, 2);
            double weight = quad_points(q, 3);

            Eigen::Vector3d xi_vec(xi, eta, zeta);
            Eigen::Vector3d phys_pos = node1_pos + JACOBIAN[i] * xi_vec;

            double N[4];
            N[0] = 1.0 - xi - eta - zeta;
            N[1] = xi;
            N[2] = eta;
            N[3] = zeta;

            double f_val = src_func(phys_pos(0), phys_pos(1), phys_pos(2));

            double coeff = weight * f_val * abs_det;

            for (int k = 0; k < 4; ++k) {
                local_F(k) += coeff * N[k];
            }
        }

        for (int k = 0; k < 4; k++) {
            F_global(nodes[k]) += local_F(k);
        }
    }

    auto end_make_global_F = std::chrono::high_resolution_clock::now();
    auto dura_make_global_F = std::chrono::duration_cast<std::chrono::microseconds>(end_make_global_F - start_make_global_F);

    std::cout << "Duration for F vector: " << dura_make_global_F.count() * 1e-6 << "seconds" << std::endl;
}

