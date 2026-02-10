#include <iostream>
#include <string>
#include <vector>
#include <cmath> 
#include <chrono> 

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <igl/slice.h>
#include <igl/setdiff.h>

#include "read_txt.h"
#include "preprocessing.h"
#include "make_local.h"
#include "indexing.h"
#include "assembly.h"
#include "solver.h"
//#include "save.h"

#define M_PI 3.14159265358979323846

double source_function(double x, double y, double z) {
    return 3.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y) * std::sin(M_PI * z);
}

double exact_function(double x, double y, double z) {
    return std::sin(M_PI * x) * std::sin(M_PI * y) * std::sin(M_PI * z);
}

void partion(int n) {
    std::cout << std::string(n, '-') << std::endl;
}

int main() {

    Eigen::Matrix<double, 4, 3> GradPhiHat;
    GradPhiHat << -1, -1, -1,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1;

    std::string tet_path = "delaunay/cube_Tri_h_128.txt";
    std::string vert_path = "delaunay/cube_Node_h_128.txt";

    Eigen::MatrixXi Tet = read_Tet(tet_path); Eigen::MatrixXd Vert = read_Vert(vert_path);

    int num_nodes = Vert.rows();
    int num_elements = Tet.rows();

    std::vector<Eigen::Matrix3d> JACOBIAN;
    std::vector<Eigen::Matrix3d> JACOBIAN_inverse;
    std::vector<double> DETERMINANT;

    JACOBIAN.resize(num_elements);
    JACOBIAN_inverse.resize(num_elements);
    DETERMINANT.resize(num_elements);

    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(num_elements * 16);

    Eigen::SparseMatrix<double> K_global;
    Eigen::VectorXd F_global = Eigen::VectorXd::Zero(num_nodes);
    Eigen::SparseMatrix<double> K_in; Eigen::VectorXd F_in;
    Eigen::VectorXi in_indices;

    partion(50);

    std::cout << "Tet size: " << Tet.rows() << "\n" << "Vert size: " << Vert.rows() << std::endl;

    partion(50);

    std::cout << "            Elapsed time            \n" << std::endl;

    auto start_solving = std::chrono::high_resolution_clock::now();

    preprocessing(Tet, Vert, JACOBIAN, JACOBIAN_inverse, DETERMINANT, num_elements, num_nodes);
    make_local_K(Tet, GradPhiHat, JACOBIAN_inverse, DETERMINANT, tripletList, num_elements);
    make_global_F(Tet, Vert, JACOBIAN, DETERMINANT, F_global, num_elements, source_function);
    assemble_system(tripletList, K_global, num_nodes);
    remain_dof(Vert, K_global, F_global, K_in, F_in, in_indices);
    Eigen::VectorXd u_in = solve_CG(K_in, F_in);

    auto end_solving = std::chrono::high_resolution_clock::now();
    auto dura_solving = std::chrono::duration_cast<std::chrono::microseconds>(end_solving - start_solving);

    std::cout << "\nTotal time: " << dura_solving.count() * 1e-6 << "seconds" << std::endl;

    partion(50);

    // 7. 결과 분석
    Eigen::VectorXd u_full = Eigen::VectorXd::Zero(num_nodes);
    for (int i = 0; i < in_indices.size(); ++i) {
        u_full(in_indices(i)) = u_in(i);
    }

    // Exact Solution 비교
    Eigen::VectorXd u_exact(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        u_exact(i) = exact_function(Vert(i, 0), Vert(i, 1), Vert(i, 2));
    }

    Eigen::VectorXd diff = u_full - u_exact;

    double l2_error_approx = diff.norm() / std::sqrt(num_nodes);

    double exact_norm = u_exact.norm();
    double max_error = (u_full - u_exact).cwiseAbs().maxCoeff();

    std::cout << "            Error Analysis            \n" << std::endl;
    //std::cout << "Relative L2 Error: " << (exact_norm > 1e-20 ? l2_error_approx / exact_norm : l2_error_approx) << std::endl;
    std::cout << "Relative L2 Error: " << l2_error_approx << std::endl;
    std::cout << "Max Error: " << max_error << std::endl;

    //SaveVec_txt(u_full, "solution_64");

    return 0;
}
