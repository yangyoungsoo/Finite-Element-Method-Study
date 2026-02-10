#include "indexing.h"
#include <vector>
#include <cmath>
#include <igl/slice.h>
#include <igl/setdiff.h>
#include <chrono>


Eigen::VectorXi find_boundary_indices(const Eigen::MatrixXd& V) {
    std::vector<int> indices;

    const double eps = 1e-9;

    for (int i = 0; i < V.rows(); ++i) {
        double x = V(i, 0);
        double y = V(i, 1);
        double z = V(i, 2);

        double eps = 1e-9;
        bool is_x_bnd = (std::abs(x) < eps) || (std::abs(x - 1.0) < eps);
        bool is_y_bnd = (std::abs(y) < eps) || (std::abs(y - 1.0) < eps);
        bool is_z_bnd = (std::abs(z) < eps) || (std::abs(z - 1.0) < eps);

        if (is_x_bnd || is_y_bnd || is_z_bnd) {
            indices.push_back(i);
        }
    }

    // vector -> VectorXi
    return Eigen::Map<Eigen::VectorXi>(indices.data(), indices.size());
}

void remain_dof(
    const Eigen::MatrixXd& Vert,
    const Eigen::SparseMatrix<double>& K_global,
    const Eigen::VectorXd& F_global,
    Eigen::SparseMatrix<double>& K_in,
    Eigen::VectorXd& F_in,
    Eigen::VectorXi& in_indices)
{
    int num_nodes = Vert.rows();

    auto start_reduction = std::chrono::high_resolution_clock::now();

    Eigen::VectorXi bnd_indices = find_boundary_indices(Vert);
    Eigen::VectorXi all_indices = Eigen::VectorXi::LinSpaced(num_nodes, 0, num_nodes - 1);
    Eigen::VectorXi ia;

    igl::setdiff(all_indices, bnd_indices, in_indices, ia);

    igl::slice(K_global, in_indices, in_indices, K_in);
    igl::slice(F_global, in_indices, F_in);

    auto end_reduction = std::chrono::high_resolution_clock::now();
    auto dura_reduction = std::chrono::duration_cast<std::chrono::microseconds>(end_reduction - start_reduction);

    std::cout << "Duration for System Reduction: " << dura_reduction.count() * 1e-6 << "seconds" << std::endl;
}