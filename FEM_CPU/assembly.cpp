#include "assembly.h"
#include "make_local.h"

#include <cmath>         
#include <chrono>
#include <iostream>

void assemble_system(
    const std::vector<Eigen::Triplet<double>>& tripletList,
    Eigen::SparseMatrix<double>& K_global,
    int num_nodes
) {
    auto start_assembling = std::chrono::high_resolution_clock::now();

    K_global.resize(num_nodes, num_nodes);
    K_global.setFromTriplets(tripletList.begin(), tripletList.end());

    auto end_assembling = std::chrono::high_resolution_clock::now();
    auto dura_assembling = std::chrono::duration_cast<std::chrono::microseconds>(end_assembling - start_assembling);

    std::cout << "Duration for Assemble System: " << dura_assembling.count() * 1e-6 << "seconds" << std::endl;
}
