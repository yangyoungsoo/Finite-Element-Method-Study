#include "solver.h"
#include <iostream>
#include <Eigen/IterativeLinearSolvers>
#include <chrono>

Eigen::VectorXd solve_CG(
    const Eigen::SparseMatrix<double>& K_in,
    const Eigen::VectorXd& F_in)
{
    

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> solver;
    solver.setTolerance(1e-6);

    auto start_solving = std::chrono::high_resolution_clock::now();

    solver.compute(K_in);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Solver compute step failed! (Matrix might be singular)" << std::endl;
        return Eigen::VectorXd();
    }

    Eigen::VectorXd u_in = solver.solve(F_in);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Solving failed! (Did not converge)" << std::endl;
        return Eigen::VectorXd();
    }

    auto end_solving = std::chrono::high_resolution_clock::now();
    auto dura_solving = std::chrono::duration_cast<std::chrono::microseconds>(end_solving - start_solving);

    std::cout << "Duration for Solving System: " << dura_solving.count() * 1e-6 << "seconds\n" << std::endl;

    std::cout << "            Conjugate Gradient Method            \n" << std::endl;
    std::cout << "  - Iterations taken: " << solver.iterations() << std::endl;
    std::cout << "  - Estimated Error:  " << solver.error() << std::endl;


    return u_in;
}