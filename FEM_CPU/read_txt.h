#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <chrono> 

// 데이터를 불러오는 함수 (Nx3, double)
// Vertex 좌표 등을 읽을 때 사용
inline Eigen::MatrixXd read_Vert(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "파일을 열 수 없습니다: " << filename << std::endl;
        return Eigen::MatrixXd(0, 0);
    }

    auto start_load = std::chrono::high_resolution_clock::now();

    std::vector<double> values;
    double val;
    // 파일의 끝까지 모든 숫자를 읽어서 벡터에 저장 (공백, 줄바꿈 자동 처리)
    while (file >> val) {
        values.push_back(val);
    }
    file.close();

    // 데이터 개수가 3의 배수인지 확인 (3D 좌표이므로)
    if (values.size() % 3 != 0) {
        std::cerr << "데이터 개수가 3의 배수가 아닙니다. 파일 형식을 확인하세요." << std::endl;
        return Eigen::MatrixXd(0, 0);
    }

    long rows = values.size() / 3;
    long cols = 3;

    // std::vector의 메모리를 Eigen Matrix로 매핑 (복사 비용 절감)
    // 파일 데이터는 행 우선(RowMajor) 순서로 저장되어 있으므로 RowMajor로 매핑합니다.
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat_row_major =
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), rows, cols);

    auto end_load = std::chrono::high_resolution_clock::now();
    auto dura_load = std::chrono::duration_cast<std::chrono::microseconds> (end_load - start_load);

    std::cout << "Duration for load vert.txt: " << dura_load.count() * 1e-6 << "seconds" << std::endl;

    // 일반적인 Eigen::MatrixXd(ColumnMajor)로 변환하여 반환
    return mat_row_major;
}

// Nx4 크기의 정수형 데이터를 불러오는 함수 (Nx4, int)
// Tetrahedron 요소 정보 등을 읽을 때 사용
inline Eigen::MatrixXi read_Tet(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "파일을 열 수 없습니다: " << filename << std::endl;
        return Eigen::MatrixXi(0, 0);
    }

    auto start_load = std::chrono::high_resolution_clock::now();

    std::vector<int> values;
    int val;
    // 파일의 끝까지 모든 숫자를 읽어서 벡터에 저장
    while (file >> val) {
        values.push_back(val);
    }
    file.close();

    // 데이터 개수가 4의 배수인지 확인 (한 줄에 4개씩이므로)
    if (values.size() % 4 != 0) {
        std::cerr << "데이터 개수가 4의 배수가 아닙니다. 파일 형식을 확인하세요." << std::endl;
        return Eigen::MatrixXi(0, 0);
    }

    long rows = values.size() / 4;
    long cols = 4;

    // std::vector의 메모리를 Eigen Matrix로 매핑
    // 파일이 행 단위(RowMajor)로 저장되어 있으므로 RowMajor 옵션 사용
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat_row_major =
        Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), rows, cols);

    // 만약 tet 파일이 1-based라면 아래 코드를 추가하여 0-based로 변환
    if (mat_row_major.minCoeff() == 1) {
        mat_row_major.array() -= 1;
    }

    auto end_load = std::chrono::high_resolution_clock::now();
    auto dura_load = std::chrono::duration_cast<std::chrono::microseconds> (end_load - start_load);

    std::cout << "Duration for load tet.txt: " << dura_load.count() * 1e-6 << "seconds" << std::endl;

    return mat_row_major;
}