#pragma once // 헤더 중복 포함 방지

#include <cmath>
#include <Eigen/Dense>

// M_PI가 정의되지 않은 환경(MSVC 등)을 위한 대비
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// [중요] 헤더에 구현체를 넣을 때는 'inline'을 붙여야 
// 여러 cpp 파일에서 include 해도 중복 정의 에러가 나지 않습니다.
inline double source_function(const Eigen::Vector3d& p) {
    double x = p(0);
    double y = p(1);
    // double z = p(2);

    return 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
}
