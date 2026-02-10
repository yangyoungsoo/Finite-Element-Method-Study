#pragma once

#include <Eigen/Dense>
#include <stdexcept>
#include <string>
#include "GQ.h"

Eigen::MatrixXd TriGaussPoints2D(int n);
Eigen::MatrixXd TriGaussPoints3D(int n);