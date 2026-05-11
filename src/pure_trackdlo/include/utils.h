#pragma once

#include "trackdlo.h"
#include <iostream>

#ifndef UTILS_H
#define UTILS_H

void signal_callback_handler(int signum);

template <typename T> void print_1d_vector (const std::vector<T>& vec) {
    for (const auto& item : vec) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}

double pt2pt_dis_sq (Eigen::MatrixXd pt1, Eigen::MatrixXd pt2);
double pt2pt_dis (Eigen::MatrixXd pt1, Eigen::MatrixXd pt2);

// 簡易CPD登録: ROSノード起動時の初期化に使用する
void reg (Eigen::MatrixXd pts, Eigen::MatrixXd& Y, double& sigma2, int M, double mu = 0, int max_iter = 50);
void remove_row(Eigen::MatrixXd& matrix, unsigned int rowToRemove);

// 最近傍順でノードを並べ替える
Eigen::MatrixXd sort_pts (Eigen::MatrixXd Y_0);

// Pure Pursuitのための直線と球の交点計算
std::vector<Eigen::MatrixXd> line_sphere_intersection (Eigen::MatrixXd point_A, Eigen::MatrixXd point_B, Eigen::MatrixXd sphere_center, double radius);

Eigen::MatrixXd cross_product (Eigen::MatrixXd vec1, Eigen::MatrixXd vec2);
double dot_product (Eigen::MatrixXd vec1, Eigen::MatrixXd vec2);

#endif
