#pragma once

// ============================================================
// evaluator.h
//
// 【目的】
//   trackdloのトラッキング精度をグラウンドトゥルースと比較して定量評価する。
//   ROSに依存しない。ベンチマーク・実験専用のライブラリ。
//
// 【処理内容】
//   - get_ground_truth_nodes(): RGB画像+点群からマーカー色でグラウンドトゥルースノードを検出
//   - compute_error() / compute_and_save_error(): 予測ノード列と真値ノード列の誤差計算
//   - calc_min_distance(): 点から線分への最短距離（区分的誤差の基礎計算）
//   - get_piecewise_error(): 区分的Hausdorff誤差（片方向）
//
// 【依存ライブラリ】
//   Eigen3, OpenCV, PCL 1.8
// ============================================================

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <chrono>
#include <string>
#include <vector>

#ifndef EVALUATOR_H
#define EVALUATOR_H

class evaluator
{
    public:
        evaluator ();
        evaluator (int length, int trial, int pct_occlusion, std::string alg, int bag_file, std::string save_location,
                   double start_record_at, double exit_at, double wait_before_occlusion, double bag_rate, int num_of_nodes);

        // グラウンドトゥルースノードをRGB画像+点群から検出する
        Eigen::MatrixXd get_ground_truth_nodes (cv::Mat rgb_img, pcl::PointCloud<pcl::PointXYZRGB> cloud_xyz);

        // ノード列を連続したチェーン順に並べ替える (head: 開始端ノード)
        Eigen::MatrixXd sort_pts (Eigen::MatrixXd Y_0, Eigen::MatrixXd head);

        // 点Eから線分ABへの最短距離を返す。closest_pt_on_AB_to_E は最近傍点
        double calc_min_distance (Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd E,
                                  Eigen::MatrixXd& closest_pt_on_AB_to_E);

        // 区分的誤差: Y_track の各点から Y_true の曲線への平均最短距離
        double get_piecewise_error (Eigen::MatrixXd Y_track, Eigen::MatrixXd Y_true);

        // 双方向区分的誤差を計算してファイルに保存する
        double compute_and_save_error (Eigen::MatrixXd Y_track, Eigen::MatrixXd Y_true);

        // 双方向区分的誤差を計算して返す (ファイル保存なし)
        double compute_error (Eigen::MatrixXd Y_track, Eigen::MatrixXd Y_true);

        void set_start_time (std::chrono::steady_clock::time_point cur_time);
        void increment_image_counter ();

        double pct_occlusion ();
        std::chrono::steady_clock::time_point start_time ();
        double recording_start_time ();
        double exit_time ();
        int length ();
        double wait_before_occlusion ();
        double rate ();
        int image_counter ();

    private:
        int length_;
        int trial_;
        int pct_occlusion_;
        std::string alg_;
        int bag_file_;
        std::vector<double> errors_;
        std::string save_location_;
        std::chrono::steady_clock::time_point start_time_;
        double start_record_at_;
        double exit_at_;
        double wait_before_occlusion_;
        bool cleared_file_;
        double bag_rate_;
        int image_counter_;
        int num_of_nodes_;
};

#endif
