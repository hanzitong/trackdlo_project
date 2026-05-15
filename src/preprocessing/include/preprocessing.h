#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Dense>
#include <vector>

// ============================================================
// preprocessing.h
//
// 【目的】
//   trackdloアルゴリズムへの入力点群を生成するための前処理ライブラリ。
//   ROSに依存しない。cv::Mat を入力として受け取り、
//   Eigen::MatrixXd (N×3) の3D点群を返す。
//
// 【処理内容】
//   Step 1: color_threshold()
//     BGR画像をHSV色空間に変換し、指定した色範囲でケーブル領域を
//     2値マスク (cv::Mat, CV_8UC1) として抽出する。
//
//   Step 2: images_to_pointcloud()
//     マスクで有効なピクセルのみを選択し、深度値とピンホールカメラモデルで
//     3D座標 (X, Y, Z) に逆投影する。
//     逆投影式:
//       X = (u - cx) * Z / fx
//       Y = (v - cy) * Z / fy
//       Z = depth_mm / 1000.0   (mm → m 変換)
//     最後にPCLのVoxelGridで点群を間引き、密度を均一化する。
//
// 【パイプライン全体像】
//   BGR画像 + 深度画像
//       │
//       ▼ color_threshold()
//     2値マスク
//       │
//       ▼ images_to_pointcloud()
//     Eigen::MatrixXd X (N×3)  ← trackdlo の入力点群
//       │
//       ▼ compute_visible_nodes()  ← tracking_step() の前に呼ぶ
//     visible_nodes, visible_nodes_extended
//
//   Step 3: compute_visible_nodes()
//     前フレームのノード座標Yをカメラ画像に投影し、どのノードが可視かを判定する。
//     判定基準:
//       (a) 点群との最短距離が visibility_threshold 以下 (=点群に対応する位置にある)
//       (b) 自己オクルージョンしていない (手前のエッジに隠れていない)
//     visible_nodes_extended は小さなギャップ (d_vis 以内) を埋めた拡張版。
//
// 【依存ライブラリ】
//   OpenCV  : HSV変換・マスク処理・エッジ投影描画
//   Eigen3  : 点群の行列表現
//   PCL 1.8 : VoxelGridによるダウンサンプリング
// ============================================================

namespace preprocessing {

// HSV閾値でケーブル色を抽出し、2値マスクを返す
// rgb_bgr  : OpenCVのBGR画像 (CV_8UC3)
// lower    : HSV下限 [H, S, V]
// upper    : HSV上限 [H, S, V]
cv::Mat color_threshold(const cv::Mat& rgb_bgr,
                        const std::vector<int>& lower,
                        const std::vector<int>& upper);

// マルチカラーDLO用: 青・赤・黄の固定閾値でマスクを生成
cv::Mat color_threshold_multicolor(const cv::Mat& rgb_bgr);

// RGB画像 + 深度画像 → セグメント済み3D点群 (N×3)
//
// rgb_bgr     : OpenCVのBGR画像 (CV_8UC3)
// depth       : 深度画像 (CV_16UC1, 単位mm, RealSense標準形式)
// proj_matrix : カメラ射影行列 (3×4)
//               [ fx  0  cx  0 ]
//               [  0 fy  cy  0 ]
//               [  0  0   1  0 ]
// mask        : color_threshold()で生成した2値マスク
// leaf_size   : ボクセルグリッドのセルサイズ (m). 0以下で無効
// occ_mask    : オクルージョンマスク (BGR, 黒=隠す領域, 省略可)
Eigen::MatrixXd images_to_pointcloud(const cv::Mat& rgb_bgr,
                                     const cv::Mat& depth,
                                     const Eigen::MatrixXd& proj_matrix,
                                     const cv::Mat& mask,
                                     double leaf_size = 0.01,
                                     const cv::Mat& occ_mask = cv::Mat());

// ノード可視性計算: tracking_step() を呼ぶ前に実行する
//
// Y                  : 前フレームのノード座標 (M×3)
// X                  : 現フレームの入力点群 (N×3)
// proj_matrix        : カメラ射影行列 (3×4)
// geodesic_coord     : 各ノードの測地線座標 (弧長, M要素)
//                      visible_nodes_extended のギャップ判定に使用する
// img_rows, img_cols : カメラ画像サイズ (投影マスク生成に使用)
// visibility_threshold : ノード-点群間距離の可視判定閾値 (m)
// d_vis              : visible_nodes_extended のギャップ許容幅 (測地線距離)
// dlo_pixel_width    : 投影エッジの描画幅 (画素, セルフオクルージョン判定に影響)
// visible_nodes      : 出力 — 可視ノードのインデックス列 (ソート済み)
// visible_nodes_extended : 出力 — 小ギャップを埋めた拡張可視ノード列
void compute_visible_nodes(const Eigen::MatrixXd& Y,
                           const Eigen::MatrixXd& X,
                           const Eigen::MatrixXd& proj_matrix,
                           const std::vector<double>& geodesic_coord,
                           int img_rows,
                           int img_cols,
                           double visibility_threshold,
                           double d_vis,
                           int dlo_pixel_width,
                           std::vector<int>& visible_nodes,
                           std::vector<int>& visible_nodes_extended);

} // namespace preprocessing
