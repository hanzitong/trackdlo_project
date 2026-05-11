#pragma once

// Eigen: 行列演算ライブラリ。アルゴリズム全体でMatrixXdを主要データ型として使用する
#include <Eigen/Dense>
#include <vector>

#ifndef TRACKDLO_H
#define TRACKDLO_H

// ============================================================
// [設計メモ] なぜ struct + フリー関数なのか
//
// ROS1 では trackdlo をクラスとして実装することで
// コールバック間のフレーム間状態 (Y_, sigma2_ 等) を
// メンバ変数として保持していた。
//
// ROS2 では rclcpp::Node を継承したクラスを必ず作ることになる。
// そのクラスの中でさらに trackdlo クラスを持つと「クラスの中のクラス」
// になり、構造が無駄に複雑になる。
//
// そこで trackdlo の実装を struct + フリー関数に分解する:
//   - TrackdloParams  : セッション全体で変化しないパラメータ
//   - TrackdloState   : フレーム間で引き継ぐ可変状態
//   - フリー関数       : 状態を引数として受け取り更新する
//
// ROS2 ノードクラスは TrackdloState/Params をメンバ変数として持ち、
// コールバックから tracking_step() を呼ぶだけになる。
// 詳細: ~/port_trackdlo/docs/architecture/trackdlo_class_vs_functions.md
// ============================================================

// セッション全体で変化しないパラメータ
struct TrackdloParams {
    double beta            = 5.0;
    double lambda          = 1.0;
    double alpha           = 0.0;
    double k_vis           = 0.0;
    double mu              = 0.05;
    int    max_iter        = 50;
    double tol             = 1e-5;
    double beta_pre_proc   = 3.0;
    double lambda_pre_proc = 1.0;
    double lle_weight      = 1.0;
    double visibility_threshold = 0.02;
};

// フレーム間で引き継ぐ可変状態
struct TrackdloState {
    Eigen::MatrixXd Y;                                    // ノード座標 (M×3)
    Eigen::MatrixXd guide_nodes;                          // 前フレームの可視ノード座標
    double sigma2 = 0.0;                                  // ガウシアン分散 (フレーム間で引き継ぐ)
    std::vector<double> geodesic_coord;                   // 各ノードの累積弧長 (測地線座標)
    std::vector<Eigen::MatrixXd> correspondence_priors;   // オクルージョン補間結果 (次フレームへ)
};

// num_nodes 個のノードを持つ初期状態を返す
TrackdloState make_trackdlo_state(int num_nodes);

// CPD (Coherent Point Drift) + LLE (Locally Linear Embedding) 登録
// X: 入力点群 (N×3)
// Y, sigma2: in/out — 更新されてトラッキング結果になる
bool cpd_lle(const Eigen::MatrixXd& X,
             Eigen::MatrixXd& Y,
             double& sigma2,
             double beta,
             double lambda,
             double lle_weight,
             double mu,
             int max_iter = 30,
             double tol = 0.0001,
             bool include_lle = true,
             std::vector<Eigen::MatrixXd> correspondence_priors = {},
             double alpha = 0,
             std::vector<int> visible_nodes = {},
             double k_vis = 0,
             double visibility_threshold = 0.01);

// フレームごとのトラッキング処理。state を in-place で更新する。
// 呼ぶ前に state.geodesic_coord を初期化しておくこと。
// visible_nodes / visible_nodes_extended は preprocessing::compute_visible_nodes() で取得する。
void tracking_step(TrackdloState& state,
                   const Eigen::MatrixXd& X,
                   const std::vector<int>& visible_nodes,
                   const std::vector<int>& visible_nodes_extended,
                   const TrackdloParams& params);

#endif
