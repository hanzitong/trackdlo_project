// hello_trackdlo.cpp
//
// pure_trackdlo の最小構成サンプル
//
// 実際のシステムでは
//   カメラ画像 → preprocessing::color_threshold() → バイナリマスク
//             → preprocessing::images_to_pointcloud() → 点群 X
// という流れで点群を得る。
// ここでは preprocessing に依存しないよう、ケーブル形状の点群を
// Eigen で直接生成して代用する。
//
// 実行:
//   cd build && ./hello_trackdlo

#include <trackdlo.h>
#include <utils.h>

#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <vector>
#include <numeric>   // std::iota

// ============================================================
// 点群生成: バイナリマスク相当
//
// ケーブルを X 軸に沿った直線としてモデル化し、
// ガウシアンノイズを加えて実際のマスクから得られる点群を模倣する。
// bend_y を 0 以外にすると Y 方向に曲げた形状を作れる。
// ============================================================
static Eigen::MatrixXd make_cable_cloud(int N, double bend_y, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 0.005);  // 5mm の標準偏差

    Eigen::MatrixXd X(N, 3);
    for (int i = 0; i < N; i++) {
        double t    = static_cast<double>(i) / (N - 1);  // 0 〜 1
        X(i, 0) = t + noise(rng);
        X(i, 1) = bend_y * std::sin(t * M_PI) + noise(rng);  // 半円弧状に曲げる
        X(i, 2) = 1.0 + noise(rng);  // カメラから 1m 先
    }
    return X;
}

// ============================================================
// 初期化
//
// 点群 X から TrackdloState を作る。
// 1. cpd_lle で初期ノード Y_init を点群にフィットさせる
// 2. sort_pts でノードをケーブル沿いに順序付ける
// 3. geodesic_coord (各ノードの累積弧長) を計算する
//    ← tracking_step の呼び出し前に必須
// 4. guide_nodes を設定する (前フレームの可視ノード)
// ============================================================
static trackdlo::TrackdloState initialize(const Eigen::MatrixXd& X, int M) {
    trackdlo::TrackdloState state = trackdlo::make_trackdlo_state(M);

    // 点群の端点を結ぶ直線上に M 点の初期ノードを配置する
    Eigen::RowVectorXd x_min = X.colwise().minCoeff();
    Eigen::RowVectorXd x_max = X.colwise().maxCoeff();
    for (int i = 0; i < M; i++) {
        double t = static_cast<double>(i) / (M - 1);
        state.Y.row(i) = x_min + t * (x_max - x_min);
    }

    // cpd_lle: 初期ノードを点群にフィットさせる
    // (reg() は Y_init を無視して原点から始めるため、ここでは使わない)
    bool ok = trackdlo::cpd_lle(
        X, state.Y, state.sigma2,
        /*beta=*/5.0, /*lambda=*/1.0, /*lle_weight=*/1.0, /*mu=*/0.05
    );
    std::cout << "  cpd_lle converged: " << (ok ? "yes" : "no") << "\n";

    // sort_pts: 最近傍グリーディ探索でノードをケーブル沿いに並べ替える
    state.Y = trackdlo::sort_pts(state.Y);

    // geodesic_coord: 隣接ノード間距離の累積和
    // tracking_step の内部で traverse_euclidean が参照するため必須
    state.geodesic_coord.resize(M);
    state.geodesic_coord[0] = 0.0;
    for (int i = 1; i < M; i++) {
        double d = trackdlo::pt2pt_dis(state.Y.row(i), state.Y.row(i - 1));
        state.geodesic_coord[i] = state.geodesic_coord[i - 1] + d;
    }

    // guide_nodes: 前フレームの可視ノード。初期化時は現在のノードをセット。
    state.guide_nodes = state.Y;

    return state;
}

int main() {
    const int M       = 10;   // トラッキングノード数
    const int N       = 200;  // 点群の点数
    const int FRAMES  = 5;    // トラッキングフレーム数

    std::cout << "=== hello_trackdlo ===\n";
    std::cout << "nodes=" << M << "  points=" << N << "\n\n";

    // --------------------------------------------------------
    // Step 1: 初期フレームの点群を作り、状態を初期化する
    // --------------------------------------------------------
    std::cout << "[frame 0] initializing...\n";
    Eigen::MatrixXd X0 = make_cable_cloud(N, /*bend_y=*/0.0, /*seed=*/0);
    trackdlo::TrackdloState state = initialize(X0, M);

    std::cout << "  Y[0] = " << state.Y.row(0) << "\n";
    std::cout << "  Y[M-1] = " << state.Y.row(M - 1) << "\n";

    // --------------------------------------------------------
    // Step 2: パラメータ設定
    // --------------------------------------------------------
    trackdlo::TrackdloParams params;
    // デフォルト値のまま使う (beta=5, lambda=1, mu=0.05, max_iter=50)
    // 必要なら params.beta = 3.0; などで調整する

    // --------------------------------------------------------
    // Step 3: トラッキングループ
    //
    // visible_nodes: オクルージョンなしの場合は全ノードを渡す。
    // 実際のシステムでは preprocessing::compute_visible_nodes() で取得する。
    // --------------------------------------------------------
    std::vector<int> all_nodes(M);
    std::iota(all_nodes.begin(), all_nodes.end(), 0);  // {0, 1, 2, ..., M-1}

    for (int f = 1; f <= FRAMES; f++) {
        // フレームごとに曲がり具合を増やして点群を生成
        double bend = 0.05 * f;
        Eigen::MatrixXd X = make_cable_cloud(N, bend, /*seed=*/f);

        // トラッキング: state.Y と state.sigma2 が in-place で更新される
        trackdlo::tracking_step(state, X, all_nodes, all_nodes, params);

        // 結果を表示
        double length = state.geodesic_coord.back();
        std::cout << "[frame " << f << "]"
                  << "  bend=" << bend
                  << "  sigma2=" << state.sigma2
                  << "  length=" << length
                  << "  Y[mid]=" << state.Y.row(M / 2)
                  << "\n";
    }

    std::cout << "\ndone.\n";
    return 0;
}
