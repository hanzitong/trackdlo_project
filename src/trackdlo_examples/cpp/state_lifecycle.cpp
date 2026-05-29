// state_lifecycle.cpp
//
// pure_trackdlo の使い方とメモリ管理パターンを示すサンプル。
// hello_trackdlo.cpp が「動かし方」を示すのに対して、
// こちらは「状態の寿命管理」に焦点を当てている。
//
// 扱うパターン:
//   Pattern 1: スタック確保 (推奨)
//   Pattern 2: std::optional<TrackdloState> (「まだ初期化していない」状態の表現)
//   Pattern 3: std::unique_ptr<TrackdloState> (動的確保が必要な場合)
//   Pattern 4: 生ポインタ (やってはいけない例)
//
// ビルド & 実行:
//   cd build && make && ./state_lifecycle

#include <trackdlo.h>
#include <utils.h>

#include <Eigen/Dense>
#include <iostream>
#include <optional>    // std::optional (C++17)
#include <memory>      // std::unique_ptr, std::make_unique
#include <vector>
#include <numeric>     // std::iota
#include <random>

// ----------------------------------------------------------------
// 合成点群の生成 (RealSense の代わり)
// ----------------------------------------------------------------
static Eigen::MatrixXd make_cloud(int N, double bend, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 0.005);
    Eigen::MatrixXd X(N, 3);
    for (int i = 0; i < N; i++) {
        double t = static_cast<double>(i) / (N - 1);
        X(i, 0) = t + noise(rng);
        X(i, 1) = bend * std::sin(t * M_PI) + noise(rng);
        X(i, 2) = 1.0 + noise(rng);
    }
    return X;
}

// ----------------------------------------------------------------
// 初期化 (全パターン共通の処理)
// ----------------------------------------------------------------
static trackdlo::TrackdloState make_state(const Eigen::MatrixXd& X, int M) {
    trackdlo::TrackdloState state = trackdlo::make_trackdlo_state(M);

    Eigen::RowVectorXd x_min = X.colwise().minCoeff();
    Eigen::RowVectorXd x_max = X.colwise().maxCoeff();
    for (int i = 0; i < M; i++) {
        double t = static_cast<double>(i) / (M - 1);
        state.Y.row(i) = x_min + t * (x_max - x_min);
    }

    trackdlo::cpd_lle(X, state.Y, state.sigma2, 5.0, 1.0, 1.0, 0.05);
    state.Y = trackdlo::sort_pts(state.Y);

    state.geodesic_coord.resize(M);
    state.geodesic_coord[0] = 0.0;
    for (int i = 1; i < M; i++) {
        state.geodesic_coord[i] = state.geodesic_coord[i-1]
            + trackdlo::pt2pt_dis(state.Y.row(i), state.Y.row(i-1));
    }
    state.guide_nodes = state.Y;
    return state;
}

// ================================================================
// Pattern 1: スタック確保 (推奨)
//
// TrackdloState はただの struct。メンバはすべて Eigen::MatrixXd と
// std::vector という RAII 型なので、スコープを抜けると自動で解放される。
// new / delete は一切不要。
// ================================================================
static void pattern1_stack() {
    std::cout << "\n[Pattern 1] スタック確保\n";

    const int M = 8;
    const int N = 150;

    Eigen::MatrixXd X0 = make_cloud(N, 0.0, 0);

    // TrackdloState をスタックに確保する。
    // スコープを抜けると自動で解放される (デストラクタが Eigen/STL を解放)。
    trackdlo::TrackdloState state = make_state(X0, M);
    trackdlo::TrackdloParams params;
    params.max_iter = 150;
    params.tol      = 1e-4;

    std::vector<int> all(M);
    std::iota(all.begin(), all.end(), 0);

    for (int f = 1; f <= 3; f++) {
        Eigen::MatrixXd X = make_cloud(N, 0.05 * f, f);
        trackdlo::tracking_step(state, X, all, all, params);
    }

    std::cout << "  Y[0]   = " << state.Y.row(0) << "\n";
    std::cout << "  Y[M-1] = " << state.Y.row(M - 1) << "\n";

    // ← ここで state がスコープを抜け、自動で解放される
}

// ================================================================
// Pattern 2: std::optional<TrackdloState>
//
// 「まだ初期化していない」状態を表現したいとき。
// Python の state = None に相当する C++ のイディオム。
//
// std::optional<T> は T のオブジェクトを「持っているかもしれない」型。
// - 未初期化: std::nullopt
// - 初期化済み: state.emplace(...) または state = make_state(...)
// - アクセス:   state->Y, (*state).Y
//
// メモリ管理: optional が破棄されると内包する TrackdloState も自動で破棄。
// ================================================================
static void pattern2_optional() {
    std::cout << "\n[Pattern 2] std::optional<TrackdloState>\n";

    const int M = 8;
    const int N = 150;

    // 未初期化状態。Python の state = None と同等。
    std::optional<trackdlo::TrackdloState> state = std::nullopt;

    trackdlo::TrackdloParams params;
    params.max_iter = 150;
    params.tol      = 1e-4;

    std::vector<int> all(M);
    std::iota(all.begin(), all.end(), 0);

    for (int f = 0; f < 4; f++) {
        Eigen::MatrixXd X = make_cloud(N, 0.05 * f, f);

        if (!state.has_value()) {
            // まだ初期化されていない → 初期化する
            state = make_state(X, M);  // optional に値を代入
            std::cout << "  frame " << f << ": initialized\n";
        } else {
            // 初期化済み → トラッキング
            // state-> で optional の中の TrackdloState にアクセス
            trackdlo::tracking_step(*state, X, all, all, params);
            std::cout << "  frame " << f
                      << ": Y[0]=" << state->Y.row(0) << "\n";
        }
    }

    // 再初期化したいときは nullopt を代入して古い state を破棄する
    state = std::nullopt;  // 古い TrackdloState が自動で解放される
    std::cout << "  reset: state = nullopt\n";

    // ← ここで optional 自体がスコープを抜け、内包する state も解放される
}

// ================================================================
// Pattern 3: std::unique_ptr<TrackdloState>
//
// 動的確保が必要な場面 (ヒープに置きたい、遅延初期化など) のパターン。
// optional で十分な場合がほとんどだが、ポリモーフィズムや
// 所有権の移動 (std::move) が必要なときに使う。
//
// unique_ptr は所有権を 1 つのポインタだけが持つスマートポインタ。
// スコープを抜けると自動で delete を呼ぶ。
// ================================================================
static void pattern3_unique_ptr() {
    std::cout << "\n[Pattern 3] std::unique_ptr<TrackdloState>\n";

    const int M = 8;
    const int N = 150;

    // 未初期化: nullptr
    std::unique_ptr<trackdlo::TrackdloState> state = nullptr;

    trackdlo::TrackdloParams params;
    params.max_iter = 150;
    params.tol      = 1e-4;

    std::vector<int> all(M);
    std::iota(all.begin(), all.end(), 0);

    Eigen::MatrixXd X0 = make_cloud(N, 0.0, 0);

    // make_unique: new TrackdloState(...) と同等だが、
    // 確保と unique_ptr への代入が分離しないため例外安全。
    state = std::make_unique<trackdlo::TrackdloState>(make_state(X0, M));
    std::cout << "  initialized\n";

    for (int f = 1; f <= 3; f++) {
        Eigen::MatrixXd X = make_cloud(N, 0.05 * f, f);
        // unique_ptr には -> でメンバにアクセスする (生ポインタと同じ記法)
        trackdlo::tracking_step(*state, X, all, all, params);
        std::cout << "  frame " << f
                  << ": Y[0]=" << state->Y.row(0) << "\n";
    }

    // reset(): 内部で delete を呼び、ポインタを nullptr にする
    state.reset();
    std::cout << "  reset: unique_ptr = nullptr\n";

    // ← スコープを抜けても state は既に nullptr なので二重解放しない
}

// ================================================================
// Pattern 4: 生ポインタ (やってはいけない例)
//
// 以下のコードは意図的にメモリリークを起こす。
// 実際のコードには書かないこと。
// ================================================================
static void pattern4_raw_pointer_BAD() {
    std::cout << "\n[Pattern 4] 生ポインタ (BAD: メモリリークの例)\n";

    const int M = 8;
    const int N = 150;

    Eigen::MatrixXd X0 = make_cloud(N, 0.0, 0);

    // new で確保 → ヒープに置かれる
    trackdlo::TrackdloState* state = new trackdlo::TrackdloState(make_state(X0, M));

    trackdlo::TrackdloParams params;
    params.max_iter = 150;
    params.tol      = 1e-4;

    std::vector<int> all(M);
    std::iota(all.begin(), all.end(), 0);

    trackdlo::tracking_step(*state, make_cloud(N, 0.05, 1), all, all, params);
    std::cout << "  Y[0]=" << state->Y.row(0) << "\n";

    // ここで例外が発生したり return が増えたりすると delete が呼ばれない。
    // delete state;   ← 書き忘れると TrackdloState がリークする

    // 正しくは:
    delete state;
    state = nullptr;  // ダングリングポインタを防ぐために nullptr に戻す
    std::cout << "  delete 済み (正しく書けばリークしない)\n";
    std::cout << "  → ただし unique_ptr を使えばこのリスク自体がなくなる\n";
}

// ================================================================
// main
// ================================================================
int main() {
    std::cout << "=== state_lifecycle ===\n";
    std::cout << "pure_trackdlo のメモリ管理パターン比較\n";

    pattern1_stack();
    pattern2_optional();
    pattern3_unique_ptr();
    pattern4_raw_pointer_BAD();

    std::cout << "\ndone.\n";
    return 0;
}
