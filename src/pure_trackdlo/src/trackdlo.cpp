#include "../include/utils.h"
#include "../include/trackdlo.h"
#include <iostream>
#include <map>
#include <algorithm>
#include <cmath>

// ============================================================
// [メモ] ヘルパー関数を1つだけ残している理由
//
// ヘルパー関数とは:
//   ある公開関数の処理の一部を切り出して、コードを読みやすくするために作った補助的な関数。
//   外部には公開せず、必ず親となる関数から呼ばれる。
//   単体では意味をなさない。
//
// traverse_euclidean は tracking_step から alignment=0/1/2 の組み合わせで
// 計7回呼ばれる。オクルージョン状態ごとに呼び方が変わるため、
// 関数として切り出しておかないと tracking_step が読めなくなる。
//
// 他のヘルパー (get_nearest_indices, calc_LLE_weights) は
// それぞれ1か所からしか呼ばれなかったため、呼び出し元に直接インライン化した。
// traverse_geodesic は呼び出し箇所がゼロだったため削除した。
// ============================================================

namespace trackdlo {

namespace {  // anonymous

// ============================================================
// traverse_euclidean — Pure Pursuit によるオクルージョン補間
//
// ─── 何をする関数か ──────────────────────────────────────────
//
// ケーブルの一部が障害物で隠れている (オクルージョン) とき、
// 隠れた区間のノードは点群が存在しないため位置が分からない。
//
// この関数は「前フレームで見えていた形 (guide_nodes) をなぞって、
// 隠れた区間のノードがおそらくこのあたりにある」という推定座標を作る。
//
// ─── アルゴリズム: Pure Pursuit ─────────────────────────────
//
// Pure Pursuit は自動運転のパス追従制御で使われる手法。
// 「現在地からケーブルの長さ 1 セグメント分だけ先の点を
//  guide_nodes 上で探し続ける」という操作を繰り返す。
// 具体的には、現在地を中心とした球と guide_nodes の線分の
// 交点 (line_sphere_intersection) を求めることで「1 セグメント先」を得る。
//
// ─── 引数まとめ ──────────────────────────────────────────────
//
//   geodesic_coord       測地線座標 (geodesic coordinate) = 曲線上の弧長パラメータ = 各ノードまでの累積経路長 (単位 m)
//                        要素数はノード数 M と同じ。
//                        geodesic_coord[0]=0 で始まり、geodesic_coord[i] はノード 0〜i を
//                        結ぶ折れ線の長さ。隣接ノード間の「1 セグメント長」=
//                        geodesic_coord[i+1] - geodesic_coord[i]
//                        Pure Pursuit の look_ahead_dist として使う
//
//   guide_nodes          前フレームの可視ノード座標 (M×3)
//                        Pure Pursuit がなぞる「パス」になる
//                        tracking_step が毎フレーム更新して渡す
//
//   visible_nodes        現フレームで可視と判定されたノードのインデックス列
//                        (compute_visible_nodes が返す visible_nodes_extended を渡す)
//                        どの区間が可視でどの区間が隠れているかを示す
//
//   alignment            伸ばす方向を指定する
//                          0 → head (index 0) 側から tail 方向へ伸ばす
//                          1 → tail (最終index) 側から head 方向へ伸ばす
//                          2 → 両端オクルージョン。alignment_node_idx を起点に tail 方向へ伸ばす
//
//   alignment_node_idx   alignment==2 のときのみ使用
//                        両端オクルージョン時の起点ノードのインデックス
//                        デフォルト -1 (alignment==0/1 では参照しない)
//
// ─── 返値 ────────────────────────────────────────────────────
//
//   std::vector<Eigen::MatrixXd>
//     各要素は (1×4) 行列 [node_index, x, y, z]
//     node_index: 推定座標を割り当てるノードの番号
//     x, y, z:    そのノードの推定 3D 座標 (単位 m)
//     → TrackdloState::correspondence_priors として cpd_lle() に渡す
//
// ─── 返値の使われ方 ──────────────────────────────────────────
//
// 返値 node_pairs は [[ノード番号, x, y, z], ...] の形で、
// correspondence_priors として cpd_lle() に渡される。
// cpd_lle はこの推定座標を「このノードはここにあるはずだ」という
// 拘束として使い、隠れた区間のノードが飛んでいかないように抑える。
//
// ============================================================
std::vector<Eigen::MatrixXd> traverse_euclidean(std::vector<double> geodesic_coord,
                                                 const Eigen::MatrixXd guide_nodes,
                                                 const std::vector<int> visible_nodes,
                                                 int alignment,
                                                 int alignment_node_idx = -1)
{
    std::vector<Eigen::MatrixXd> node_pairs = {};

    // ガイドノードが1個しかない極端なケース: そのノードをそのまま返す
    if (guide_nodes.rows() == 1) {
        Eigen::MatrixXd node_pair(1, 4);
        node_pair << visible_nodes[0], guide_nodes(0, 0), guide_nodes(0, 1), guide_nodes(0, 2);
        node_pairs.push_back(node_pair);
        return node_pairs;
    }

    if (alignment == 0) {
        // head ノードを最初の対応点として登録する
        Eigen::MatrixXd node_pair(1, 4);
        node_pair << visible_nodes[0], guide_nodes(0, 0), guide_nodes(0, 1), guide_nodes(0, 2);
        node_pairs.push_back(node_pair);

        // 先頭から連続している可視ノードを取り出す (i==visible_nodes[i] である間)
        std::vector<int> consecutive_visible_nodes = {};
        for (int i = 0; i < static_cast<int>(visible_nodes.size()); i++) {
            if (i == visible_nodes[i]) {
                consecutive_visible_nodes.push_back(i);
            }
            else {
                break;
            }
        }

        int last_found_index = 0;
        int seg_dist_it = 0;
        Eigen::MatrixXd cur_center = guide_nodes.row(0);

        // Pure Pursuit: look_ahead_dist (次セグメント長) 先の guide_nodes 上の交点を求め続ける
        while (last_found_index+1 <= static_cast<int>(consecutive_visible_nodes.size())-1 &&
               seg_dist_it+1 <= static_cast<int>(geodesic_coord.size())-1)
        {
            double look_ahead_dist = fabs(geodesic_coord[seg_dist_it+1] - geodesic_coord[seg_dist_it]);
            bool found_intersection = false;
            std::vector<double> intersection = {};

            for (int i = last_found_index; i+1 <= static_cast<int>(consecutive_visible_nodes.size())-1; i++) {
                std::vector<Eigen::MatrixXd> intersections = line_sphere_intersection(guide_nodes.row(i), guide_nodes.row(i+1), cur_center, look_ahead_dist);
                if (intersections.size() == 0) {
                    continue;
                }
                else if (intersections.size() == 1 && pt2pt_dis(intersections[0], guide_nodes.row(i+1)) > pt2pt_dis(cur_center, guide_nodes.row(i+1))) {
                    continue;
                }
                else {
                    found_intersection = true;
                    last_found_index = i;
                    // 交点が2個ある場合は次のガイドノードに近い方を採用する
                    if (intersections.size() == 2) {
                        if (pt2pt_dis(intersections[0], guide_nodes.row(i+1)) <= pt2pt_dis(intersections[1], guide_nodes.row(i+1))) {
                            intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                            cur_center = intersections[0];
                        }
                        else {
                            intersection = {intersections[1](0, 0), intersections[1](0, 1), intersections[1](0, 2)};
                            cur_center = intersections[1];
                        }
                    }
                    else {
                        intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                        cur_center = intersections[0];
                    }
                    break;
                }
            }

            if (!found_intersection) break;

            Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(1, 4);
            temp(0, 0) = seg_dist_it + 1;
            temp(0, 1) = intersection[0];
            temp(0, 2) = intersection[1];
            temp(0, 3) = intersection[2];
            node_pairs.push_back(temp);
            seg_dist_it += 1;
        }
    }
    else if (alignment == 1) {
        // tail ノードを最初の対応点として登録する
        Eigen::MatrixXd node_pair(1, 4);
        node_pair << visible_nodes.back(), guide_nodes(guide_nodes.rows()-1, 0), guide_nodes(guide_nodes.rows()-1, 1), guide_nodes(guide_nodes.rows()-1, 2);
        node_pairs.push_back(node_pair);

        // 末尾から連続している可視ノードを取り出す
        std::vector<int> consecutive_visible_nodes = {};
        for (int i = 1; i <= static_cast<int>(visible_nodes.size()); i++) {
            if (visible_nodes[visible_nodes.size()-i] == static_cast<int>(geodesic_coord.size())-i) {
                consecutive_visible_nodes.push_back(geodesic_coord.size()-i);
            }
            else {
                break;
            }
        }

        int last_found_index = guide_nodes.rows()-1;
        int seg_dist_it = geodesic_coord.size()-1;
        Eigen::MatrixXd cur_center = guide_nodes.row(guide_nodes.rows()-1);

        // alignment==0 と同じ Pure Pursuit を tail から head 方向へ実行する
        while (last_found_index-1 >= static_cast<int>(guide_nodes.rows() - consecutive_visible_nodes.size()) &&
               seg_dist_it-1 >= 0)
        {
            double look_ahead_dist = fabs(geodesic_coord[seg_dist_it] - geodesic_coord[seg_dist_it-1]);
            bool found_intersection = false;
            std::vector<double> intersection = {};

            for (int i = last_found_index; i >= static_cast<int>(guide_nodes.rows() - consecutive_visible_nodes.size() + 1); i--) {
                std::vector<Eigen::MatrixXd> intersections = line_sphere_intersection(guide_nodes.row(i), guide_nodes.row(i-1), cur_center, look_ahead_dist);
                if (intersections.size() == 0) {
                    continue;
                }
                else if (intersections.size() == 1 && pt2pt_dis(intersections[0], guide_nodes.row(i-1)) > pt2pt_dis(cur_center, guide_nodes.row(i-1))) {
                    continue;
                }
                else {
                    found_intersection = true;
                    last_found_index = i;
                    if (intersections.size() == 2) {
                        if (pt2pt_dis(intersections[0], guide_nodes.row(i-1)) <= pt2pt_dis(intersections[1], guide_nodes.row(i-1))) {
                            intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                            cur_center = intersections[0];
                        }
                        else {
                            intersection = {intersections[1](0, 0), intersections[1](0, 1), intersections[1](0, 2)};
                            cur_center = intersections[1];
                        }
                    }
                    else {
                        intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                        cur_center = intersections[0];
                    }
                    break;
                }
            }

            if (!found_intersection) break;

            Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(1, 4);
            temp(0, 0) = seg_dist_it - 1;
            temp(0, 1) = intersection[0];
            temp(0, 2) = intersection[1];
            temp(0, 3) = intersection[2];
            node_pairs.push_back(temp);
            seg_dist_it -= 1;
        }
    }
    else {
        // alignment == 2: 両端オクルージョン。alignment_node_idx を起点に tail 方向へ伸ばす
        Eigen::MatrixXd node_pair(1, 4);
        node_pair << visible_nodes[alignment_node_idx], guide_nodes(alignment_node_idx, 0), guide_nodes(alignment_node_idx, 1), guide_nodes(alignment_node_idx, 2);
        node_pairs.push_back(node_pair);

        // alignment_node_idx から連続している可視ノードを取り出す
        std::vector<int> consecutive_visible_nodes_2 = {visible_nodes[alignment_node_idx]};
        for (int i = alignment_node_idx+1; i < static_cast<int>(visible_nodes.size()); i++) {
            if (visible_nodes[i] - visible_nodes[i-1] == 1) {
                consecutive_visible_nodes_2.push_back(visible_nodes[i]);
            }
            else {
                break;
            }
        }

        int last_found_index = alignment_node_idx;
        int seg_dist_it = visible_nodes[alignment_node_idx];
        Eigen::MatrixXd cur_center = guide_nodes.row(alignment_node_idx);

        while (last_found_index+1 <= alignment_node_idx + static_cast<int>(consecutive_visible_nodes_2.size())-1 &&
               seg_dist_it+1 <= static_cast<int>(geodesic_coord.size())-1)
        {
            double look_ahead_dist = fabs(geodesic_coord[seg_dist_it+1] - geodesic_coord[seg_dist_it]);
            bool found_intersection = false;
            std::vector<double> intersection = {};

            for (int i = last_found_index; i+1 <= alignment_node_idx + static_cast<int>(consecutive_visible_nodes_2.size())-1; i++) {
                std::vector<Eigen::MatrixXd> intersections = line_sphere_intersection(guide_nodes.row(i), guide_nodes.row(i+1), cur_center, look_ahead_dist);
                if (intersections.size() == 0) {
                    continue;
                }
                else if (intersections.size() == 1 && pt2pt_dis(intersections[0], guide_nodes.row(i+1)) > pt2pt_dis(cur_center, guide_nodes.row(i+1))) {
                    continue;
                }
                else {
                    found_intersection = true;
                    last_found_index = i;
                    if (intersections.size() == 2) {
                        if (pt2pt_dis(intersections[0], guide_nodes.row(i+1)) <= pt2pt_dis(intersections[1], guide_nodes.row(i+1))) {
                            intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                            cur_center = intersections[0];
                        }
                        else {
                            intersection = {intersections[1](0, 0), intersections[1](0, 1), intersections[1](0, 2)};
                            cur_center = intersections[1];
                        }
                    }
                    else {
                        intersection = {intersections[0](0, 0), intersections[0](0, 1), intersections[0](0, 2)};
                        cur_center = intersections[0];
                    }
                    break;
                }
            }

            if (!found_intersection) break;

            Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(1, 4);
            temp(0, 0) = seg_dist_it + 1;
            temp(0, 1) = intersection[0];
            temp(0, 2) = intersection[1];
            temp(0, 3) = intersection[2];
            node_pairs.push_back(temp);
            seg_dist_it += 1;
        }
    }

    return node_pairs;
}

} // namespace (anonymous)


// ============================================================
// 公開 API
// ============================================================

// ============================================================
// make_trackdlo_state — 初期状態を生成するファクトリ関数
//
// ─── 何をする関数か ──────────────────────────────────────────
//
// TrackdloState は「フレーム間で引き継ぐ可変状態」を持つ構造体。
// この関数はその初期値 (Y = 零行列, sigma2 = 0) を持つインスタンスを返す。
//
// Y の初期化は呼び出し側の責任。
// 一般的には最初のフレームで sort_pts() / reg() を使って
// 入力点群から等間隔に num_nodes 個のノードを抽出して Y に代入する。
//
// ─── 引数まとめ ──────────────────────────────────────────────
//
//   num_nodes   ノード数 M。セッション全体で変化しない定数。
//               典型値は 10〜20 程度。ケーブルの全長を基に決める。
//
// ─── 返値 ────────────────────────────────────────────────────
//
//   TrackdloState   Y が (M×3) 零行列に初期化された状態オブジェクト
//
// ============================================================
TrackdloState make_trackdlo_state(int num_nodes) {
    TrackdloState state;
    state.Y = Eigen::MatrixXd::Zero(num_nodes, 3);
    state.guide_nodes = state.Y;
    state.sigma2 = 0.0;
    return state;
}

// ============================================================
// cpd_lle — CPD + LLE によるノード位置合わせ (アルゴリズム中核)
//
// ─── 何をする関数か ──────────────────────────────────────────
//
// trackdlo の最も重要な関数。入力点群 X に対してノード配列 Y を
// 「なめらかに動かして合わせる」EM (期待値最大化) 最適化アルゴリズム。
//
// 呼び出すたびに Y と sigma2 が in-place で更新される。
// 呼び出し元は更新後の Y を state.Y として次フレームに引き継ぐ。
//
// ─── アルゴリズム概要: CPD + LLE ────────────────────────────
//
// ■ CPD (Coherent Point Drift) の基本アイデア
//   各ノード Y[m] を「ガウシアン混合モデル (GMM) の中心」として扱う。
//   点群 X の各点がどのノードに属するかを確率 P で表し、
//   X の尤度が最大になるようにノードを動かす。
//   sigma2 は GMM の分散で、収束が進むにつれて小さくなっていく。
//
// ■ LLE (Locally Linear Embedding) 正則化
//   「各ノードは近傍ノードの線形結合で表せる」という仮定。
//   これにより隣り合うノードが極端に離れないよう形状が平滑化される。
//   行列 H = (I-L)^T (I-L) として目的関数に加える。
//
// ■ 測地線距離ベースの対応確率
//   単純なユークリッド距離ではなく、ケーブル上の弧長距離 (測地線距離) で
//   対応確率を計算するよう改良されている。
//   これにより「ケーブルが折れ曲がっている場合に遠い側のノードに
//   誤対応する問題」を防ぐ。
//
// ■ EM ループ (max_iter 回まで繰り返し)
//   Eステップ: 各点とノードの対応確率行列 P (M×N) を更新する
//   Mステップ: P を固定して W (変形パラメータ) を解く線形方程式
//              A*W = B を解き、T = Y_0 + G*W でノード位置を更新する
//   収束判定: ノード移動量の平均 < tol になったら早期終了
//
// ─── 引数まとめ ──────────────────────────────────────────────
//
//   X_orig      入力点群 (N×3)。各フレームで変わる変数。
//               内部で各ノードから 0.1m 以内の点だけに絞り込む。
//
//   Y           ノード座標 (M×3)。in-place で更新される。
//               呼び出し前は前フレームの結果を入れておく。
//
//   sigma2      GMM の分散。in-place で更新される。
//               0 を渡すと初回自動初期化する。フレーム間で引き継ぐ。
//
//   beta        ガウシアンカーネル G の帯域幅 (m 単位)。
//               大きいほど遠くのノード同士が連動して動く (滑らかになる)。
//               典型値: 0.35
//
//   lambda      CPD 正則化強度。大きいほど変形が小さく抑えられる。
//               典型値: 50000
//
//   lle_weight  LLE 形状正則化の重み。大きいほど直線状を保とうとする。
//               典型値: 10
//
//   mu          外れ値 (ノイズ点) の割合 [0,1]。
//               大きいほど外れ値への耐性が上がるが感度が下がる。
//               典型値: 0.1
//
//   max_iter    EM ループの最大反復回数。
//               典型値: 50
//
//   tol         収束判定閾値 (ノード移動量の平均)。
//               典型値: 0.0002
//
//   include_lle LLE 正則化を使うか否かのフラグ。
//               tracking_step から呼ぶ本番パスは false
//               (correspondence_priors で制約するため LLE は不要)。
//               前処理パスは true。
//
//   correspondence_priors
//               traverse_euclidean が生成したオクルージョン補間結果。
//               各要素は (1×4) 行列 [node_index, x, y, z]。
//               このノードはこの座標にいるはず、という拘束として使う。
//               可視ノードのみの場合は空ベクトル {} を渡す。
//
//   alpha       correspondence_priors の拘束強度。
//               大きいほど traverse_euclidean の推定に強く従う。
//               典型値: 3.0
//
//   visible_nodes
//               現フレームで可視と判定されたノードのインデックス列。
//               空なら全ノード可視として扱う。
//
//   k_vis       可視確率の減衰係数。大きいほど非可視ノードの影響が
//               強く抑制される。典型値: 50
//
//   visibility_threshold
//               ノードが「可視」と見なす最大距離 (m)。
//               ノード ↔ 最近傍点の距離がこれ以下なら P_vis = 1。
//               典型値: 0.008
//
// ─── 返値 ────────────────────────────────────────────────────
//
//   bool   true = max_iter 内に収束した
//          false = max_iter 到達で打ち切り (非収束)
//   Y と sigma2 は引数として in-place 更新される (返値ではない)
//
// ============================================================
bool cpd_lle(const Eigen::MatrixXd& X_orig,
             Eigen::MatrixXd& out_Y,
             double& out_sigma2,
             double beta,
             double lambda,
             double lle_weight,
             double mu,
             int max_iter,
             double tol,
             bool include_lle,
             std::vector<Eigen::MatrixXd> correspondence_priors,
             double alpha,
             std::vector<int> visible_nodes,
             double k_vis,
             double visibility_threshold)
{
    // 各ノードから0.1m以内の点のみを使用して計算量を削減する
    Eigen::MatrixXd X_temp = Eigen::MatrixXd::Zero(X_orig.rows(), 3);
    int valid_pt_counter = 0;
    for (int i = 0; i < X_orig.rows(); i++) {
        double shortest_dist = 100000;
        for (int j = 0; j < out_Y.rows(); j++) {
            double dist = (out_Y.row(j) - X_orig.row(i)).norm();
            if (dist < shortest_dist) shortest_dist = dist;
        }
        if (shortest_dist < 0.1) {
            X_temp.row(valid_pt_counter) = X_orig.row(i);
            valid_pt_counter += 1;
        }
    }
    Eigen::MatrixXd X = X_temp.topRows(valid_pt_counter);

    bool converged = true;

    int M = out_Y.rows();
    int N = X.rows();
    int D = 3;

    Eigen::MatrixXd Y_0 = out_Y.replicate(1, 1);

    // 隣接ノード間の測地線座標 (累積弧長) を計算する
    // CPDのガウシアンカーネルGは測地線距離ベースで計算するため必要
    Eigen::MatrixXd converted_node_dis    = Eigen::MatrixXd::Zero(M, M);
    Eigen::MatrixXd converted_node_dis_sq = Eigen::MatrixXd::Zero(M, M);
    std::vector<double> converted_node_coord = {0.0};
    double cur_sum = 0;
    for (int i = 0; i < M-1; i++) {
        cur_sum += pt2pt_dis(Y_0.row(i+1), Y_0.row(i));
        converted_node_coord.push_back(cur_sum);
    }
    for (int i = 0; i < static_cast<int>(converted_node_coord.size()); i++) {
        for (int j = 0; j < static_cast<int>(converted_node_coord.size()); j++) {
            converted_node_dis_sq(i, j) = pow(converted_node_coord[i] - converted_node_coord[j], 2);
            converted_node_dis(i, j)    = abs(converted_node_coord[i] - converted_node_coord[j]);
        }
    }

    // ガウシアンカーネル行列G: 測地線距離ベースのMatern 3/2カーネル
    Eigen::MatrixXd G = 1/(2*beta * 2*beta) *
        (-sqrt(2)*converted_node_dis/beta).array().exp() *
        (2*converted_node_dis.array() + sqrt(2)*beta);

    // ---- LLE重み行列 L を計算する (M×M) --------------------------------
    // 各ノードを隣接 k=6 ノードの線形結合で近似するときの係数行列。
    // H = (I-L)^T (I-L) が形状正則化項になる。
    // [注意] k/2=3 近傍を使うため M > 6 が必要。
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(M, M);
    for (int i = 0; i < M; i++) {
        // i の k/2=3 近傍インデックスを求める (鎖端では片側のみ)
        // [注意] M <= 6 のとき i + half_k が M を超えうる。
        // std::min で M-1 にクランプして out-of-bounds を防ぐ。
        const int half_k = 3;
        std::vector<int> indices;
        if (i - half_k < 0) {
            for (int j = 0; j <= std::min(i + half_k, M - 1); j++) {
                if (j != i) indices.push_back(j);
            }
        }
        else if (i + half_k >= M) {
            for (int j = i - half_k; j <= M - 1; j++) {
                if (j != i) indices.push_back(j);
            }
        }
        else {
            for (int j = i - half_k; j <= i + half_k; j++) {
                if (j != i) indices.push_back(j);
            }
        }

        Eigen::MatrixXd xi = Y_0.row(i);
        Eigen::MatrixXd Xi(indices.size(), Y_0.cols());
        for (int r = 0; r < static_cast<int>(indices.size()); r++) {
            Xi.row(r) = Y_0.row(indices[r]);
        }

        // Gi = (xi - Xi)^T (xi - Xi): 近傍との差分の共分散行列
        Eigen::MatrixXd component = xi.replicate(Xi.rows(), 1).transpose() - Xi.transpose();
        Eigen::MatrixXd Gi = component.transpose() * component;
        Eigen::MatrixXd Gi_inv;
        if (Gi.determinant() != 0) {
            Gi_inv = Gi.inverse();
        }
        else {
            // Gi が特異行列になることがある → 対角に微小値を足して正則化
            Gi.diagonal().array() += 1e-5;
            Gi_inv = Gi.inverse();
        }

        // wi = Gi_inv * 1 / (1^T * Gi_inv * 1): 正規化された重みベクトル
        Eigen::MatrixXd ones_col = Eigen::MatrixXd::Constant(Xi.rows(), 1, 1.0);
        Eigen::MatrixXd ones_row = Eigen::MatrixXd::Constant(1, Xi.rows(), 1.0);
        Eigen::MatrixXd wi = (Gi_inv * ones_col) / (ones_row * Gi_inv * ones_col).value();

        for (int c = 0; c < static_cast<int>(indices.size()); c++) {
            L(i, indices[c]) = wi(c);
        }
    }
    Eigen::MatrixXd H = (Eigen::MatrixXd::Identity(M, M) - L).transpose() *
                        (Eigen::MatrixXd::Identity(M, M) - L);
    // ---------------------------------------------------------------------

    // correspondence_priors から J 行列と Y_extended を構築する
    // J: prior が存在するノードの行だけ 1 になる単位行列のサブセット
    // Y_extended: prior 座標で Y_0 の対応行を上書きしたもの
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(M, M);
    Eigen::MatrixXd Y_extended = Y_0.replicate(1, 1);
    if (correspondence_priors.size() != 0) {
        for (int i = 0; i < static_cast<int>(correspondence_priors.size()); i++) {
            int index = correspondence_priors[i](0, 0);
            Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(1, 3);
            temp(0, 0) = correspondence_priors[i](0, 1);
            temp(0, 1) = correspondence_priors[i](0, 2);
            temp(0, 2) = correspondence_priors[i](0, 3);
            J.row(index) = Eigen::MatrixXd::Identity(M, M).row(index);
            Y_extended.row(index) = temp;
        }
    }

    // 各ノードと各点間の二乗距離行列を初期化する
    Eigen::MatrixXd diff_xy = Eigen::MatrixXd::Zero(M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            diff_xy(i, j) = (Y_0.row(i) - X.row(j)).squaredNorm();
        }
    }

    if (out_sigma2 == 0) {
        out_sigma2 = diff_xy.sum() / static_cast<double>(D * M * N);
    }

    for (int it = 0; it < max_iter; it++) {

        // 各ノードの最近傍点群点への距離を更新する (P_vis 計算に使用)
        std::map<int, double> shortest_node_pt_dists;
        for (int m = 0; m < M; m++) {
            double shortest_dist = 10000;
            for (int n = 0; n < N; n++) {
                diff_xy(m, n) = (out_Y.row(m) - X.row(n)).squaredNorm();
                double dist = (out_Y.row(m) - X.row(n)).norm();
                if (dist < shortest_dist) shortest_dist = dist;
            }
            if (shortest_dist <= visibility_threshold) shortest_dist = 0;
            shortest_node_pt_dists.insert(std::pair<int, double>(m, shortest_dist));
        }

        // E step: 対応確率行列 P を計算する
        Eigen::MatrixXd P = (-0.5 * diff_xy / out_sigma2).array().exp();
        double c = pow((2 * M_PI * out_sigma2), static_cast<double>(D)/2) * mu / (1 - mu) * static_cast<double>(M)/N;
        P = P.array().rowwise() / (P.colwise().sum().array() + c);

        // 測地線距離ベースの P に更新する
        // 各点について最も対応確率が高いノードとその隣を特定し、
        // 残りのノードへの距離は測地線距離で補完する
        std::vector<int> max_p_nodes(P.cols(), 0);
        Eigen::MatrixXd pts_dis_sq_geodesic = Eigen::MatrixXd::Zero(M, N);

        for (int i = 0; i < N; i++) {
            P.col(i).maxCoeff(&max_p_nodes[i]);
            int max_p_node = max_p_nodes[i];

            int potential_2nd_1 = max_p_node - 1;
            // M < 3 のとき "2" が範囲外になるため std::min で上限を M-1 にクランプする
            if (potential_2nd_1 < 0) potential_2nd_1 = std::min(2, M - 1);
            int potential_2nd_2 = max_p_node + 1;
            // M < 4 のとき M-3 が負になるため std::max で 0 にクランプする
            if (potential_2nd_2 >= M) potential_2nd_2 = std::max(M - 3, 0);

            int next_max_p_node = (pt2pt_dis(out_Y.row(potential_2nd_1), X.row(i)) < pt2pt_dis(out_Y.row(potential_2nd_2), X.row(i)))
                                  ? potential_2nd_1 : potential_2nd_2;

            pts_dis_sq_geodesic(max_p_node, i)  = pt2pt_dis_sq(out_Y.row(max_p_node), X.row(i));
            pts_dis_sq_geodesic(next_max_p_node, i) = pt2pt_dis_sq(out_Y.row(next_max_p_node), X.row(i));

            if (max_p_node < next_max_p_node) {
                for (int j = 0; j < max_p_node; j++) {
                    pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[max_p_node]) + pt2pt_dis(out_Y.row(max_p_node), X.row(i)), 2);
                }
                for (int j = next_max_p_node; j < M; j++) {
                    pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[next_max_p_node]) + pt2pt_dis(out_Y.row(next_max_p_node), X.row(i)), 2);
                }
            }
            else {
                for (int j = 0; j < next_max_p_node; j++) {
                    pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[next_max_p_node]) + pt2pt_dis(out_Y.row(next_max_p_node), X.row(i)), 2);
                }
                for (int j = max_p_node; j < M; j++) {
                    pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[max_p_node]) + pt2pt_dis(out_Y.row(max_p_node), X.row(i)), 2);
                }
            }
        }

        P = (-0.5 * pts_dis_sq_geodesic / out_sigma2).array().exp();

        // 可視確率 P_vis で P を重み付けする (一部ノードがオクルージョン下にある場合)
        if (visible_nodes.size() != static_cast<size_t>(out_Y.rows()) && !visible_nodes.empty() && k_vis != 0) {
            Eigen::MatrixXd P_vis = Eigen::MatrixXd::Ones(P.rows(), P.cols());
            double total_P_vis = 0;
            for (int i = 0; i < out_Y.rows(); i++) {
                double P_vis_i = exp(-k_vis * shortest_node_pt_dists[i]);
                total_P_vis += P_vis_i;
                P_vis.row(i) = P_vis_i * P_vis.row(i);
            }
            P_vis = P_vis / total_P_vis;
            P = P.cwiseProduct(P_vis);
            c = pow((2 * M_PI * out_sigma2), static_cast<double>(D)/2) * mu / (1 - mu) / N;
            P = P.array().rowwise() / (P.colwise().sum().array() + c);
        }
        else {
            P = P.array().rowwise() / (P.colwise().sum().array() + c);
        }

        Eigen::MatrixXd Pt1 = P.colwise().sum();
        Eigen::MatrixXd P1  = P.rowwise().sum();
        double Np = P1.sum();
        Eigen::MatrixXd PX = P * X;

        // M step: W を解いて新しいノード位置 T = Y_0 + G*W を得る
        Eigen::MatrixXd A_matrix;
        Eigen::MatrixXd B_matrix;
        if (include_lle) {
            if (correspondence_priors.size() != 0) {
                A_matrix = P1.asDiagonal()*G + lambda*out_sigma2 * Eigen::MatrixXd::Identity(M, M) + out_sigma2*lle_weight * H*G + alpha*J*G;
                B_matrix = PX - P1.asDiagonal()*Y_0 - out_sigma2*lle_weight * H*Y_0 + alpha*(Y_extended - Y_0);
            }
            else {
                A_matrix = P1.asDiagonal()*G + lambda*out_sigma2 * Eigen::MatrixXd::Identity(M, M) + out_sigma2*lle_weight * H*G;
                B_matrix = PX - P1.asDiagonal()*Y_0 - out_sigma2*lle_weight * H*Y_0;
            }
        }
        else {
            if (correspondence_priors.size() != 0) {
                A_matrix = P1.asDiagonal() * G + lambda * out_sigma2 * Eigen::MatrixXd::Identity(M, M) + alpha*J*G;
                B_matrix = PX - P1.asDiagonal() * Y_0 + alpha*(Y_extended - Y_0);
            }
            else {
                A_matrix = P1.asDiagonal() * G + lambda * out_sigma2 * Eigen::MatrixXd::Identity(M, M);
                B_matrix = PX - P1.asDiagonal() * Y_0;
            }
        }

        Eigen::MatrixXd W = A_matrix.completeOrthogonalDecomposition().solve(B_matrix);
        Eigen::MatrixXd T = Y_0 + G * W;

        double trXtdPt1X = (X.transpose() * Pt1.asDiagonal() * X).trace();
        double trPXtT    = (PX.transpose() * T).trace();
        double trTtdP1T  = (T.transpose() * P1.asDiagonal() * T).trace();
        out_sigma2 = (trXtdPt1X - 2*trPXtT + trTtdP1T) / (Np * D);
        // Np=0 などで out_sigma2 が NaN/負 になると out_Y が NaN に汚染されて
        // 後続の traverse_euclidean でヒープ破壊が起きる。
        // 最小値にクランプして NaN 伝播を防ぐ。
        if (std::isnan(out_sigma2) || out_sigma2 <= 0) out_sigma2 = 1e-8;

        if (pt2pt_dis(out_Y, Y_0 + G*W) / out_Y.rows() < tol) {
            out_Y = Y_0 + G*W;
            std::cout << "Iteration until convergence: " << (it+1) << std::endl;
            break;
        }
        else {
            out_Y = Y_0 + G*W;
        }

        if (it == max_iter - 1) {
            std::cerr << "optimization did not converge!" << std::endl;
            converged = false;
            break;
        }
    }

    return converged;
}

// ============================================================
// tracking_step — 1フレーム分のトラッキング処理オーケストレーター
//
// ─── 何をする関数か ──────────────────────────────────────────
//
// 1フレーム (1ステップ) ごとに呼ぶ関数。
// ROS2 コールバックからでも while/for ループからでも呼ぶ関数。
// 前処理で生成された入力点群 X と可視ノードのインデックス列を受け取り、
// 内部で cpd_lle / traverse_euclidean を組み合わせて
// state.Y (ノード座標) を最終更新する。
//
// 内部では「前処理 cpd_lle → オクルージョン状態判定 →
// traverse_euclidean で correspondence_priors 構築 → 本番 cpd_lle」
// という順に処理を進める。各処理の詳細は Step 1〜4 を参照。
//
// ─── 処理フロー ──────────────────────────────────────────────
//
//  Step 1: 可視ノード座標を state.guide_nodes に抽出する
//          visible_nodes_extended に含まれるノードの座標だけを取り出す。
//          guide_nodes = 前フレームの可視ノード位置のスナップショット。
//          Pure Pursuit がなぞる「パス」になる。
//
//  Step 2: guide_nodes を簡易 CPD で粗く合わせる (前処理パス)
//          beta_pre_proc / lambda_pre_proc は本番とは別パラメータ。
//          include_lle=true で形状制約を強くかけて素早く収束させる。
//          sigma2 は一時変数 sigma2_pre_proc を使い state.sigma2 は変えない。
//
//  Step 3: オクルージョン状態を 5 分類して correspondence_priors を構築
//          分類:
//            全可視 / 小オクルージョン  → traverse_euclidean x2 (両側) + 平均
//            中間オクルージョン          → traverse_euclidean x2 (両側) + 結合
//            片端 (tail) オクルージョン  → traverse_euclidean x1 (alignment=0)
//            片端 (head) オクルージョン  → traverse_euclidean x1 (alignment=1)
//            両端オクルージョン          → traverse_euclidean x1 (alignment=2)
//
//  Step 4: cpd_lle で state.Y を最終更新
//          include_lle=false、correspondence_priors を拘束として渡す。
//          state.sigma2 も更新される。
//
// ─── 引数まとめ ──────────────────────────────────────────────
//
//   state                    in/out。Y・guide_nodes・sigma2・
//                            correspondence_priors を書き換える。
//                            geodesic_coord は読み取るだけ (呼び出し前に初期化必須)。
//
//   X                        入力点群 (N×3)。各フレームで変わる。
//                            preprocessing::images_to_pointcloud() の出力。
//
//   visible_nodes            可視ノードのインデックス列。
//                            compute_visible_nodes() の出力 (厳しい閾値版)。
//                            オクルージョン状態の分類に使う。
//
//   visible_nodes_extended   可視ノードのインデックス列 (拡張版)。
//                            compute_visible_nodes() の出力 (緩い閾値版)。
//                            guide_nodes 抽出と traverse_euclidean に使う。
//
//   params                   TrackdloParams。beta / lambda 等の定数。
//
// ─── 返値 ────────────────────────────────────────────────────
//
//   なし (void)。state が in-place で更新される。
//   呼び出し後に state.Y を読み取ると最新ノード座標が得られる。
//
//   [メモ] in-place 更新とは
//     新しいオブジェクトを返すのではなく、渡した変数自体を書き換えること。
//     C++ では引数に & (参照) をつけることで実現する。
//     「関数を呼んだら引数が変わっていた」という動作が in-place 更新。
//
// ============================================================
TrackdloState tracking_step(TrackdloState state,
                   const Eigen::MatrixXd& X,
                   const std::vector<int>& visible_nodes,
                   const std::vector<int>& visible_nodes_extended,
                   const TrackdloParams& params)
{
    state.correspondence_priors = {};

    // 可視ノードの座標を guide_nodes に抽出する
    state.guide_nodes = Eigen::MatrixXd::Zero(visible_nodes_extended.size(), 3);
    if (static_cast<int>(visible_nodes_extended.size()) != state.Y.rows()) {
        for (int i = 0; i < static_cast<int>(visible_nodes_extended.size()); i++) {
            state.guide_nodes.row(i) = state.Y.row(visible_nodes_extended[i]);
        }
    }
    else {
        state.guide_nodes = state.Y.replicate(1, 1);
    }

    // 前処理: guide_nodes を簡易CPD で粗く合わせる (sigma2 は一時変数で更新)
    // visible_nodes_extended が空のときは guide_nodes も 0×3 になるのでスキップする
    if (state.guide_nodes.rows() > 0) {
        double sigma2_pre_proc = state.sigma2;
        cpd_lle(X, state.guide_nodes, sigma2_pre_proc,  // out_Y=guide_nodes, out_sigma2=sigma2_pre_proc (in-place update)
                params.beta_pre_proc, params.lambda_pre_proc, params.lle_weight,
                params.mu, params.max_iter, params.tol, true);
    }

    // オクルージョン状態を判定して correspondence_priors を構築する
    // visible_nodes_extended が空の場合は [0] アクセスが未定義動作になるため先に弾く
    if (visible_nodes_extended.empty()) {
        // 可視ノード 0 個: correspondence_priors なし。cpd_lle が自力で点群に合わせる。
        state.correspondence_priors = {};
    }
    else if (static_cast<int>(visible_nodes_extended.size()) == state.Y.rows()) {
        if (visible_nodes.size() == visible_nodes_extended.size()) {
            std::cout << "All nodes visible" << std::endl;
        }
        else {
            std::cout << "Minor occlusion" << std::endl;
        }

        // head/tail 両側から Pure Pursuit して平均を取る
        std::vector<Eigen::MatrixXd> priors_vec_1 = traverse_euclidean(state.geodesic_coord, state.guide_nodes, visible_nodes_extended, 0);
        std::vector<Eigen::MatrixXd> priors_vec_2 = traverse_euclidean(state.geodesic_coord, state.guide_nodes, visible_nodes_extended, 1);
        std::reverse(priors_vec_2.begin(), priors_vec_2.end());

        // traverse_euclidean が M 個未満を返した場合 (sigma2 が一時的に NaN に
        // なった後のリカバリ途中など) は out-of-bounds になる。
        // priors が足りなければ correspondence_priors なし (cpd_lle が自力で合わせる)。
        state.correspondence_priors = {};
        if (priors_vec_1.size() == static_cast<size_t>(state.Y.rows()) &&
            priors_vec_2.size() == static_cast<size_t>(state.Y.rows()))
        {
            for (int i = 0; i < state.Y.rows(); i++) {
                int offset = i - (state.Y.rows() - static_cast<int>(priors_vec_2.size()));
                if (i < priors_vec_2[0](0, 0) && i < static_cast<int>(priors_vec_1.size())) {
                    state.correspondence_priors.push_back(priors_vec_1[i]);
                }
                else if (i > priors_vec_1[priors_vec_1.size()-1](0, 0) &&
                         offset >= 0 && offset < static_cast<int>(priors_vec_2.size()))
                {
                    state.correspondence_priors.push_back(priors_vec_2[offset]);
                }
                else if (i < static_cast<int>(priors_vec_1.size()) &&
                         offset >= 0 && offset < static_cast<int>(priors_vec_2.size()))
                {
                    state.correspondence_priors.push_back((priors_vec_1[i] + priors_vec_2[offset]) / 2.0);
                }
            }
        }
    }
    else if (visible_nodes_extended[0] == 0 &&
             visible_nodes_extended[visible_nodes_extended.size()-1] == state.Y.rows()-1)
    {
        std::cout << "Mid-section occluded" << std::endl;
        state.correspondence_priors = traverse_euclidean(state.geodesic_coord, state.guide_nodes, visible_nodes_extended, 0);
        std::vector<Eigen::MatrixXd> priors_vec_2 = traverse_euclidean(state.geodesic_coord, state.guide_nodes, visible_nodes_extended, 1);
        state.correspondence_priors.insert(state.correspondence_priors.end(), priors_vec_2.begin(), priors_vec_2.end());
    }
    else if (visible_nodes_extended[0] == 0) {
        std::cout << "Tail occluded" << std::endl;
        state.correspondence_priors = traverse_euclidean(state.geodesic_coord, state.guide_nodes, visible_nodes_extended, 0);
    }
    else if (visible_nodes_extended[visible_nodes_extended.size()-1] == state.Y.rows()-1) {
        std::cout << "Head occluded" << std::endl;
        state.correspondence_priors = traverse_euclidean(state.geodesic_coord, state.guide_nodes, visible_nodes_extended, 1);
    }
    else {
        std::cout << "Both ends occluded" << std::endl;

        // 最も動きが小さいノードを起点とする
        int alignment_node_idx = -1;
        double moved_dist = 999999;
        for (int i = 0; i < static_cast<int>(visible_nodes.size()); i++) {
            if (pt2pt_dis(state.Y.row(visible_nodes[i]), state.guide_nodes.row(i)) < moved_dist) {
                moved_dist = pt2pt_dis(state.Y.row(visible_nodes[i]), state.guide_nodes.row(i));
                alignment_node_idx = i;
            }
        }
        state.correspondence_priors = traverse_euclidean(state.geodesic_coord, state.guide_nodes, visible_nodes_extended, 2, alignment_node_idx);
    }

    cpd_lle(X, state.Y, state.sigma2,               // out_Y=state.Y, out_sigma2=state.sigma2 (in-place update)
            params.beta, params.lambda, params.lle_weight,
            params.mu, params.max_iter, params.tol,
            false,
            state.correspondence_priors, params.alpha,
            visible_nodes_extended, params.k_vis, params.visibility_threshold);

    return state;
}

} // namespace trackdlo
