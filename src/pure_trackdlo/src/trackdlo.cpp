#include "../include/utils.h"
#include "../include/trackdlo.h"
#include <iostream>
#include <map>
#include <algorithm>
#include <cmath>

// ============================================================
// [メモ] ヘルパー関数を1つだけ残している理由
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

namespace {  // このファイル内だけで使うファイルローカル関数

// Pure Pursuit によるオクルージョン補間
// guide_nodes の可視部分をたどり、geodesic_coord の間隔ごとに
// ノード座標を推定して node_pairs として返す。
//
// alignment の意味:
//   0 → head (index 0) 側から tail 方向へ伸ばす
//   1 → tail (最終index) 側から head 方向へ伸ばす
//   2 → 両端オクルージョン。alignment_node_idx を起点に tail 方向へ伸ばす
//
// 返値: [[node_index, x, y, z], ...] の行列リスト (correspondence_priors の形式)
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

TrackdloState make_trackdlo_state(int num_nodes) {
    TrackdloState state;
    state.Y = Eigen::MatrixXd::Zero(num_nodes, 3);
    state.guide_nodes = state.Y;
    state.sigma2 = 0.0;
    return state;
}

bool cpd_lle(const Eigen::MatrixXd& X_orig,
             Eigen::MatrixXd& Y,
             double& sigma2,
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
        for (int j = 0; j < Y.rows(); j++) {
            double dist = (Y.row(j) - X_orig.row(i)).norm();
            if (dist < shortest_dist) shortest_dist = dist;
        }
        if (shortest_dist < 0.1) {
            X_temp.row(valid_pt_counter) = X_orig.row(i);
            valid_pt_counter += 1;
        }
    }
    Eigen::MatrixXd X = X_temp.topRows(valid_pt_counter);

    bool converged = true;

    int M = Y.rows();
    int N = X.rows();
    int D = 3;

    Eigen::MatrixXd Y_0 = Y.replicate(1, 1);

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
        const int half_k = 3;
        std::vector<int> indices;
        if (i - half_k < 0) {
            for (int j = 0; j <= i + half_k; j++) {
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

    if (sigma2 == 0) {
        sigma2 = diff_xy.sum() / static_cast<double>(D * M * N);
    }

    for (int it = 0; it < max_iter; it++) {

        // 各ノードの最近傍点群点への距離を更新する (P_vis 計算に使用)
        std::map<int, double> shortest_node_pt_dists;
        for (int m = 0; m < M; m++) {
            double shortest_dist = 10000;
            for (int n = 0; n < N; n++) {
                diff_xy(m, n) = (Y.row(m) - X.row(n)).squaredNorm();
                double dist = (Y.row(m) - X.row(n)).norm();
                if (dist < shortest_dist) shortest_dist = dist;
            }
            if (shortest_dist <= visibility_threshold) shortest_dist = 0;
            shortest_node_pt_dists.insert(std::pair<int, double>(m, shortest_dist));
        }

        // E step: 対応確率行列 P を計算する
        Eigen::MatrixXd P = (-0.5 * diff_xy / sigma2).array().exp();
        double c = pow((2 * M_PI * sigma2), static_cast<double>(D)/2) * mu / (1 - mu) * static_cast<double>(M)/N;
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
            if (potential_2nd_1 == -1) potential_2nd_1 = 2;
            int potential_2nd_2 = max_p_node + 1;
            if (potential_2nd_2 == M) potential_2nd_2 = M - 3;

            int next_max_p_node = (pt2pt_dis(Y.row(potential_2nd_1), X.row(i)) < pt2pt_dis(Y.row(potential_2nd_2), X.row(i)))
                                  ? potential_2nd_1 : potential_2nd_2;

            pts_dis_sq_geodesic(max_p_node, i)  = pt2pt_dis_sq(Y.row(max_p_node), X.row(i));
            pts_dis_sq_geodesic(next_max_p_node, i) = pt2pt_dis_sq(Y.row(next_max_p_node), X.row(i));

            if (max_p_node < next_max_p_node) {
                for (int j = 0; j < max_p_node; j++) {
                    pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[max_p_node]) + pt2pt_dis(Y.row(max_p_node), X.row(i)), 2);
                }
                for (int j = next_max_p_node; j < M; j++) {
                    pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[next_max_p_node]) + pt2pt_dis(Y.row(next_max_p_node), X.row(i)), 2);
                }
            }
            else {
                for (int j = 0; j < next_max_p_node; j++) {
                    pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[next_max_p_node]) + pt2pt_dis(Y.row(next_max_p_node), X.row(i)), 2);
                }
                for (int j = max_p_node; j < M; j++) {
                    pts_dis_sq_geodesic(j, i) = pow(abs(converted_node_coord[j] - converted_node_coord[max_p_node]) + pt2pt_dis(Y.row(max_p_node), X.row(i)), 2);
                }
            }
        }

        P = (-0.5 * pts_dis_sq_geodesic / sigma2).array().exp();

        // 可視確率 P_vis で P を重み付けする (一部ノードがオクルージョン下にある場合)
        if (visible_nodes.size() != static_cast<size_t>(Y.rows()) && !visible_nodes.empty() && k_vis != 0) {
            Eigen::MatrixXd P_vis = Eigen::MatrixXd::Ones(P.rows(), P.cols());
            double total_P_vis = 0;
            for (int i = 0; i < Y.rows(); i++) {
                double P_vis_i = exp(-k_vis * shortest_node_pt_dists[i]);
                total_P_vis += P_vis_i;
                P_vis.row(i) = P_vis_i * P_vis.row(i);
            }
            P_vis = P_vis / total_P_vis;
            P = P.cwiseProduct(P_vis);
            c = pow((2 * M_PI * sigma2), static_cast<double>(D)/2) * mu / (1 - mu) / N;
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
                A_matrix = P1.asDiagonal()*G + lambda*sigma2 * Eigen::MatrixXd::Identity(M, M) + sigma2*lle_weight * H*G + alpha*J*G;
                B_matrix = PX - P1.asDiagonal()*Y_0 - sigma2*lle_weight * H*Y_0 + alpha*(Y_extended - Y_0);
            }
            else {
                A_matrix = P1.asDiagonal()*G + lambda*sigma2 * Eigen::MatrixXd::Identity(M, M) + sigma2*lle_weight * H*G;
                B_matrix = PX - P1.asDiagonal()*Y_0 - sigma2*lle_weight * H*Y_0;
            }
        }
        else {
            if (correspondence_priors.size() != 0) {
                A_matrix = P1.asDiagonal() * G + lambda * sigma2 * Eigen::MatrixXd::Identity(M, M) + alpha*J*G;
                B_matrix = PX - P1.asDiagonal() * Y_0 + alpha*(Y_extended - Y_0);
            }
            else {
                A_matrix = P1.asDiagonal() * G + lambda * sigma2 * Eigen::MatrixXd::Identity(M, M);
                B_matrix = PX - P1.asDiagonal() * Y_0;
            }
        }

        Eigen::MatrixXd W = A_matrix.completeOrthogonalDecomposition().solve(B_matrix);
        Eigen::MatrixXd T = Y_0 + G * W;

        double trXtdPt1X = (X.transpose() * Pt1.asDiagonal() * X).trace();
        double trPXtT    = (PX.transpose() * T).trace();
        double trTtdP1T  = (T.transpose() * P1.asDiagonal() * T).trace();
        sigma2 = (trXtdPt1X - 2*trPXtT + trTtdP1T) / (Np * D);

        if (pt2pt_dis(Y, Y_0 + G*W) / Y.rows() < tol) {
            Y = Y_0 + G*W;
            std::cout << "Iteration until convergence: " << (it+1) << std::endl;
            break;
        }
        else {
            Y = Y_0 + G*W;
        }

        if (it == max_iter - 1) {
            std::cerr << "optimization did not converge!" << std::endl;
            converged = false;
            break;
        }
    }

    return converged;
}

void tracking_step(TrackdloState& state,
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
    double sigma2_pre_proc = state.sigma2;
    cpd_lle(X, state.guide_nodes, sigma2_pre_proc,
            params.beta_pre_proc, params.lambda_pre_proc, params.lle_weight,
            params.mu, params.max_iter, params.tol, true);

    // オクルージョン状態を判定して correspondence_priors を構築する
    if (static_cast<int>(visible_nodes_extended.size()) == state.Y.rows()) {
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

        state.correspondence_priors = {};
        for (int i = 0; i < state.Y.rows(); i++) {
            if (i < priors_vec_2[0](0, 0) && i < static_cast<int>(priors_vec_1.size())) {
                state.correspondence_priors.push_back(priors_vec_1[i]);
            }
            else if (i > priors_vec_1[priors_vec_1.size()-1](0, 0) &&
                     (i-(state.Y.rows()-static_cast<int>(priors_vec_2.size()))) < static_cast<int>(priors_vec_2.size()))
            {
                state.correspondence_priors.push_back(priors_vec_2[i-(state.Y.rows()-priors_vec_2.size())]);
            }
            else {
                state.correspondence_priors.push_back((priors_vec_1[i] + priors_vec_2[i-(state.Y.rows()-priors_vec_2.size())]) / 2.0);
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

    cpd_lle(X, state.Y, state.sigma2,
            params.beta, params.lambda, params.lle_weight,
            params.mu, params.max_iter, params.tol,
            false,
            state.correspondence_priors, params.alpha,
            visible_nodes_extended, params.k_vis, params.visibility_threshold);
}

} // namespace trackdlo
