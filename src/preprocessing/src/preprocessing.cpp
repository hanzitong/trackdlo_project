#include "../include/preprocessing.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <map>

namespace preprocessing {


// ============================================================
// [メモ] VoxelGridアルゴリズムの仕組み
//
// 目的: 密な点群を間引いて計算量を減らす。
//
// 手順:
//   1. 空間を leaf_size (m) の立方体格子(ボクセル)に分割する。
//      各点 (x, y, z) が属するボクセルのインデックスは
//        kx = floor(x / leaf_size)
//        ky = floor(y / leaf_size)
//        kz = floor(z / leaf_size)
//
//   2. 同じボクセルに属する点を集め、その重心を代表点とする。
//
//   3. 各ボクセルから代表点1個だけを出力する。
//
// 以下はアルゴリズムをEigen+STLで書いた参考実装 (実際にはPCLを使う):
//
//   std::map<std::tuple<int,int,int>, std::pair<Eigen::RowVector3d, int>> voxels;
//   for (int i = 0; i < pts.rows(); i++) {
//       int kx = (int)std::floor(pts(i,0) / leaf_size);
//       int ky = (int)std::floor(pts(i,1) / leaf_size);
//       int kz = (int)std::floor(pts(i,2) / leaf_size);
//       voxels[{kx, ky, kz}].first  += pts.row(i);  // 座標の合計
//       voxels[{kx, ky, kz}].second += 1;            // 個数
//   }
//   // 各ボクセルの平均座標を代表点とする
//   for (auto& [key, val] : voxels)
//       result.row(row++) = val.first / val.second;
//
// PCLのVoxelGridは内部でハッシュを使うためより高速だが、原理は同じ。
// ============================================================

cv::Mat color_threshold(const cv::Mat& rgb_bgr,
                        const std::vector<int>& lower,
                        const std::vector<int>& upper) {
    cv::Mat hsv, mask;
    cv::cvtColor(rgb_bgr, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv,
                cv::Scalar(lower[0], lower[1], lower[2]),
                cv::Scalar(upper[0], upper[1], upper[2]),
                mask);
    return mask;
}


cv::Mat color_threshold_multicolor(const cv::Mat& rgb_bgr) {
    cv::Mat hsv;
    cv::cvtColor(rgb_bgr, hsv, cv::COLOR_BGR2HSV);

    // 青
    cv::Mat mask_blue;
    cv::inRange(hsv, cv::Scalar(90, 90, 60), cv::Scalar(130, 255, 255), mask_blue);

    // 赤 (HSVで赤は0付近と180付近に分かれるため2範囲を合成)
    cv::Mat mask_red_1, mask_red_2, mask_red;
    cv::inRange(hsv, cv::Scalar(130, 60, 50), cv::Scalar(255, 255, 255), mask_red_1);
    cv::inRange(hsv, cv::Scalar(0,   60, 50), cv::Scalar(10,  255, 255), mask_red_2);
    cv::bitwise_or(mask_red_1, mask_red_2, mask_red);

    // 黄
    cv::Mat mask_yellow;
    cv::inRange(hsv, cv::Scalar(15, 100, 80), cv::Scalar(40, 255, 255), mask_yellow);

    cv::Mat mask;
    cv::bitwise_or(mask_red,    mask_blue, mask);
    cv::bitwise_or(mask_yellow, mask,      mask);
    return mask;
}


Eigen::MatrixXd images_to_pointcloud(const cv::Mat& rgb_bgr,
                               const cv::Mat& depth,
                               const Eigen::MatrixXd& proj_matrix,
                               const cv::Mat& mask_in,
                               double leaf_size,
                               const cv::Mat& occ_mask) {
    // オクルージョンマスクが指定されていれば適用する
    // occ_mask はBGR画像で、黒(0)の領域がオクルージョン領域
    cv::Mat mask;
    if (!occ_mask.empty()) {
        cv::Mat occ_gray;
        cv::cvtColor(occ_mask, occ_gray, cv::COLOR_BGR2GRAY);
        cv::bitwise_and(mask_in, occ_gray, mask);
    } else {
        mask = mask_in;
    }

    // カメラ内部パラメータをproj_matrixから取り出す
    double fx = proj_matrix(0, 0);
    double fy = proj_matrix(1, 1);
    double cx = proj_matrix(0, 2);
    double cy = proj_matrix(1, 2);

    // マスクが立っているピクセルについて、ピンホールモデルで3D座標に逆投影する
    // depth は uint16_t (mm単位) → /1000.0 でメートル変換
    // 逆投影式:
    //   X = (u - cx) * Z / fx
    //   Y = (v - cy) * Z / fy
    //   Z = depth_mm / 1000.0
    pcl::PointCloud<pcl::PointXYZ> cloud;
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if (mask.at<uchar>(i, j) == 0) continue;

            double pc_z = depth.at<uint16_t>(i, j) / 1000.0;
            if (pc_z <= 0.0) continue;  // 深度が無効な画素をスキップ

            pcl::PointXYZ pt;
            pt.x = static_cast<float>((j - cx) * pc_z / fx);
            pt.y = static_cast<float>((i - cy) * pc_z / fy);
            pt.z = static_cast<float>(pc_z);
            cloud.push_back(pt);
        }
    }

    if (cloud.empty()) return Eigen::MatrixXd(0, 3);

    // PCL VoxelGridでダウンサンプリング
    pcl::PointCloud<pcl::PointXYZ> downsampled;
    if (leaf_size > 0.0) {
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(cloud.makeShared());
        vg.setLeafSize(static_cast<float>(leaf_size),
                       static_cast<float>(leaf_size),
                       static_cast<float>(leaf_size));
        vg.filter(downsampled);
    } else {
        downsampled = cloud;
    }

    // pcl::PointCloud → Eigen::MatrixXd
    // getMatrixXfMap(): PCL内部のfloat配列をEigen行列として参照する
    //   返値は (4×N): [x, y, z, padding] の4行
    //   topRows(3): x,y,z の3行だけ取る
    //   transpose(): (3×N) → (N×3)
    //   cast<double>(): float → double
    return downsampled.getMatrixXfMap().topRows(3).transpose().cast<double>();
}


// ============================================================
// compute_visible_nodes — ノード可視性計算
//
// ─── 背景・前提 ────────────────────────────────────────────
//
// trackdlo はケーブル (DLO) の形状を M 個の代表点 (ノード) で表す。
// ノードの座標列 Y (M×3) が「今フレームでのケーブルの推定形状」であり、
// これは前のフレームから引き継いだ状態量 (TrackdloState::Y) として保持されている。
//
// RealSense などの RGB-D カメラは 1 フレームごとに
//   ・カラー画像   (RGB)
//   ・深度画像     (各画素までの距離)
// を出力する。これらから color_threshold() → images_to_pointcloud() を経て
// ケーブル色の 3D 点群 X (N×3) が得られる。
//
// この関数は、前フレームのノード Y と現フレームの点群 X を突き合わせ、
// 各ノードが「今のカメラ視点から見えているか」を判定する。
//
// ─── なぜ可視性が必要か ─────────────────────────────────
//
// ケーブルは箱や机の下に隠れることがある (オクルージョン)。
// 隠れている区間のノードに対して「ここに点群があるはずだ」と
// 強く引き寄せようとすると、点群の無い方向へノードが飛んでしまう。
// visible_nodes = 「現在カメラで見えているノードの番号リスト」を
// tracking_step() に渡すことで、見えていないノードは点群への引き寄せを
// 弱め、代わりに測地線補間で形状を維持するよう切り替えられる。
//
// ─── 判定の2条件 ──────────────────────────────────────────
//
// ノード m が可視とみなされるには以下を両方満たす必要がある:
//
//   条件 A: 点群との最短距離 ≤ visibility_threshold (m)
//     ノードの 3D 座標の近傍に点群の点が存在する
//     = 「そこにケーブルが実際に見えている」
//
//   条件 B: セルフオクルージョンしていない
//     ケーブル自身が自分の一部を隠す場合がある (例: ループ状になった時)。
//     近い方のエッジを先にカメラ投影画像に描画し、
//     後で投影されるエッジの位置がすでに描画済みなら「後ろに隠れている」と判定する。
//
// ─── 2 種類の出力 ──────────────────────────────────────────
//
//   visible_nodes
//     上記 2 条件を満たしたノードのインデックス列 (ソート済み)。
//     ケーブルが部分的に隠れている場合、隠れている区間のインデックスは含まれない。
//
//   visible_nodes_extended
//     visible_nodes の隣接ペア間の測地線距離が d_vis 以内であれば
//     間のノードも可視とみなして補完したリスト。
//     小さなオクルージョンギャップ (数 cm 程度の隙間) を埋める目的で使う。
//     tracking_step() 内部では cpd_lle の前処理 (guide_nodes の更新) に使われる。
//
// ─── 引数まとめ ────────────────────────────────────────────
//
//   Y                    前フレームから引き継いだノード座標 (M×3, 単位 m)
//                        M = ノード数 (make_trackdlo_state(M) で決める定数。典型値 10〜20)
//   X                    現フレームの入力点群 (N×3, 単位 m)
//                        N = 点群の点数 (フレームごとに変わる変数。典型値 数百〜数万)
//   proj_matrix          カメラ射影行列 (3×4)  [ fx 0 cx 0 / 0 fy cy 0 / 0 0 1 0 ]
//   geodesic_coord       各ノードの累積弧長 (M 要素, 単位 m)
//                        visible_nodes_extended のギャップ判定 (d_vis との比較) に使用
//   img_rows, img_cols   カメラ画像のサイズ (画素)。投影マスクの生成に使用
//   visibility_threshold 条件 A の閾値 (m)。典型値: 0.01 〜 0.02 m
//   d_vis                visible_nodes_extended のギャップ許容幅 (m)。典型値: 0.05 〜 0.1 m
//   dlo_pixel_width      投影エッジの描画幅 (画素)。大きいほどセルフオクルージョン判定が緩くなる
//   visible_nodes        出力: 可視ノードのインデックス列
//   visible_nodes_extended 出力: ギャップ補完済みの拡張可視ノードインデックス列
//
// ============================================================
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
                           std::vector<int>& visible_nodes_extended)
{
    visible_nodes.clear();
    visible_nodes_extended.clear();

    // Step 1: 各ノードから最近傍点群点への距離を計算する
    // shortest_node_pt_dists[m] = ノードm から点群X への最短距離
    std::map<int, double> shortest_node_pt_dists;
    for (int m = 0; m < Y.rows(); m++) {
        double shortest_dist = 1e5;
        for (int n = 0; n < X.rows(); n++) {
            double dist = (Y.row(m) - X.row(n)).norm();
            if (dist < shortest_dist) shortest_dist = dist;
        }
        shortest_node_pt_dists[m] = shortest_dist;
    }

    // Step 2: 各エッジをカメラからの距離でソートする (近い順に処理)
    // セルフオクルージョン判定: 手前のエッジを先に描画し、後ろのエッジが隠れるかを判定する
    std::vector<double> averaged_node_camera_dists;
    std::vector<int> indices_vec;
    for (int i = 0; i < Y.rows()-1; i++) {
        averaged_node_camera_dists.push_back(((Y.row(i) + Y.row(i+1)) / 2).norm());
        indices_vec.push_back(i);
    }
    std::sort(indices_vec.begin(), indices_vec.end(),
        [&](const int& a, const int& b) {
            return averaged_node_camera_dists[a] < averaged_node_camera_dists[b];
        });

    // Step 3: 各ノードの 3D 座標 (X,Y,Z) に 1 を付け足して (X,Y,Z,1) にし (同次座標に拡張)、
    //         射影行列とかけることで 2D 画像座標に投影する
    // Y_h = [Y | 1] (M×4), image_coords = proj_matrix * Y_h^T (M×3)
    // 画像座標: u = image_coords(i,0)/image_coords(i,2)
    //           v = image_coords(i,1)/image_coords(i,2)
    Eigen::MatrixXd Y_h = Y.replicate(1, 1);
    Y_h.conservativeResize(Y_h.rows(), Y_h.cols() + 1);
    Y_h.col(Y_h.cols() - 1) = Eigen::MatrixXd::Ones(Y_h.rows(), 1);
    Eigen::MatrixXd image_coords = (proj_matrix * Y_h.transpose()).transpose();

    // Step 4: 近いエッジから順に投影マスクに描画し、セルフオクルージョンを判定する
    // projected_edges の画素が 0 → まだ手前のエッジに隠れていない = 可視
    cv::Mat projected_edges = cv::Mat::zeros(img_rows, img_cols, CV_8U);

    for (int idx : indices_vec) {
        int col_1 = static_cast<int>(image_coords(idx,   0) / image_coords(idx,   2));
        int row_1 = static_cast<int>(image_coords(idx,   1) / image_coords(idx,   2));
        int col_2 = static_cast<int>(image_coords(idx+1, 0) / image_coords(idx+1, 2));
        int row_2 = static_cast<int>(image_coords(idx+1, 1) / image_coords(idx+1, 2));

        // ノードidxが可視か判定: 投影位置が未使用 かつ 点群に近い
        if (projected_edges.at<uchar>(row_1, col_1) == 0) {
            if (shortest_node_pt_dists[idx] <= visibility_threshold) {
                if (std::find(visible_nodes.begin(), visible_nodes.end(), idx) == visible_nodes.end()) {
                    visible_nodes.push_back(idx);
                }
            }
        }

        // ノードidx+1が可視か判定
        if (projected_edges.at<uchar>(row_2, col_2) == 0) {
            if (shortest_node_pt_dists[idx+1] <= visibility_threshold) {
                if (std::find(visible_nodes.begin(), visible_nodes.end(), idx+1) == visible_nodes.end()) {
                    visible_nodes.push_back(idx+1);
                }
            }
        }

        // このエッジを投影マスクに描画 → 後続の遠いエッジのオクルージョン判定に使う
        cv::line(projected_edges,
                 cv::Point(col_1, row_1), cv::Point(col_2, row_2),
                 cv::Scalar(255, 255, 255), dlo_pixel_width);
    }

    // 元のノード順序 (連続性) を保つためにソートする
    std::sort(visible_nodes.begin(), visible_nodes.end());

    // Step 5: visible_nodes_extended を構築する
    // 隣接する可視ノード間の測地線距離が d_vis 以内なら、間のノードも可視とみなして埋める
    // (小さなオクルージョンギャップを補完する)
    for (int i = 0; i < static_cast<int>(visible_nodes.size()) - 1; i++) {
        visible_nodes_extended.push_back(visible_nodes[i]);
        if (std::fabs(geodesic_coord[visible_nodes[i+1]] - geodesic_coord[visible_nodes[i]]) <= d_vis) {
            for (int j = 1; j < visible_nodes[i+1] - visible_nodes[i]; j++) {
                visible_nodes_extended.push_back(visible_nodes[i] + j);
            }
        }
    }
    if (!visible_nodes.empty()) {
        visible_nodes_extended.push_back(visible_nodes.back());
    }
}

} // namespace preprocessing
