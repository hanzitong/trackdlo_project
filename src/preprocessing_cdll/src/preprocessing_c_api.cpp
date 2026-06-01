#include "../include/preprocessing_c_api.h"
#include <preprocessing.h>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <vector>

// ================================================================
// [メモ] trackdlo_c_api.cpp と同じ行優先変換ヘルパー
//
// Python (numpy) は行優先 (row-major)。Eigen はデフォルトで列優先 (column-major)。
// Eigen::Map<RowMajor> を使うことでコピーなしに行優先メモリを正しく解釈する。
// ================================================================

static Eigen::MatrixXd from_row_major(const double* data, int rows, int cols) {
    return Eigen::Map<
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    >(data, rows, cols);
}

static void to_row_major(const Eigen::MatrixXd& mat, double* out) {
    Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    >(out, mat.rows(), mat.cols()) = mat;
}

extern "C" {

// ----------------------------------------------------------------
// prep_images_to_pointcloud
// ----------------------------------------------------------------
void prep_images_to_pointcloud(
    const uint8_t*  bgr,
    const uint16_t* depth,
    const uint8_t*  mask,
    int rows, int cols,
    double fx, double fy, double cx, double cy,
    double leaf_size,
    double* out_pts, int* out_n
) {
    // raw pointer → cv::Mat (データのコピーなし。ポインタが指すメモリを参照するだけ)
    // const_cast が必要な理由: cv::Mat は非 const ポインタを取るが、実際には書き込まない
    cv::Mat bgr_mat  (rows, cols, CV_8UC3,  const_cast<uint8_t* >(bgr));
    cv::Mat depth_mat(rows, cols, CV_16UC1, const_cast<uint16_t*>(depth));
    cv::Mat mask_mat (rows, cols, CV_8UC1,  const_cast<uint8_t* >(mask));

    // カメラ内部パラメータ (fx, fy, cx, cy) から 3×4 射影行列を構築する
    // [ fx   0  cx  0 ]
    // [  0  fy  cy  0 ]
    // [  0   0   1  0 ]
    Eigen::MatrixXd proj(3, 4);
    proj << fx,  0, cx, 0,
             0, fy, cy, 0,
             0,  0,  1, 0;

    Eigen::MatrixXd pts = preprocessing::images_to_pointcloud(
        bgr_mat, depth_mat, proj, mask_mat, leaf_size
    );

    *out_n = static_cast<int>(pts.rows());
    if (pts.rows() > 0) {
        to_row_major(pts, out_pts);
    }
}

// ----------------------------------------------------------------
// prep_compute_visible_nodes
// ----------------------------------------------------------------
void prep_compute_visible_nodes(
    const double* Y, int M,
    const double* X, int N,
    const double* geodesic_coord,
    double fx, double fy, double cx, double cy,
    int img_rows, int img_cols,
    double visibility_threshold,
    double d_vis,
    int dlo_pixel_width,
    int* out_vn,  int* out_vn_len,
    int* out_vne, int* out_vne_len
) {
    Eigen::MatrixXd Y_mat = from_row_major(Y, M, 3);
    Eigen::MatrixXd X_mat = from_row_major(X, N, 3);

    // double* → std::vector<double>
    std::vector<double> geo(geodesic_coord, geodesic_coord + M);

    Eigen::MatrixXd proj(3, 4);
    proj << fx,  0, cx, 0,
              0, fy, cy, 0,
              0,  0,  1, 0;

    std::vector<int> visible_nodes;
    std::vector<int> visible_nodes_extended;

    preprocessing::compute_visible_nodes(
        Y_mat, X_mat, proj, geo,
        img_rows, img_cols,
        visibility_threshold, d_vis, dlo_pixel_width,
        visible_nodes, visible_nodes_extended
    );

    // std::vector<int> → int* (呼び元のバッファにコピー)
    *out_vn_len = static_cast<int>(visible_nodes.size());
    for (int i = 0; i < *out_vn_len; ++i) {
        out_vn[i] = visible_nodes[i];
    }

    *out_vne_len = static_cast<int>(visible_nodes_extended.size());
    for (int i = 0; i < *out_vne_len; ++i) {
        out_vne[i] = visible_nodes_extended[i];
    }
}

} // extern "C"
