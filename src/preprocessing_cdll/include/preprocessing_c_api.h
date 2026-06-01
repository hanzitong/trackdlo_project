#pragma once

// ================================================================
// preprocessing C API — preprocessing を C の ABI で再公開するブリッジヘッダ
//
// ─── 目的 ────────────────────────────────────────────────────────
//
// trackdlo_cdll と同じ設計方針。
// preprocessing は C++ (cv::Mat / Eigen::MatrixXd) で書かれているため、
// Python の ctypes から直接呼べない。このヘッダは:
//
//   cv::Mat       → uint8_t* / uint16_t* (生配列)
//   Eigen::MatrixXd → double* (行優先)
//
// に変換して C の ABI で公開する。
//
// ─── 公開する関数 ─────────────────────────────────────────────────
//
//   prep_images_to_pointcloud
//     BGR 画像 + 深度画像 + マスク → 3D 点群 (N×3, 行優先)
//     カメラ内部パラメータ (fx, fy, cx, cy) を個別引数で受け取り、
//     内部で 3×4 射影行列を構築してから preprocessing::images_to_pointcloud() を呼ぶ。
//
//   prep_compute_visible_nodes
//     ノード座標 Y + 点群 X → visible_nodes / visible_nodes_extended
//     出力は呼び元が M 要素の int バッファを用意し、実際の要素数を out_*_len で受け取る。
//
// ─── 可変長出力の設計 ──────────────────────────────────────────────
//
//   点群の出力点数 N はビルド時に不明なため、呼び元が最大サイズ (rows*cols*3) の
//   バッファを渡し、関数が実際に書き込んだ要素数を out_n に返す方式をとる。
//   Python 側では rows*cols*3 の np.empty を確保してから呼び出す。
//
// ================================================================

#include <stdint.h>

// ----------------------------------------------------------------
// PREPROC_API — シンボルエクスポートマクロ (trackdlo_cdll の TRACKDLO_API と同じ仕組み)
// ----------------------------------------------------------------
#ifdef PREPROC_WINDOWS
  #define PREPROC_API __declspec(dllexport)
#else
  #define PREPROC_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------
// prep_images_to_pointcloud
//
// bgr      : uint8_t*, rows × cols × 3 (BGR, 行優先)
// depth    : uint16_t*, rows × cols (mm, 行優先, RealSense z16 フォーマット)
// mask     : uint8_t*, rows × cols (0=背景, >0=ケーブル, 行優先)
// rows, cols : 画像サイズ (画素)
// fx, fy, cx, cy : カメラ内部パラメータ
// leaf_size  : VoxelGrid のセルサイズ (m)。0.0 で無効 (ダウンサンプリングなし)
// out_pts    : double*, 呼び元が rows*cols*3 の領域を確保すること (最悪ケース)
// out_n      : int*, 実際に書き込まれた点数
// ----------------------------------------------------------------
PREPROC_API void prep_images_to_pointcloud(
    const uint8_t*  bgr,
    const uint16_t* depth,
    const uint8_t*  mask,
    int rows, int cols,
    double fx, double fy, double cx, double cy,
    double leaf_size,
    double* out_pts, int* out_n
);

// ----------------------------------------------------------------
// prep_compute_visible_nodes
//
// Y              : double*, M×3 行優先 (前フレームのノード座標)
// M              : ノード数
// X              : double*, N×3 行優先 (現フレームの点群)
// N              : 点群の点数
// geodesic_coord : double*, M 要素 (各ノードの累積弧長 [m])
// fx, fy, cx, cy : カメラ内部パラメータ
// img_rows, img_cols : カメラ画像サイズ (セルフオクルージョン判定用)
// visibility_threshold : ノード-点群間距離の可視判定閾値 (m)
// d_vis          : extended 側のギャップ許容幅 (m)
// dlo_pixel_width : 投影エッジの描画幅 (画素)
// out_vn         : int*, 呼び元が M 要素の領域を確保すること
// out_vn_len     : int*, visible_nodes の実際の要素数
// out_vne        : int*, 呼び元が M 要素の領域を確保すること
// out_vne_len    : int*, visible_nodes_extended の実際の要素数
// ----------------------------------------------------------------
PREPROC_API void prep_compute_visible_nodes(
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
);

#ifdef __cplusplus
}
#endif
