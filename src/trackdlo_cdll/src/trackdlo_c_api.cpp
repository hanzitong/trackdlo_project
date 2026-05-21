#include "../include/trackdlo_c_api.h"
#include <trackdlo.h>
#include <utils.h>
#include <Eigen/Dense>
#include <vector>

// ================================================================
// [メモ] 行優先 ↔ Eigen::MatrixXd の変換
//
// Python (numpy) はデフォルトで行優先 (C-contiguous, row-major) に
// 配列をメモリに並べる。例: 3×2 行列なら
//   [row0_col0, row0_col1, row1_col0, row1_col1, row2_col0, row2_col1]
//
// Eigen の MatrixXd はデフォルトで列優先 (column-major, Fortran order)。
//   [row0_col0, row1_col0, row2_col0, row0_col1, ...]
//
// Eigen::Map を使うと既存メモリをコピーなしで行列として解釈できる。
// Eigen::RowMajor を指定すると行優先のメモリを正しく解釈する。
//
// 以下の2つのヘルパー関数で変換を行う:
//   from_row_major : double* (行優先) → MatrixXd (列優先)
//   to_row_major   : MatrixXd (列優先) → double* (行優先) に書き出す
// ================================================================

static Eigen::MatrixXd from_row_major(const double* data, int rows, int cols) {
    // Eigen::Map は既存メモリを行列として「解釈」するビュー
    // コピーを避けてそのままコンストラクタに渡すと列優先の MatrixXd が作られる
    return Eigen::Map<
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    >(data, rows, cols);
}

static void to_row_major(const Eigen::MatrixXd& mat, double* out) {
    Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    >(out, mat.rows(), mat.cols()) = mat;
}

// ================================================================
// State の生成・破棄
// ================================================================

extern "C" TdloState tdlo_state_create(int num_nodes) {
    // new で C++ オブジェクトをヒープに確保し、void* として返す
    // 呼び元は tdlo_state_free で必ず解放すること (メモリリーク防止)
    return new trackdlo::TrackdloState(trackdlo::make_trackdlo_state(num_nodes));
}

extern "C" void tdlo_state_free(TdloState s) {
    // static_cast<T*>: void* を適切な型に戻す
    // (C++ では void* からの変換に static_cast が必要)
    delete static_cast<trackdlo::TrackdloState*>(s);
}

extern "C" int tdlo_state_num_nodes(TdloState s) {
    return static_cast<int>(
        static_cast<trackdlo::TrackdloState*>(s)->Y.rows()
    );
}

extern "C" void tdlo_state_get_Y(TdloState s, double* out) {
    const trackdlo::TrackdloState* st = static_cast<const trackdlo::TrackdloState*>(s);
    to_row_major(st->Y, out);
}

extern "C" void tdlo_state_set_Y(TdloState s, const double* data, int M) {
    static_cast<trackdlo::TrackdloState*>(s)->Y = from_row_major(data, M, 3);
}

extern "C" double tdlo_state_get_sigma2(TdloState s) {
    return static_cast<trackdlo::TrackdloState*>(s)->sigma2;
}

extern "C" void tdlo_state_set_sigma2(TdloState s, double sigma2) {
    static_cast<trackdlo::TrackdloState*>(s)->sigma2 = sigma2;
}

extern "C" void tdlo_state_set_geodesic_coord(TdloState s, const double* data, int n) {
    trackdlo::TrackdloState* st = static_cast<trackdlo::TrackdloState*>(s);
    // assign(begin, end) で std::vector を配列から初期化する
    st->geodesic_coord.assign(data, data + n);
}

extern "C" void tdlo_state_get_geodesic_coord(TdloState s, double* out, int* n) {
    const trackdlo::TrackdloState* st = static_cast<const trackdlo::TrackdloState*>(s);
    *n = static_cast<int>(st->geodesic_coord.size());
    std::copy(st->geodesic_coord.begin(), st->geodesic_coord.end(), out);
}

extern "C" void tdlo_state_set_guide_nodes(TdloState s, const double* data, int M) {
    static_cast<trackdlo::TrackdloState*>(s)->guide_nodes = from_row_major(data, M, 3);
}

// ================================================================
// デフォルトパラメータ
// ================================================================

extern "C" TdloParams tdlo_default_params(void) {
    trackdlo::TrackdloParams cpp;
    TdloParams c;
    c.beta                 = cpp.beta;
    c.lambda               = cpp.lambda;
    c.alpha                = cpp.alpha;
    c.k_vis                = cpp.k_vis;
    c.mu                   = cpp.mu;
    c.max_iter             = cpp.max_iter;
    c.tol                  = cpp.tol;
    c.beta_pre_proc        = cpp.beta_pre_proc;
    c.lambda_pre_proc      = cpp.lambda_pre_proc;
    c.lle_weight           = cpp.lle_weight;
    c.visibility_threshold = cpp.visibility_threshold;
    return c;
}

// ================================================================
// tracking_step
// ================================================================

extern "C" void tdlo_tracking_step(
    TdloState         s,
    const double*     X,    int X_rows,
    const int*        vn,   int vn_len,
    const int*        vne,  int vne_len,
    const TdloParams* params
) {
    trackdlo::TrackdloState* st = static_cast<trackdlo::TrackdloState*>(s);

    Eigen::MatrixXd X_mat = from_row_major(X, X_rows, 3);

    std::vector<int> visible_nodes(vn, vn + vn_len);
    std::vector<int> visible_nodes_ext(vne, vne + vne_len);

    trackdlo::TrackdloParams p;
    p.beta                 = params->beta;
    p.lambda               = params->lambda;
    p.alpha                = params->alpha;
    p.k_vis                = params->k_vis;
    p.mu                   = params->mu;
    p.max_iter             = params->max_iter;
    p.tol                  = params->tol;
    p.beta_pre_proc        = params->beta_pre_proc;
    p.lambda_pre_proc      = params->lambda_pre_proc;
    p.lle_weight           = params->lle_weight;
    p.visibility_threshold = params->visibility_threshold;

    trackdlo::tracking_step(*st, X_mat, visible_nodes, visible_nodes_ext, p);
}

// ================================================================
// cpd_lle
// ================================================================

extern "C" int tdlo_cpd_lle(
    const double* X,     int X_rows,
    double*       Y,     int Y_rows,
    double*       sigma2,
    double beta, double lambda, double lle_weight, double mu,
    int max_iter, double tol, int include_lle,
    double alpha,
    const int* vn, int vn_len,
    double k_vis, double visibility_threshold
) {
    Eigen::MatrixXd X_mat = from_row_major(X, X_rows, 3);
    Eigen::MatrixXd Y_mat = from_row_major(Y, Y_rows, 3);
    double sig2 = *sigma2;

    std::vector<int> visible_nodes;
    if (vn != nullptr && vn_len > 0) {
        visible_nodes.assign(vn, vn + vn_len);
    }

    bool converged = trackdlo::cpd_lle(
        X_mat, Y_mat, sig2,
        beta, lambda, lle_weight, mu,
        max_iter, tol, include_lle != 0,
        {}, alpha, visible_nodes, k_vis, visibility_threshold
    );

    to_row_major(Y_mat, Y);
    *sigma2 = sig2;
    return converged ? 1 : 0;
}

// ================================================================
// sort_pts
// ================================================================

extern "C" void tdlo_sort_pts(const double* Y_in, int M, double* Y_out) {
    Eigen::MatrixXd sorted = trackdlo::sort_pts(from_row_major(Y_in, M, 3));
    to_row_major(sorted, Y_out);
}

// ================================================================
// reg
// ================================================================

extern "C" void tdlo_reg(
    const double* pts, int pts_rows,
    double*       Y,   int M,
    double*       sigma2,
    double mu, int max_iter
) {
    Eigen::MatrixXd pts_mat = from_row_major(pts, pts_rows, 3);
    Eigen::MatrixXd Y_mat   = from_row_major(Y, M, 3);
    double sig2 = *sigma2;

    trackdlo::reg(pts_mat, Y_mat, sig2, M, mu, max_iter);

    to_row_major(Y_mat, Y);
    *sigma2 = sig2;
}
