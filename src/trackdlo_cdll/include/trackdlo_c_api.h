#pragma once

// ================================================================
// trackdlo C API
//
// pure_trackdlo (C++) を C の ABI で公開するヘッダ。
// Python の ctypes、C、Julia など、C 関数を呼べる言語から使える。
//
// [設計メモ] 不透明ハンドル (opaque handle) パターン
//   TrackdloState は C++ クラスのインスタンスを void* として管理する。
//   C 側からは中身を直接触れず、get/set 関数を通じてのみアクセスする。
//   こうすることで C++ の実装詳細 (Eigen の MatrixXd など) を
//   C 側に見せずに済む。
//
// [メモリレイアウト] 行列の向き
//   Eigen の MatrixXd はデフォルトで列優先 (column-major)。
//   numpy のデフォルトは行優先 (row-major, C-contiguous)。
//   この API では全ての行列を行優先で受け渡す。
//   C++ 側で Eigen::RowMajor マップを使って変換している。
// ================================================================

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------
// 不透明ハンドル: C 側からは void* として扱う
// ----------------------------------------------------------------
typedef void* TdloState;

// ----------------------------------------------------------------
// パラメータ構造体
// TrackdloParams のフィールドを C 構造体として公開する。
// Python 側では ctypes.Structure として定義して使う。
// ----------------------------------------------------------------
typedef struct {
    double beta;
    double lambda;               /* Python 側では lambda_ として公開 (予約語のため) */
    double alpha;
    double k_vis;
    double mu;
    int    max_iter;
    double tol;
    double beta_pre_proc;
    double lambda_pre_proc;
    double lle_weight;
    double visibility_threshold;
} TdloParams;

// ----------------------------------------------------------------
// State の生成・破棄
// ----------------------------------------------------------------
TdloState tdlo_state_create  (int num_nodes);
void      tdlo_state_free    (TdloState s);
int       tdlo_state_num_nodes(TdloState s);

// Y (M×3 行優先): out は呼び元が double[M*3] を確保する
void   tdlo_state_get_Y     (TdloState s, double* out);
void   tdlo_state_set_Y     (TdloState s, const double* data, int M);

double tdlo_state_get_sigma2(TdloState s);
void   tdlo_state_set_sigma2(TdloState s, double sigma2);

// geodesic_coord (M 個の double)
void   tdlo_state_set_geodesic_coord(TdloState s, const double* data, int n);
void   tdlo_state_get_geodesic_coord(TdloState s, double* out, int* n);

// guide_nodes (M×3 行優先)
void   tdlo_state_set_guide_nodes(TdloState s, const double* data, int M);

// ----------------------------------------------------------------
// デフォルト値で初期化された TdloParams を返す
// ----------------------------------------------------------------
TdloParams tdlo_default_params(void);

// ----------------------------------------------------------------
// tracking_step
//   X:                      点群 (N×3 行優先)
//   visible_nodes:          可視ノードのインデックス配列 (NULL で全ノード可視)
//   visible_nodes_extended: 拡張可視ノードのインデックス配列
//   params:                 トラッキングパラメータ
// state.Y を in-place で更新する。
// ----------------------------------------------------------------
void tdlo_tracking_step(
    TdloState         s,
    const double*     X,                        int X_rows,
    const int*        visible_nodes,            int vn_len,
    const int*        visible_nodes_extended,   int vne_len,
    const TdloParams* params
);

// ----------------------------------------------------------------
// cpd_lle
//   X, Y: N×3 / M×3 行優先。Y は in-place で更新される。
//   sigma2: in/out。
//   visible_nodes: NULL の場合は空リストとして扱う。
//   戻り値: 収束した場合 1、しなかった場合 0
// ----------------------------------------------------------------
int tdlo_cpd_lle(
    const double* X,      int X_rows,
    double*       Y,      int Y_rows,
    double*       sigma2,
    double beta, double lambda, double lle_weight, double mu,
    int max_iter, double tol, int include_lle,
    double alpha,
    const int* visible_nodes, int vn_len,
    double k_vis, double visibility_threshold
);

// ----------------------------------------------------------------
// sort_pts
//   Y_in (M×3 行優先) → Y_out (M×3 行優先、最近傍順に並べ替え)
//   Y_out は呼び元が double[M*3] を確保する。
// ----------------------------------------------------------------
void tdlo_sort_pts(const double* Y_in, int M, double* Y_out);

// ----------------------------------------------------------------
// reg (簡易 CPD 登録、初期化用)
//   pts: N×3 行優先, Y: M×3 行優先 (in-place 更新)
// ----------------------------------------------------------------
void tdlo_reg(
    const double* pts, int pts_rows,
    double*       Y,   int M,
    double*       sigma2,
    double mu, int max_iter
);

#ifdef __cplusplus
}
#endif
