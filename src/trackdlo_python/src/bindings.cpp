#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>   // Eigen::MatrixXd <-> numpy.ndarray 自動変換
#include <pybind11/stl.h>     // std::vector <-> Python list 自動変換
#include <trackdlo.h>
#include <utils.h>

namespace py = pybind11;

PYBIND11_MODULE(trackdlo_python, m) {
    m.doc() = "Python bindings for pure_trackdlo (pybind11)";

    // ----------------------------------------------------------------
    // TrackdloParams
    // ----------------------------------------------------------------
    // [注意] C++ の lambda フィールドは Python の予約語と衝突するため
    //        Python 側の属性名を lambda_ にする。
    py::class_<trackdlo::TrackdloParams>(m, "TrackdloParams")
        .def(py::init<>())
        .def_readwrite("beta",                 &trackdlo::TrackdloParams::beta)
        .def_readwrite("lambda_",              &trackdlo::TrackdloParams::lambda)
        .def_readwrite("alpha",                &trackdlo::TrackdloParams::alpha)
        .def_readwrite("k_vis",                &trackdlo::TrackdloParams::k_vis)
        .def_readwrite("mu",                   &trackdlo::TrackdloParams::mu)
        .def_readwrite("max_iter",             &trackdlo::TrackdloParams::max_iter)
        .def_readwrite("tol",                  &trackdlo::TrackdloParams::tol)
        .def_readwrite("beta_pre_proc",        &trackdlo::TrackdloParams::beta_pre_proc)
        .def_readwrite("lambda_pre_proc",      &trackdlo::TrackdloParams::lambda_pre_proc)
        .def_readwrite("lle_weight",           &trackdlo::TrackdloParams::lle_weight)
        .def_readwrite("visibility_threshold", &trackdlo::TrackdloParams::visibility_threshold);

    // ----------------------------------------------------------------
    // TrackdloState
    // ----------------------------------------------------------------
    py::class_<trackdlo::TrackdloState>(m, "TrackdloState")
        .def_readwrite("Y",                     &trackdlo::TrackdloState::Y)
        .def_readwrite("guide_nodes",           &trackdlo::TrackdloState::guide_nodes)
        .def_readwrite("sigma2",                &trackdlo::TrackdloState::sigma2)
        .def_readwrite("geodesic_coord",        &trackdlo::TrackdloState::geodesic_coord)
        .def_readwrite("correspondence_priors", &trackdlo::TrackdloState::correspondence_priors);

    // ----------------------------------------------------------------
    // make_trackdlo_state
    // ----------------------------------------------------------------
    m.def("make_trackdlo_state", &trackdlo::make_trackdlo_state, py::arg("num_nodes"));

    // ----------------------------------------------------------------
    // tracking_step
    // TrackdloState& を参照渡しするため、pybind11 が自動的に Python 側
    // のオブジェクトへの変更を反映する。呼び出し後 state.Y が更新される。
    // ----------------------------------------------------------------
    m.def("tracking_step",
        &trackdlo::tracking_step,
        py::arg("state"),
        py::arg("X"),
        py::arg("visible_nodes"),
        py::arg("visible_nodes_extended"),
        py::arg("params"));

    // ----------------------------------------------------------------
    // cpd_lle
    // C++ では Y と sigma2 が参照渡しで in-place 変更される。
    // Python には参照渡しがないため、ラムダでラップしてタプルで返す:
    //   (Y_updated, sigma2_updated, converged)
    //
    // [注意] 引数名 lambda_ は Python 予約語 lambda を避けるため。
    //        ラムダ内変数は lam としている。
    // ----------------------------------------------------------------
    m.def("cpd_lle",
        [](const Eigen::MatrixXd& X,
           Eigen::MatrixXd Y,
           double sigma2,
           double beta,
           double lam,
           double lle_weight,
           double mu,
           int max_iter,
           double tol,
           bool include_lle,
           std::vector<Eigen::MatrixXd> correspondence_priors,
           double alpha,
           std::vector<int> visible_nodes,
           double k_vis,
           double visibility_threshold) {
            bool converged = trackdlo::cpd_lle(
                X, Y, sigma2, beta, lam, lle_weight, mu,
                max_iter, tol, include_lle, correspondence_priors,
                alpha, visible_nodes, k_vis, visibility_threshold);
            return py::make_tuple(Y, sigma2, converged);
        },
        py::arg("X"),
        py::arg("Y"),
        py::arg("sigma2"),
        py::arg("beta"),
        py::arg("lambda_"),
        py::arg("lle_weight"),
        py::arg("mu"),
        py::arg("max_iter")                = 30,
        py::arg("tol")                     = 0.0001,
        py::arg("include_lle")             = true,
        py::arg("correspondence_priors")   = std::vector<Eigen::MatrixXd>{},
        py::arg("alpha")                   = 0.0,
        py::arg("visible_nodes")           = std::vector<int>{},
        py::arg("k_vis")                   = 0.0,
        py::arg("visibility_threshold")    = 0.01);

    // ----------------------------------------------------------------
    // reg
    // Y と sigma2 がout引数。タプル (Y, sigma2) で返す。
    // ----------------------------------------------------------------
    m.def("reg",
        [](const Eigen::MatrixXd& pts,
           Eigen::MatrixXd Y,
           double sigma2,
           int M,
           double mu,
           int max_iter) {
            trackdlo::reg(pts, Y, sigma2, M, mu, max_iter);
            return py::make_tuple(Y, sigma2);
        },
        py::arg("pts"),
        py::arg("Y"),
        py::arg("sigma2"),
        py::arg("M"),
        py::arg("mu")       = 0.0,
        py::arg("max_iter") = 50);

    // ----------------------------------------------------------------
    // ユーティリティ関数 (直接バインド)
    // ----------------------------------------------------------------
    m.def("pt2pt_dis_sq",            &trackdlo::pt2pt_dis_sq);
    m.def("pt2pt_dis",               &trackdlo::pt2pt_dis);
    m.def("sort_pts",                &trackdlo::sort_pts);
    m.def("cross_product",           &trackdlo::cross_product);
    m.def("dot_product",             &trackdlo::dot_product);
    m.def("line_sphere_intersection",&trackdlo::line_sphere_intersection);

    // remove_row は matrix を in-place で変更するためラッパーが必要
    m.def("remove_row",
        [](Eigen::MatrixXd matrix, unsigned int row) {
            trackdlo::remove_row(matrix, row);
            return matrix;
        },
        py::arg("matrix"),
        py::arg("row"));
}
