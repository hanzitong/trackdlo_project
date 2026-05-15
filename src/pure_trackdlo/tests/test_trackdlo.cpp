#include <gtest/gtest.h>
#include "../include/trackdlo.h"


// ---- make_trackdlo_state -----------------------------------------------

TEST(MakeState, NodeCountResultSize) {
    trackdlo::TrackdloState state = trackdlo::make_trackdlo_state(5);
    EXPECT_EQ(state.Y.rows(), 5);
    EXPECT_EQ(state.Y.cols(), 3);
}

TEST(MakeState, InitialSigma2IsZero) {
    trackdlo::TrackdloState state = trackdlo::make_trackdlo_state(5);
    EXPECT_DOUBLE_EQ(state.sigma2, 0.0);
}

// ---- state の直接操作 ---------------------------------------------------

TEST(StateManipulation, AssignNodes) {
    trackdlo::TrackdloState state = trackdlo::make_trackdlo_state(3);
    Eigen::MatrixXd Y(3, 3);
    Y << 1.0, 0.0, 0.0,
         2.0, 0.0, 0.0,
         3.0, 0.0, 0.0;
    state.Y = Y;
    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(state.Y(i, 0), Y(i, 0), 1e-12);
    }
}

TEST(StateManipulation, SetSigma2) {
    trackdlo::TrackdloState state = trackdlo::make_trackdlo_state(3);
    state.sigma2 = 0.5;
    EXPECT_DOUBLE_EQ(state.sigma2, 0.5);
}

// ---- TrackdloParams のデフォルト値 --------------------------------------

TEST(Params, DefaultValues) {
    trackdlo::TrackdloParams p;
    EXPECT_DOUBLE_EQ(p.beta,   5.0);
    EXPECT_DOUBLE_EQ(p.lambda, 1.0);
    EXPECT_DOUBLE_EQ(p.mu,     0.05);
    EXPECT_EQ(p.max_iter,      50);
}

// ---- cpd_lle スモークテスト ---------------------------------------------

TEST(CpdLle, DoesNotCrashAndKeepsSize) {
    // calc_LLE_weights は k=6 (k/2=3) を使うため、
    // M > 2*(k/2) = 6 が必要。M=10 で境界外アクセスを回避する。
    int M = 10;
    int N = 20;
    Eigen::MatrixXd Y(M, 3);
    for (int i = 0; i < M; i++) {
        Y(i, 0) = 0.1 * i;
        Y(i, 1) = 0.0;
        Y(i, 2) = 0.0;
    }
    Eigen::MatrixXd X(N, 3);
    for (int i = 0; i < N; i++) {
        X(i, 0) = 0.05 * i;
        X(i, 1) = 0.0;
        X(i, 2) = 0.0;
    }
    double sigma2 = 0.01;

    bool converged = trackdlo::cpd_lle(X, Y, sigma2, 5.0, 1.0, 1.0, 0.05);

    EXPECT_EQ(Y.rows(), M);
    EXPECT_EQ(Y.cols(), 3);
    EXPECT_TRUE(converged == true || converged == false);
}
