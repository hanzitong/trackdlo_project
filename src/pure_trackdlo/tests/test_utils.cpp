#include <gtest/gtest.h>
#include "../include/utils.h"


// ---- pt2pt_dis_sq / pt2pt_dis ----------------------------------------

TEST(Pt2PtDis, SamePointIsZero) {
    Eigen::MatrixXd p(1, 3);
    p << 1.0, 2.0, 3.0;
    EXPECT_DOUBLE_EQ(trackdlo::pt2pt_dis_sq(p, p), 0.0);
    EXPECT_DOUBLE_EQ(trackdlo::pt2pt_dis(p, p),    0.0);
}

TEST(Pt2PtDis, KnownDistance) {
    // (0,0,0) と (3,4,0) の距離は 5
    Eigen::MatrixXd a(1, 3), b(1, 3);
    a << 0.0, 0.0, 0.0;
    b << 3.0, 4.0, 0.0;
    EXPECT_DOUBLE_EQ(trackdlo::pt2pt_dis_sq(a, b), 25.0);
    EXPECT_DOUBLE_EQ(trackdlo::pt2pt_dis(a, b),     5.0);
}

// ---- cross_product --------------------------------------------------------

TEST(CrossProduct, StandardBasis) {
    // (1,0,0) × (0,1,0) = (0,0,1)
    Eigen::MatrixXd x(1, 3), y(1, 3);
    x << 1.0, 0.0, 0.0;
    y << 0.0, 1.0, 0.0;
    Eigen::MatrixXd result = trackdlo::cross_product(x, y);
    EXPECT_DOUBLE_EQ(result(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(result(0, 2), 1.0);
}

TEST(CrossProduct, ParallelVectorsGiveZero) {
    // 平行ベクトルの外積はゼロ
    Eigen::MatrixXd v(1, 3);
    v << 1.0, 0.0, 0.0;
    Eigen::MatrixXd result = trackdlo::cross_product(v, v);
    EXPECT_DOUBLE_EQ(result(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(result(0, 2), 0.0);
}

// ---- dot_product ----------------------------------------------------------

TEST(DotProduct, OrthogonalIsZero) {
    Eigen::MatrixXd x(1, 3), y(1, 3);
    x << 1.0, 0.0, 0.0;
    y << 0.0, 1.0, 0.0;
    EXPECT_DOUBLE_EQ(trackdlo::dot_product(x, y), 0.0);
}

TEST(DotProduct, ParallelUnitVectors) {
    Eigen::MatrixXd v(1, 3);
    v << 1.0, 0.0, 0.0;
    EXPECT_DOUBLE_EQ(trackdlo::dot_product(v, v), 1.0);
}

// ---- remove_row -----------------------------------------------------------

TEST(RemoveRow, RemoveMiddleRow) {
    Eigen::MatrixXd m(3, 3);
    m << 1, 0, 0,
         0, 1, 0,
         0, 0, 1;
    trackdlo::remove_row(m, 1);
    ASSERT_EQ(m.rows(), 2);
    // 残るのは行0と行2
    EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m(1, 2), 1.0);
}

TEST(RemoveRow, RemoveLastRow) {
    Eigen::MatrixXd m(3, 3);
    m << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;
    trackdlo::remove_row(m, 2);
    ASSERT_EQ(m.rows(), 2);
    EXPECT_DOUBLE_EQ(m(1, 0), 4.0);
}

// ---- sort_pts -------------------------------------------------------------

TEST(SortPts, StraightLineGetsOrdered) {
    // x方向に並ぶ3点をランダム順で入力
    // sort_ptsは最近傍グリーディ探索なので、連続した点が隣り合うはず
    Eigen::MatrixXd pts(3, 3);
    pts << 2.0, 0.0, 0.0,  // 本来2番目
           0.0, 0.0, 0.0,  // 本来0番目
           1.0, 0.0, 0.0;  // 本来1番目
    Eigen::MatrixXd sorted = trackdlo::sort_pts(pts);
    ASSERT_EQ(sorted.rows(), 3);
    // 隣接点間距離がすべて1.0になっているか確認
    double d01 = trackdlo::pt2pt_dis(sorted.row(0), sorted.row(1));
    double d12 = trackdlo::pt2pt_dis(sorted.row(1), sorted.row(2));
    EXPECT_NEAR(d01, 1.0, 1e-9);
    EXPECT_NEAR(d12, 1.0, 1e-9);
}

// ---- line_sphere_intersection --------------------------------------------

TEST(LineSphereIntersection, TwoIntersections) {
    // 原点中心・半径1の球と、x軸上の線分 (-2,0,0)-(2,0,0)
    // 交点は (-1,0,0) と (1,0,0)
    Eigen::MatrixXd A(1, 3), B(1, 3), C(1, 3);
    A << -2.0, 0.0, 0.0;
    B <<  2.0, 0.0, 0.0;
    C <<  0.0, 0.0, 0.0;
    auto result = trackdlo::line_sphere_intersection(A, B, C, 1.0);
    ASSERT_EQ(result.size(), 2u);
    // 2点のx座標が ±1 (順序は問わない)
    double x0 = result[0](0, 0);
    double x1 = result[1](0, 0);
    EXPECT_NEAR(std::abs(x0), 1.0, 1e-9);
    EXPECT_NEAR(std::abs(x1), 1.0, 1e-9);
    EXPECT_NEAR(x0 + x1, 0.0, 1e-9);  // 符号が逆
}

TEST(LineSphereIntersection, NoIntersection) {
    // 球の外側を通る線分
    Eigen::MatrixXd A(1, 3), B(1, 3), C(1, 3);
    A << 0.0, 2.0, 0.0;
    B << 1.0, 2.0, 0.0;
    C << 0.0, 0.0, 0.0;
    auto result = trackdlo::line_sphere_intersection(A, B, C, 1.0);
    EXPECT_EQ(result.size(), 0u);
}
