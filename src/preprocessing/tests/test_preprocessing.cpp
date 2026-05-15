#include <gtest/gtest.h>
#include "../include/preprocessing.h"

#include <opencv2/core.hpp>
#include <Eigen/Dense>


// ---- color_threshold -------------------------------------------------------

TEST(ColorThreshold, PureBlueImageGivesWhiteMask) {
    // HSVで青 (H=110, S=200, V=200) の単色画像を作る
    // blue閾値 [90,90,60]-[130,255,255] に収まるので、マスクは全白になるはず
    cv::Mat hsv(10, 10, CV_8UC3, cv::Scalar(110, 200, 200));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    cv::Mat mask = preprocessing::color_threshold(bgr, {90, 90, 60}, {130, 255, 255});

    // 全ピクセルが255 (白) であるか
    EXPECT_EQ(cv::countNonZero(mask), mask.rows * mask.cols);
}

TEST(ColorThreshold, BlackImageGivesBlackMask) {
    // 真っ黒な画像はどの閾値でもマスクが黒になる
    cv::Mat bgr(10, 10, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat mask = preprocessing::color_threshold(bgr, {0, 0, 50}, {180, 255, 255});
    EXPECT_EQ(cv::countNonZero(mask), 0);
}

// ---- images_to_pointcloud --------------------------------------------------

// テスト用カメラ射影行列 (fx=500, fy=500, cx=320, cy=240)
static Eigen::MatrixXd make_proj_matrix() {
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(3, 4);
    P(0, 0) = 500.0;  // fx
    P(1, 1) = 500.0;  // fy
    P(0, 2) = 320.0;  // cx
    P(1, 2) = 240.0;  // cy
    P(2, 2) = 1.0;
    return P;
}

TEST(ImagesToPointcloud, EmptyMaskGivesZeroRows) {
    // 全黒マスク → 有効ピクセルなし → 0行のMatrixXd
    cv::Mat rgb(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat depth(480, 640, CV_16UC1, cv::Scalar(1000));  // 全ピクセル1m
    cv::Mat mask(480, 640, CV_8UC1, cv::Scalar(0));       // 全黒

    Eigen::MatrixXd X = preprocessing::images_to_pointcloud(rgb, depth, make_proj_matrix(), mask);
    EXPECT_EQ(X.rows(), 0);
}

TEST(ImagesToPointcloud, CenterPixelProjectsCorrectly) {
    // pixel (320, 240) depth=1000mm → (0, 0, 1.0) m
    // 主点の画素なので X=0, Y=0, Z=1.0 になるはず
    cv::Mat rgb(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat depth(480, 640, CV_16UC1, cv::Scalar(0));
    depth.at<uint16_t>(240, 320) = 1000;  // 1点だけ1000mm

    cv::Mat mask(480, 640, CV_8UC1, cv::Scalar(0));
    mask.at<uchar>(240, 320) = 255;  // その1点だけ有効

    // leaf_size=0 でダウンサンプリングなし
    Eigen::MatrixXd X = preprocessing::images_to_pointcloud(rgb, depth, make_proj_matrix(), mask, 0.0);

    ASSERT_EQ(X.rows(), 1);
    EXPECT_NEAR(X(0, 0), 0.0, 1e-6);   // X = (320-320)*1.0/500 = 0
    EXPECT_NEAR(X(0, 1), 0.0, 1e-6);   // Y = (240-240)*1.0/500 = 0
    EXPECT_NEAR(X(0, 2), 1.0, 1e-6);   // Z = 1000/1000 = 1.0
}

TEST(ImagesToPointcloud, OffCenterPixelProjectsCorrectly) {
    // pixel (321, 240) depth=1000mm
    // X = (321-320)*1.0/500 = 0.002
    // Y = (240-240)*1.0/500 = 0
    // Z = 1.0
    cv::Mat rgb(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat depth(480, 640, CV_16UC1, cv::Scalar(0));
    depth.at<uint16_t>(240, 321) = 1000;

    cv::Mat mask(480, 640, CV_8UC1, cv::Scalar(0));
    mask.at<uchar>(240, 321) = 255;

    Eigen::MatrixXd X = preprocessing::images_to_pointcloud(rgb, depth, make_proj_matrix(), mask, 0.0);

    ASSERT_EQ(X.rows(), 1);
    EXPECT_NEAR(X(0, 0), 0.002, 1e-6);
    EXPECT_NEAR(X(0, 1), 0.0,   1e-6);
    EXPECT_NEAR(X(0, 2), 1.0,   1e-6);
}
