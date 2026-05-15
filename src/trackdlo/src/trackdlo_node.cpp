// ============================================================
// trackdlo_node.cpp  —  ROS2薄ラッパー
//
// 役割: カメラトピックを受け取り、pure_trackdlo と preprocessing を呼んで
//       トラッキング結果をパブリッシュするだけ。アルゴリズムは一切持たない。
//
// 依存:
//   - pure_trackdlo  (namespace trackdlo)  → tracking_step(), cpd_lle() 等
//   - preprocessing  (namespace preprocessing) → color_threshold(), images_to_pointcloud(), compute_visible_nodes()
// ============================================================

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <trackdlo.h>
#include <preprocessing.h>

#include "trackdlo_node_utils.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <string>
#include <vector>

// ============================================================
// ROS2ノード本体
// ============================================================
class TrackdloNode : public rclcpp::Node {
public:
    TrackdloNode() : Node("trackdlo_node")
    {
        // ---- パラメータ宣言とロード ----
        // アルゴリズムパラメータ
        this->declare_parameter("beta",            0.35);
        this->declare_parameter("lambda",          50000.0);
        this->declare_parameter("alpha",           3.0);
        this->declare_parameter("mu",              0.1);
        this->declare_parameter("max_iter",        50);
        this->declare_parameter("tol",             0.0002);
        this->declare_parameter("k_vis",           50.0);
        this->declare_parameter("visibility_threshold", 0.008);
        this->declare_parameter("beta_pre_proc",   3.0);
        this->declare_parameter("lambda_pre_proc", 1.0);
        this->declare_parameter("lle_weight",      10.0);

        // 前処理パラメータ
        this->declare_parameter("d_vis",           0.06);
        this->declare_parameter("dlo_pixel_width", 40);
        this->declare_parameter("downsample_leaf_size", 0.008);
        this->declare_parameter("multi_color_dlo", false);

        // HSV閾値 (スペース区切り文字列で受け取る)
        this->declare_parameter("hsv_threshold_upper_limit", std::string("130 255 255"));
        this->declare_parameter("hsv_threshold_lower_limit", std::string("90 90 30"));

        // トピック / フレーム
        this->declare_parameter("camera_info_topic", std::string("/camera/aligned_depth_to_color/camera_info"));
        this->declare_parameter("rgb_topic",          std::string("/camera/color/image_raw"));
        this->declare_parameter("depth_topic",        std::string("/camera/aligned_depth_to_color/image_raw"));
        this->declare_parameter("result_frame_id",    std::string("camera_color_optical_frame"));

        // パラメータを構造体に詰める
        params_.beta            = this->get_parameter("beta").as_double();
        params_.lambda          = this->get_parameter("lambda").as_double();
        params_.alpha           = this->get_parameter("alpha").as_double();
        params_.mu              = this->get_parameter("mu").as_double();
        params_.max_iter        = this->get_parameter("max_iter").as_int();
        params_.tol             = this->get_parameter("tol").as_double();
        params_.k_vis           = this->get_parameter("k_vis").as_double();
        params_.visibility_threshold = this->get_parameter("visibility_threshold").as_double();
        params_.beta_pre_proc   = this->get_parameter("beta_pre_proc").as_double();
        params_.lambda_pre_proc = this->get_parameter("lambda_pre_proc").as_double();
        params_.lle_weight      = this->get_parameter("lle_weight").as_double();

        d_vis_              = this->get_parameter("d_vis").as_double();
        dlo_pixel_width_    = this->get_parameter("dlo_pixel_width").as_int();
        downsample_leaf_size_ = this->get_parameter("downsample_leaf_size").as_double();
        multi_color_dlo_    = this->get_parameter("multi_color_dlo").as_bool();
        result_frame_id_    = this->get_parameter("result_frame_id").as_string();

        hsv_upper_ = parse_hsv_string(this->get_parameter("hsv_threshold_upper_limit").as_string());
        hsv_lower_ = parse_hsv_string(this->get_parameter("hsv_threshold_lower_limit").as_string());

        std::string camera_info_topic = this->get_parameter("camera_info_topic").as_string();
        std::string rgb_topic         = this->get_parameter("rgb_topic").as_string();
        std::string depth_topic       = this->get_parameter("depth_topic").as_string();

        // ---- サブスクライバー ----
        // 初期ノード: initialize.py が /trackdlo/init_nodes にパブリッシュするPointCloud2
        init_nodes_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/trackdlo/init_nodes", 1,
            std::bind(&TrackdloNode::on_init_nodes, this, std::placeholders::_1));

        // カメラ内部パラメータ: 一度受け取れば十分
        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            camera_info_topic, 1,
            std::bind(&TrackdloNode::on_camera_info, this, std::placeholders::_1));

        // シミュレーション用オクルージョンマスク (オプション)
        occ_mask_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/mask_with_occlusion", 10,
            std::bind(&TrackdloNode::on_occ_mask, this, std::placeholders::_1));

        // RGB + Depth を時刻同期して受け取る
        // ApproximateTime: 厳密な同一タイムスタンプを要求せず、近い時刻のペアをマッチする
        // → 実機カメラでは aligned_depth でも数ms のずれが生じるため Approximate を使う
        auto sensor_qos = rclcpp::SensorDataQoS();
        rgb_sub_.subscribe(this, rgb_topic,   sensor_qos.get_rmw_qos_profile());
        depth_sub_.subscribe(this, depth_topic, sensor_qos.get_rmw_qos_profile());

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), rgb_sub_, depth_sub_);
        sync_->registerCallback(
            std::bind(&TrackdloNode::on_image_pair, this,
                      std::placeholders::_1, std::placeholders::_2));

        // ---- パブリッシャー ----
        results_pub_     = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/trackdlo/results_marker", 30);
        guide_nodes_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/trackdlo/guide_nodes", 30);
        corr_priors_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/trackdlo/corr_priors", 30);
        results_img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/trackdlo/results_img", 30);
        pc_pub_          = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/trackdlo/filtered_pointcloud", 30);
        result_pc_pub_   = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/trackdlo/results_pc", 30);

        RCLCPP_INFO(this->get_logger(), "TrackdloNode initialized. Waiting for init_nodes and camera_info...");
    }

private:
    // ---- アルゴリズム状態 ----
    trackdlo::TrackdloState  state_;
    trackdlo::TrackdloParams params_;
    Eigen::MatrixXd proj_matrix_{Eigen::MatrixXd::Zero(3, 4)};
    bool initialized_{false};
    bool received_init_nodes_{false};
    bool received_proj_matrix_{false};
    Eigen::MatrixXd init_nodes_;
    cv::Mat occ_mask_;

    // ---- 前処理パラメータ (TrackdloParams に含まれないもの) ----
    double d_vis_;
    int    dlo_pixel_width_;
    double downsample_leaf_size_;
    bool   multi_color_dlo_;
    std::string result_frame_id_;
    std::vector<int> hsv_lower_, hsv_upper_;

    // ---- サブスクライバー ----
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr init_nodes_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr  camera_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr       occ_mask_sub_;

    // message_filters: 時刻同期用
    // ApproximateTime: 近い時刻のメッセージをペアにする (ExactTime より実用的)
    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // ---- パブリッシャー ----
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr results_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr guide_nodes_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr corr_priors_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr              results_img_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr        pc_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr        result_pc_pub_;

    // ---- コールバック ----

    // 初期ノード受信: initialize.py (or initialize_node) からの一回限りのメッセージ
    void on_init_nodes(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        pcl::fromROSMsg(*msg, cloud);

        // PCL行列 (4×N: x,y,z,padding) の上3行を転置して (N×3) にする
        init_nodes_ = cloud.getMatrixXfMap().topRows(3).transpose().cast<double>();

        // 無効な点 (z=0) を除去する
        // ROS1の参照実装では除去していなかったが、深度が取れていない点は除くほうが安全
        std::vector<int> valid_rows;
        for (int i = 0; i < init_nodes_.rows(); i++) {
            if (init_nodes_.row(i).norm() > 1e-6) valid_rows.push_back(i);
        }
        Eigen::MatrixXd filtered(valid_rows.size(), 3);
        for (int i = 0; i < static_cast<int>(valid_rows.size()); i++) {
            filtered.row(i) = init_nodes_.row(valid_rows[i]);
        }
        init_nodes_ = filtered;

        received_init_nodes_ = true;
        // 一度受け取ったらもう購読しなくてよい
        init_nodes_sub_.reset();

        RCLCPP_INFO(this->get_logger(), "Received init_nodes: %ld nodes", init_nodes_.rows());
        try_initialize();
    }

    // カメラ内部行列受信: 一度取れれば十分
    void on_camera_info(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        // CameraInfo::P は row-major の 3×4 射影行列
        for (int i = 0; i < 12; i++) {
            proj_matrix_(i / 4, i % 4) = msg->p[i];
        }
        received_proj_matrix_ = true;
        camera_info_sub_.reset();

        RCLCPP_INFO(this->get_logger(), "Received camera_info (fx=%.1f, fy=%.1f)",
                    proj_matrix_(0,0), proj_matrix_(1,1));
        try_initialize();
    }

    // シミュレーション用オクルージョンマスク (任意)
    void on_occ_mask(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            occ_mask_ = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_WARN(this->get_logger(), "occ_mask conversion failed: %s", e.what());
        }
    }

    // 初期ノードとカメラ行列が揃ったら TrackdloState を初期化する
    void try_initialize()
    {
        if (!received_init_nodes_ || !received_proj_matrix_) return;
        if (initialized_) return;

        state_ = trackdlo::make_trackdlo_state(init_nodes_.rows());
        state_.Y      = init_nodes_;
        state_.sigma2 = 0.001;

        // 測地線座標: 各ノードまでの累積弧長
        state_.geodesic_coord = {0.0};
        for (int i = 0; i < init_nodes_.rows() - 1; i++) {
            double seg = (init_nodes_.row(i+1) - init_nodes_.row(i)).norm();
            state_.geodesic_coord.push_back(state_.geodesic_coord.back() + seg);
        }

        initialized_ = true;
        RCLCPP_INFO(this->get_logger(),
                    "Tracker initialized with %ld nodes. Ready.", init_nodes_.rows());
    }

    // メインコールバック: RGB + Depth が揃うたびに呼ばれる
    void on_image_pair(const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
                       const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg)
    {
        if (!initialized_) return;

        // ---- 画像変換 ----
        cv::Mat bgr, depth;
        try {
            bgr   = cv_bridge::toCvShare(rgb_msg,   "bgr8")->image;
            depth = cv_bridge::toCvShare(depth_msg, depth_msg->encoding)->image;
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_WARN(this->get_logger(), "cv_bridge error: %s", e.what());
            return;
        }

        // ---- 色閾値処理 → 2値マスク ----
        cv::Mat mask;
        if (multi_color_dlo_) {
            mask = preprocessing::color_threshold_multicolor(bgr);
        } else {
            mask = preprocessing::color_threshold(bgr, hsv_lower_, hsv_upper_);
        }

        // ---- 点群生成 ----
        Eigen::MatrixXd X = preprocessing::images_to_pointcloud(
            bgr, depth, proj_matrix_, mask, downsample_leaf_size_,
            occ_mask_.empty() ? cv::Mat() : occ_mask_);

        if (X.rows() == 0) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Empty point cloud, skipping frame");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Point cloud: %ld pts", X.rows());

        // ---- 可視ノード計算 ----
        std::vector<int> visible_nodes, visible_nodes_extended;
        preprocessing::compute_visible_nodes(
            state_.Y, X, proj_matrix_, state_.geodesic_coord,
            bgr.rows, bgr.cols,
            params_.visibility_threshold, d_vis_, dlo_pixel_width_,
            visible_nodes, visible_nodes_extended);

        if (visible_nodes.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "No visible nodes, skipping frame");
            return;
        }

        // ---- トラッキング ----
        auto t0 = std::chrono::steady_clock::now();
        trackdlo::tracking_step(state_, X, visible_nodes, visible_nodes_extended, params_);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
        RCLCPP_INFO(this->get_logger(), "tracking_step: %.1f ms", ms);

        // ---- パブリッシュ ----
        auto stamp = rgb_msg->header.stamp;
        publish_markers(stamp, visible_nodes_extended);
        publish_images(bgr, stamp, visible_nodes_extended);
        publish_pointclouds(X, stamp);
    }

    // トラッキング結果をMarkerArrayでパブリッシュ
    void publish_markers(const rclcpp::Time& stamp, const std::vector<int>& vis)
    {
        // メイン結果: 可視ノードをオレンジ、オクルージョン下を赤で表示
        auto results = make_marker_array(
            state_.Y, result_frame_id_, "trackdlo_result",
            {1.0, 150.0/255.0, 0.0, 1.0},  // オレンジ
            {0.0, 1.0,         0.0, 1.0},  // 緑エッジ
            0.01, 0.005, vis,
            {1.0, 0.0, 0.0, 1.0},           // 赤 (オクルージョンノード)
            {1.0, 0.0, 0.0, 1.0});
        for (auto& m : results.markers) m.header.stamp = stamp;
        results_pub_->publish(results);

        // ガイドノード (内部デバッグ用)
        auto guide = make_marker_array(
            state_.guide_nodes, result_frame_id_, "guide_nodes",
            {0.0, 0.0, 0.0, 0.5}, {0.0, 0.0, 1.0, 0.5}, 0.008, 0.003);
        for (auto& m : guide.markers) m.header.stamp = stamp;
        guide_nodes_pub_->publish(guide);

        // 対応点 priors (内部デバッグ用)
        auto priors = make_priors_marker_array(
            state_.correspondence_priors, result_frame_id_, "corr_priors",
            {1.0, 0.0, 0.0, 0.5});
        for (auto& m : priors.markers) m.header.stamp = stamp;
        corr_priors_pub_->publish(priors);
    }

    // トラッキング結果をカメラ画像上に描画してパブリッシュ
    void publish_images(const cv::Mat& bgr,
                        const rclcpp::Time& stamp,
                        const std::vector<int>& vis)
    {
        // ノードを画像座標に投影する
        // Y_h = [Y | 1] (M×4), image_coords = P * Y_h^T → 各列が [u*w, v*w, w]
        Eigen::MatrixXd Y_h = state_.Y.replicate(1, 1);
        Y_h.conservativeResize(Y_h.rows(), Y_h.cols() + 1);
        Y_h.col(Y_h.cols() - 1) = Eigen::MatrixXd::Ones(Y_h.rows(), 1);
        Eigen::MatrixXd img_coords = (proj_matrix_ * Y_h.transpose()).transpose();

        cv::Mat vis_img = bgr.clone();
        for (int i = 0; i < state_.Y.rows() - 1; i++) {
            int x1 = static_cast<int>(img_coords(i,   0) / img_coords(i,   2));
            int y1 = static_cast<int>(img_coords(i,   1) / img_coords(i,   2));
            int x2 = static_cast<int>(img_coords(i+1, 0) / img_coords(i+1, 2));
            int y2 = static_cast<int>(img_coords(i+1, 1) / img_coords(i+1, 2));

            bool i_vis   = std::find(vis.begin(), vis.end(), i)   != vis.end();
            bool ip1_vis = std::find(vis.begin(), vis.end(), i+1) != vis.end();
            cv::Scalar line_color = (i_vis && ip1_vis)
                ? cv::Scalar(0, 255, 0)    // 緑: 可視エッジ
                : cv::Scalar(0, 0, 255);   // 赤: オクルージョン下エッジ

            cv::line(vis_img, {x1, y1}, {x2, y2}, line_color, 3);

            cv::Scalar nc1 = i_vis   ? cv::Scalar(0, 150, 255) : cv::Scalar(0, 0, 255);
            cv::Scalar nc2 = ip1_vis ? cv::Scalar(0, 150, 255) : cv::Scalar(0, 0, 255);
            cv::circle(vis_img, {x1, y1}, 5, nc1, -1);
            cv::circle(vis_img, {x2, y2}, 5, nc2, -1);
        }

        sensor_msgs::msg::Image::SharedPtr out =
            cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", vis_img).toImageMsg();
        out->header.stamp = stamp;
        out->header.frame_id = result_frame_id_;
        results_img_pub_->publish(*out);
    }

    // フィルタ済み点群と結果点群をパブリッシュ
    void publish_pointclouds(const Eigen::MatrixXd& X, const rclcpp::Time& stamp)
    {
        // フィルタ済み入力点群
        pcl::PointCloud<pcl::PointXYZ> x_cloud;
        for (int i = 0; i < X.rows(); i++) {
            x_cloud.emplace_back(
                static_cast<float>(X(i,0)),
                static_cast<float>(X(i,1)),
                static_cast<float>(X(i,2)));
        }
        sensor_msgs::msg::PointCloud2 x_msg;
        pcl::toROSMsg(x_cloud, x_msg);
        x_msg.header.stamp    = stamp;
        x_msg.header.frame_id = result_frame_id_;
        pc_pub_->publish(x_msg);

        // トラッキング結果点群 (評価用)
        pcl::PointCloud<pcl::PointXYZ> y_cloud;
        for (int i = 0; i < state_.Y.rows(); i++) {
            y_cloud.emplace_back(
                static_cast<float>(state_.Y(i,0)),
                static_cast<float>(state_.Y(i,1)),
                static_cast<float>(state_.Y(i,2)));
        }
        sensor_msgs::msg::PointCloud2 y_msg;
        pcl::toROSMsg(y_cloud, y_msg);
        y_msg.header.stamp    = stamp;
        y_msg.header.frame_id = result_frame_id_;
        result_pc_pub_->publish(y_msg);
    }
};

// ============================================================
// エントリーポイント
// ============================================================
int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrackdloNode>());
    rclcpp::shutdown();
    return 0;
}
