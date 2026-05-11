#include "../include/evaluator.h"
#include "../../pure_trackdlo/include/utils.h"

#include <iostream>
#include <fstream>
#include <stdexcept>

evaluator::evaluator () {}

evaluator::evaluator (int length, int trial, int pct_occlusion, std::string alg, int bag_file, std::string save_location,
                      double start_record_at, double exit_at, double wait_before_occlusion, double bag_rate, int num_of_nodes)
{
    length_ = length;
    trial_ = trial;
    pct_occlusion_ = pct_occlusion;
    alg_ = alg;
    bag_file_ = bag_file;
    save_location_ = save_location;
    start_record_at_ = start_record_at;
    exit_at_ = exit_at;
    wait_before_occlusion_ = wait_before_occlusion;
    cleared_file_ = false;
    bag_rate_ = bag_rate;
    num_of_nodes_ = num_of_nodes;
    image_counter_ = 0;
}

int evaluator::image_counter () {
    return image_counter_;
}

void evaluator::increment_image_counter () {
    image_counter_ += 1;
}

void evaluator::set_start_time (std::chrono::steady_clock::time_point cur_time) {
    start_time_ = cur_time;
}

std::chrono::steady_clock::time_point evaluator::start_time () {
    return start_time_;
}

double evaluator::pct_occlusion () {
    return pct_occlusion_;
}

double evaluator::recording_start_time () {
    return start_record_at_;
}

double evaluator::exit_time () {
    return exit_at_;
}

int evaluator::length () {
    return length_;
}

double evaluator::wait_before_occlusion () {
    return wait_before_occlusion_;
}

double evaluator::rate () {
    return bag_rate_;
}

// グラウンドトゥルースノードの並べ替え
// utils.h の sort_pts() とは異なり、head (開始端) を指定してチェーン方向を決める
Eigen::MatrixXd evaluator::sort_pts (Eigen::MatrixXd Y_0, Eigen::MatrixXd head) {
    int N = Y_0.rows();
    Eigen::MatrixXd Y_0_sorted = Eigen::MatrixXd::Zero(N, 3);
    std::vector<Eigen::MatrixXd> Y_0_sorted_vec = {};
    std::vector<bool> selected_node(N, false);
    selected_node[0] = true;
    int last_visited_b = 0;

    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            G(i, j) = (Y_0.row(i) - Y_0.row(j)).squaredNorm();
        }
    }

    int reverse = 0;
    int counter = 0;
    int reverse_on = 0;
    int insertion_counter = 0;

    while (counter < N-1) {
        double minimum = INFINITY;
        int a = 0;
        int b = 0;

        for (int m = 0; m < N; m++) {
            if (selected_node[m] == true) {
                for (int n = 0; n < N; n++) {
                    if ((!selected_node[n]) && (G(m, n) != 0.0)) {
                        if (minimum > G(m, n)) {
                            minimum = G(m, n);
                            a = m;
                            b = n;
                        }
                    }
                }
            }
        }

        if (counter == 0) {
            Y_0_sorted_vec.push_back(Y_0.row(a));
            Y_0_sorted_vec.push_back(Y_0.row(b));
        }
        else {
            if (last_visited_b != a) {
                reverse += 1;
                reverse_on = a;
                insertion_counter = 1;
            }

            if (reverse % 2 == 1) {
                auto it = find(Y_0_sorted_vec.begin(), Y_0_sorted_vec.end(), Y_0.row(a));
                Y_0_sorted_vec.insert(it, Y_0.row(b));
            }
            else if (reverse != 0) {
                auto it = find(Y_0_sorted_vec.begin(), Y_0_sorted_vec.end(), Y_0.row(reverse_on));
                Y_0_sorted_vec.insert(it + insertion_counter, Y_0.row(b));
                insertion_counter += 1;
            }
            else {
                Y_0_sorted_vec.push_back(Y_0.row(b));
            }
        }

        last_visited_b = b;
        selected_node[b] = true;
        counter += 1;
    }

    // head との距離が遠ければ逆順にする (開始端を head 側に合わせる)
    if (pt2pt_dis(Y_0_sorted_vec[0], head) > 0.08) {
        std::reverse(Y_0_sorted_vec.begin(), Y_0_sorted_vec.end());
    }

    for (int i = 0; i < N; i++) {
        Y_0_sorted.row(i) = Y_0_sorted_vec[i];
    }

    return Y_0_sorted;
}

// RGB画像と点群からマーカー色ノード (青・赤・黄) を検出する
// SimpleBlobDetector でマーカー位置を検出し、点群で3D座標に変換する
Eigen::MatrixXd evaluator::get_ground_truth_nodes (cv::Mat rgb_img, pcl::PointCloud<pcl::PointXYZRGB> cloud_xyz) {
    cv::Mat mask_blue, mask_red_1, mask_red_2, mask_red, mask_yellow, mask_markers;
    cv::Mat cur_image_hsv, mask_without_occlusion_block;

    cv::cvtColor(rgb_img, cur_image_hsv, cv::COLOR_BGR2HSV);

    std::vector<int> lower_blue   = {90,  80,  80};
    std::vector<int> upper_blue   = {130, 255, 255};
    std::vector<int> lower_red_1  = {130, 60,  50};
    std::vector<int> upper_red_1  = {255, 255, 255};
    std::vector<int> lower_red_2  = {0,   60,  50};
    std::vector<int> upper_red_2  = {10,  255, 255};
    std::vector<int> lower_yellow = {15,  100, 80};
    std::vector<int> upper_yellow = {40,  255, 255};

    cv::inRange(cur_image_hsv, cv::Scalar(lower_blue[0],   lower_blue[1],   lower_blue[2]),   cv::Scalar(upper_blue[0],   upper_blue[1],   upper_blue[2]),   mask_blue);
    cv::inRange(cur_image_hsv, cv::Scalar(lower_red_1[0],  lower_red_1[1],  lower_red_1[2]),  cv::Scalar(upper_red_1[0],  upper_red_1[1],  upper_red_1[2]),  mask_red_1);
    cv::inRange(cur_image_hsv, cv::Scalar(lower_red_2[0],  lower_red_2[1],  lower_red_2[2]),  cv::Scalar(upper_red_2[0],  upper_red_2[1],  upper_red_2[2]),  mask_red_2);
    cv::inRange(cur_image_hsv, cv::Scalar(lower_yellow[0], lower_yellow[1], lower_yellow[2]), cv::Scalar(upper_yellow[0], upper_yellow[1], upper_yellow[2]), mask_yellow);

    cv::bitwise_or(mask_red_1, mask_red_2, mask_red);
    cv::bitwise_or(mask_red, mask_blue, mask_without_occlusion_block);
    cv::bitwise_or(mask_yellow, mask_without_occlusion_block, mask_without_occlusion_block);
    cv::bitwise_or(mask_red, mask_yellow, mask_markers);

    // SimpleBlobDetector でマーカー候補ブロブを検出する
    cv::SimpleBlobDetector::Params blob_params;
    blob_params.filterByColor = false;
    blob_params.filterByArea = true;
    blob_params.minArea = 10;
    blob_params.filterByCircularity = false;
    blob_params.filterByInertia = true;
    blob_params.filterByConvexity = false;
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(blob_params);

    std::vector<cv::KeyPoint> keypoints_markers;
    detector->detect(mask_markers, keypoints_markers);

    pcl::PointCloud<pcl::PointXYZRGB> cur_nodes_xyz;

    for (const cv::KeyPoint& key_point : keypoints_markers) {
        auto keypoint_pc = cloud_xyz(static_cast<int>(key_point.pt.x), static_cast<int>(key_point.pt.y));

        // bag_file_ ごとのフィルタリング条件 (実験セットアップ依存)
        if (bag_file_ == 2) {
            if (keypoint_pc.x < -0.15 || keypoint_pc.y < -0.15 || keypoint_pc.z < 0.58) continue;
        }
        else if (bag_file_ == 1) {
            if ((keypoint_pc.x < 0.0 && keypoint_pc.y < 0.05) || keypoint_pc.z < 0.58 ||
                 keypoint_pc.x < -0.2 || (keypoint_pc.x < 0.1 && keypoint_pc.y < -0.05)) continue;
        }
        else {
            if (keypoint_pc.z < 0.58) continue;
        }

        cur_nodes_xyz.push_back(keypoint_pc);
    }

    // PCL点群の上位3行 (XYZ) を Eigen::MatrixXd に変換して返す
    return cur_nodes_xyz.getMatrixXfMap().topRows(3).transpose().cast<double>();
}

// 点Eから線分ABへの最短距離を計算する
// closest_pt_on_AB_to_E には線分上の最近傍点が格納される
double evaluator::calc_min_distance (Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd E,
                                     Eigen::MatrixXd& closest_pt_on_AB_to_E) {
    Eigen::MatrixXd AB = B - A;
    Eigen::MatrixXd AE = E - A;

    double distance = cross_product(AE, AB).norm() / AB.norm();
    closest_pt_on_AB_to_E = A + AB * dot_product(AE, AB) / dot_product(AB, AB);

    Eigen::MatrixXd AP = closest_pt_on_AB_to_E - A;
    // 最近傍点が線分外に出た場合は端点への距離を使う
    if (dot_product(AP, AB) < 0 || dot_product(AP, AB) > dot_product(AB, AB)) {
        Eigen::MatrixXd BE = E - B;
        double distance_AE = sqrt(dot_product(AE, AE));
        double distance_BE = sqrt(dot_product(BE, BE));
        if (distance_AE > distance_BE) {
            distance = distance_BE;
            closest_pt_on_AB_to_E = B.replicate(1, 1);
        }
        else {
            distance = distance_AE;
            closest_pt_on_AB_to_E = A.replicate(1, 1);
        }
    }

    return distance;
}

// Y_track の各点から Y_true の折れ線曲線への平均最短距離 (片方向誤差)
double evaluator::get_piecewise_error (Eigen::MatrixXd Y_track, Eigen::MatrixXd Y_true) {
    double total_distances_to_curve = 0.0;

    for (int idx = 0; idx < Y_track.rows(); idx++) {
        double dist = -1;
        Eigen::MatrixXd closest_pt = Eigen::MatrixXd::Zero(1, 3);

        for (int i = 0; i < Y_true.rows()-1; i++) {
            Eigen::MatrixXd closest_pt_i = Eigen::MatrixXd::Zero(1, 3);
            double dist_i = calc_min_distance(Y_true.row(i), Y_true.row(i+1), Y_track.row(idx), closest_pt_i);
            if (dist == -1 || dist_i < dist) {
                dist = dist_i;
                closest_pt = closest_pt_i.replicate(1, 1);
            }
        }

        total_distances_to_curve += dist;
    }

    return total_distances_to_curve / Y_track.rows();
}

// 双方向区分的誤差を計算してテキストファイルに追記する
double evaluator::compute_and_save_error (Eigen::MatrixXd Y_track, Eigen::MatrixXd Y_true) {
    double E1 = get_piecewise_error(Y_track, Y_true);
    double E2 = get_piecewise_error(Y_true, Y_track);
    double cur_frame_error = (E1 + E2) / 2;
    errors_.push_back(cur_frame_error);

    // 保存先ファイル名: bag_file_ の種別で切り替える
    std::string dir;
    if (bag_file_ == 0) {
        dir = save_location_ + alg_ + "_" + std::to_string(trial_) + "_" + std::to_string(pct_occlusion_) + "_stationary_error.txt";
    }
    else if (bag_file_ == 1) {
        dir = save_location_ + alg_ + "_" + std::to_string(trial_) + "_" + std::to_string(pct_occlusion_) + "_perpendicular_motion_error.txt";
    }
    else if (bag_file_ == 2) {
        dir = save_location_ + alg_ + "_" + std::to_string(trial_) + "_" + std::to_string(pct_occlusion_) + "_parallel_motion_error.txt";
    }
    else if (bag_file_ == 4) {
        dir = save_location_ + alg_ + "_" + std::to_string(trial_) + "_" + std::to_string(pct_occlusion_) + "_short_rope_folding_error.txt";
    }
    else if (bag_file_ == 5) {
        dir = save_location_ + alg_ + "_" + std::to_string(trial_) + "_" + std::to_string(pct_occlusion_) + "_short_rope_stationary_error.txt";
    }
    else {
        throw std::invalid_argument("Invalid bag file ID!");
    }

    double time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time_).count() / 1000.0 * bag_rate_;

    if (cleared_file_ == false) {
        std::ofstream error_list(dir);
        error_list << std::to_string(time_diff - start_record_at_) + " " + std::to_string(cur_frame_error) + "\n";
        error_list.close();
        cleared_file_ = true;
    }
    else {
        std::ofstream error_list(dir, std::fstream::app);
        error_list << std::to_string(time_diff - start_record_at_) + " " + std::to_string(cur_frame_error) + "\n";
        error_list.close();
    }

    return cur_frame_error;
}

// 双方向区分的誤差を計算して返す (ファイル保存なし)
double evaluator::compute_error (Eigen::MatrixXd Y_track, Eigen::MatrixXd Y_true) {
    double E1 = get_piecewise_error(Y_track, Y_true);
    double E2 = get_piecewise_error(Y_true, Y_track);
    return (E1 + E2) / 2;
}
