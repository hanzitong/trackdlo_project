#pragma once

// ============================================================
// node_utils.hpp
//
// trackdlo_node.cpp 専用のヘルパー関数群。
// ROS2メッセージ変換と文字列パースのみを担う。
// アルゴリズムロジックは持たない。
// ============================================================

#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <rclcpp/duration.hpp>

#include <Eigen/Dense>

#include <array>
#include <chrono>
#include <sstream>
#include <string>
#include <vector>

// ============================================================
// ノード座標行列 → RViz MarkerArray
//
// Y        : ノード座標 (M×3)
// vis      : 可視ノードのインデックス列。空なら全ノードを可視扱い
// node_rgba / edge_rgba : 可視ノード・エッジの色 {r, g, b, a}
// occ_*    : オクルージョン下のノード・エッジの色
// ============================================================
inline visualization_msgs::msg::MarkerArray make_marker_array(
    const Eigen::MatrixXd& Y,
    const std::string& frame_id,
    const std::string& ns,
    std::array<double, 4> node_rgba,
    std::array<double, 4> edge_rgba,
    double node_size = 0.01,
    double edge_width = 0.005,
    const std::vector<int>& vis = {},
    std::array<double, 4> occ_node_rgba = {1.0, 0.0, 0.0, 1.0},
    std::array<double, 4> occ_edge_rgba = {1.0, 0.0, 0.0, 1.0})
{
    visualization_msgs::msg::MarkerArray arr;
    // 1秒後に自動消去。次フレームの結果が上書きするので古いマーカーが残らない
    rclcpp::Duration lifetime = rclcpp::Duration(std::chrono::seconds(1));

    // ---- ノード (SPHERE_LIST: 1マーカーで全ノードをまとめて描く) ----
    visualization_msgs::msg::Marker node_marker;
    node_marker.header.frame_id = frame_id;
    node_marker.ns = ns + "_nodes";
    node_marker.id = 0;
    node_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    node_marker.action = visualization_msgs::msg::Marker::ADD;
    node_marker.scale.x = node_size;
    node_marker.scale.y = node_size;
    node_marker.scale.z = node_size;
    node_marker.lifetime = lifetime;

    for (int i = 0; i < Y.rows(); i++) {
        geometry_msgs::msg::Point p;
        p.x = Y(i, 0);  p.y = Y(i, 1);  p.z = Y(i, 2);
        node_marker.points.push_back(p);

        bool is_vis = vis.empty() ||
            std::find(vis.begin(), vis.end(), i) != vis.end();
        std_msgs::msg::ColorRGBA c;
        if (is_vis) {
            c.r = node_rgba[0];  c.g = node_rgba[1];
            c.b = node_rgba[2];  c.a = node_rgba[3];
        } else {
            c.r = occ_node_rgba[0];  c.g = occ_node_rgba[1];
            c.b = occ_node_rgba[2];  c.a = occ_node_rgba[3];
        }
        node_marker.colors.push_back(c);
    }
    arr.markers.push_back(node_marker);

    // ---- エッジ (LINE_LIST: 2点ペアを繰り返す) ----
    visualization_msgs::msg::Marker edge_marker;
    edge_marker.header.frame_id = frame_id;
    edge_marker.ns = ns + "_edges";
    edge_marker.id = 1;
    edge_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    edge_marker.action = visualization_msgs::msg::Marker::ADD;
    edge_marker.scale.x = edge_width;
    edge_marker.lifetime = lifetime;

    for (int i = 0; i < Y.rows() - 1; i++) {
        geometry_msgs::msg::Point p1, p2;
        p1.x = Y(i,   0);  p1.y = Y(i,   1);  p1.z = Y(i,   2);
        p2.x = Y(i+1, 0);  p2.y = Y(i+1, 1);  p2.z = Y(i+1, 2);
        edge_marker.points.push_back(p1);
        edge_marker.points.push_back(p2);

        bool i_vis   = vis.empty() || std::find(vis.begin(), vis.end(), i)   != vis.end();
        bool ip1_vis = vis.empty() || std::find(vis.begin(), vis.end(), i+1) != vis.end();
        std_msgs::msg::ColorRGBA c;
        if (i_vis && ip1_vis) {
            c.r = edge_rgba[0];  c.g = edge_rgba[1];
            c.b = edge_rgba[2];  c.a = edge_rgba[3];
        } else {
            c.r = occ_edge_rgba[0];  c.g = occ_edge_rgba[1];
            c.b = occ_edge_rgba[2];  c.a = occ_edge_rgba[3];
        }
        // LINE_LIST は各点ごとに色を指定する
        edge_marker.colors.push_back(c);
        edge_marker.colors.push_back(c);
    }
    arr.markers.push_back(edge_marker);

    return arr;
}

// ============================================================
// 対応点 priors → RViz MarkerArray (点のみ)
// priors の各要素は [node_idx, x, y, z] の 1×4 行列
// ============================================================
inline visualization_msgs::msg::MarkerArray make_priors_marker_array(
    const std::vector<Eigen::MatrixXd>& priors,
    const std::string& frame_id,
    const std::string& ns,
    std::array<double, 4> rgba,
    double size = 0.008)
{
    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker m;
    m.header.frame_id = frame_id;
    m.ns = ns;
    m.id = 0;
    m.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = m.scale.y = m.scale.z = size;
    m.lifetime = rclcpp::Duration(std::chrono::seconds(1));

    std_msgs::msg::ColorRGBA c;
    c.r = rgba[0];  c.g = rgba[1];  c.b = rgba[2];  c.a = rgba[3];

    for (const auto& prior : priors) {
        geometry_msgs::msg::Point p;
        p.x = prior(0, 1);  p.y = prior(0, 2);  p.z = prior(0, 3);
        m.points.push_back(p);
        m.colors.push_back(c);
    }
    arr.markers.push_back(m);
    return arr;
}

// ============================================================
// スペース区切り文字列 "90 90 30" → {90, 90, 30}
// ============================================================
inline std::vector<int> parse_hsv_string(const std::string& s)
{
    std::vector<int> result;
    std::istringstream ss(s);
    int v;
    while (ss >> v) result.push_back(v);
    return result;
}
