#include "../include/trackdlo.h"
#include "../include/utils.h"

namespace trackdlo {

void signal_callback_handler(int signum) {
   exit(signum);
}

double pt2pt_dis_sq (Eigen::MatrixXd pt1, Eigen::MatrixXd pt2) {
    return (pt1 - pt2).rowwise().squaredNorm().sum();
}

double pt2pt_dis (Eigen::MatrixXd pt1, Eigen::MatrixXd pt2) {
    return (pt1 - pt2).rowwise().norm().sum();
}

// 簡易CPD登録: Y軸方向に均等配置したノードをEステップ・Mステップで収束させる
// trackdloクラスのcpd_lleと違い、LLEや対応点制約なし
void reg (Eigen::MatrixXd pts, Eigen::MatrixXd& Y, double& sigma2, int M, double mu, int max_iter) {
    // initial guess
    Eigen::MatrixXd X = pts.replicate(1, 1);
    Y = Eigen::MatrixXd::Zero(M, 3);
    for (int i = 0; i < M; i ++) {
        Y(i, 1) = 0.1 / static_cast<double>(M) * static_cast<double>(i);
        Y(i, 0) = 0;
        Y(i, 2) = 0;
    }

    int N = X.rows();
    int D = 3;

    // diff_xy should be a (M * N) matrix
    Eigen::MatrixXd diff_xy = Eigen::MatrixXd::Zero(M, N);
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < N; j ++) {
            diff_xy(i, j) = (Y.row(i) - X.row(j)).squaredNorm();
        }
    }

    // initialize sigma2
    sigma2 = diff_xy.sum() / static_cast<double>(D * M * N);

    for (int it = 0; it < max_iter; it ++) {
        // update diff_xy
        for (int i = 0; i < M; i ++) {
            for (int j = 0; j < N; j ++) {
                diff_xy(i, j) = (Y.row(i) - X.row(j)).squaredNorm();
            }
        }

        Eigen::MatrixXd P = (-0.5 * diff_xy / sigma2).array().exp();
        Eigen::MatrixXd P_stored = P.replicate(1, 1);
        double c = pow((2 * M_PI * sigma2), static_cast<double>(D)/2) * mu / (1 - mu) * static_cast<double>(M)/N;
        P = P.array().rowwise() / (P.colwise().sum().array() + c);

        Eigen::MatrixXd Pt1 = P.colwise().sum();
        Eigen::MatrixXd P1 = P.rowwise().sum();
        double Np = P1.sum();
        Eigen::MatrixXd PX = P * X;

        Eigen::MatrixXd P1_expanded = Eigen::MatrixXd::Zero(M, D);
        P1_expanded.col(0) = P1;
        P1_expanded.col(1) = P1;
        P1_expanded.col(2) = P1;

        Y = PX.cwiseQuotient(P1_expanded);

        double numerator = 0;
        double denominator = 0;

        for (int m = 0; m < M; m ++) {
            for (int n = 0; n < N; n ++) {
                numerator += P(m, n)*diff_xy(m, n);
                denominator += P(m, n)*D;
            }
        }

        sigma2 = numerator / denominator;
    }
}

// link to original code: https://stackoverflow.com/a/46303314
void remove_row(Eigen::MatrixXd& matrix, unsigned int rowToRemove) {
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.bottomRows(numRows-rowToRemove);

    matrix.conservativeResize(numRows,numCols);
}

Eigen::MatrixXd sort_pts (Eigen::MatrixXd Y_0) {
    int N = Y_0.rows();
    Eigen::MatrixXd Y_0_sorted = Eigen::MatrixXd::Zero(N, 3);
    std::vector<Eigen::MatrixXd> Y_0_sorted_vec = {};
    std::vector<bool> selected_node(N, false);
    selected_node[0] = true;
    int last_visited_b = 0;

    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(N, N);
    for (int i = 0; i < N; i ++) {
        for (int j = 0; j < N; j ++) {
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

        for (int m = 0; m < N; m ++) {
            if (selected_node[m] == true) {
                for (int n = 0; n < N; n ++) {
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

    // copy to Y_0_sorted
    for (int i = 0; i < N; i ++) {
        Y_0_sorted.row(i) = Y_0_sorted_vec[i];
    }

    return Y_0_sorted;
}

// 線分ABと球(中心sphere_center, 半径radius)の交点を返す
// Pure Pursuitのルックアヘッド距離計算に使用する
bool isBetween (Eigen::MatrixXd x, Eigen::MatrixXd a, Eigen::MatrixXd b) {
    bool in_bound = true;

    for (int i = 0; i < 3; i ++) {
        if (!(a(0, i)-0.0001 <= x(0, i) && x(0, i) <= b(0, i)+0.0001) &&
            !(b(0, i)-0.0001 <= x(0, i) && x(0, i) <= a(0, i)+0.0001)) {
            in_bound = false;
        }
    }

    return in_bound;
}

std::vector<Eigen::MatrixXd> line_sphere_intersection (Eigen::MatrixXd point_A, Eigen::MatrixXd point_B, Eigen::MatrixXd sphere_center, double radius) {
    std::vector<Eigen::MatrixXd> intersections = {};

    double a = pt2pt_dis_sq(point_A, point_B);
    double b = 2 * ((point_B(0, 0) - point_A(0, 0))*(point_A(0, 0) - sphere_center(0, 0)) +
                    (point_B(0, 1) - point_A(0, 1))*(point_A(0, 1) - sphere_center(0, 1)) +
                    (point_B(0, 2) - point_A(0, 2))*(point_A(0, 2) - sphere_center(0, 2)));
    double c = pt2pt_dis_sq(point_A, sphere_center) - pow(radius, 2);

    double delta = pow(b, 2) - 4*a*c;

    double d1 = (-b + sqrt(delta)) / (2*a);
    double d2 = (-b - sqrt(delta)) / (2*a);

    if (delta < 0) {
        return {};
    }
    else if (delta > 0) {
        // two solutions
        double x1 = point_A(0, 0) + d1*(point_B(0, 0) - point_A(0, 0));
        double y1 = point_A(0, 1) + d1*(point_B(0, 1) - point_A(0, 1));
        double z1 = point_A(0, 2) + d1*(point_B(0, 2) - point_A(0, 2));
        Eigen::MatrixXd pt1(1, 3);
        pt1 << x1, y1, z1;

        double x2 = point_A(0, 0) + d2*(point_B(0, 0) - point_A(0, 0));
        double y2 = point_A(0, 1) + d2*(point_B(0, 1) - point_A(0, 1));
        double z2 = point_A(0, 2) + d2*(point_B(0, 2) - point_A(0, 2));
        Eigen::MatrixXd pt2(1, 3);
        pt2 << x2, y2, z2;

        if (isBetween(pt1, point_A, point_B)) {
            intersections.push_back(pt1);
        }
        if (isBetween(pt2, point_A, point_B)) {
            intersections.push_back(pt2);
        }
    }
    else {
        // one solution
        d1 = -b / (2*a);
        double x1 = point_A(0, 0) + d1*(point_B(0, 0) - point_A(0, 0));
        double y1 = point_A(0, 1) + d1*(point_B(0, 1) - point_A(0, 1));
        double z1 = point_A(0, 2) + d1*(point_B(0, 2) - point_A(0, 2));
        Eigen::MatrixXd pt1(1, 3);
        pt1 << x1, y1, z1;

        if (isBetween(pt1, point_A, point_B)) {
            intersections.push_back(pt1);
        }
    }

    return intersections;
}

Eigen::MatrixXd cross_product (Eigen::MatrixXd vec1, Eigen::MatrixXd vec2) {
    Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(1, 3);

    ret(0, 0) = vec1(0, 1)*vec2(0, 2) - vec1(0, 2)*vec2(0, 1);
    ret(0, 1) = -(vec1(0, 0)*vec2(0, 2) - vec1(0, 2)*vec2(0, 0));
    ret(0, 2) = vec1(0, 0)*vec2(0, 1) - vec1(0, 1)*vec2(0, 0);

    return ret;
}

double dot_product (Eigen::MatrixXd vec1, Eigen::MatrixXd vec2) {
    return vec1(0, 0)*vec2(0, 0) + vec1(0, 1)*vec2(0, 1) + vec1(0, 2)*vec2(0, 2);
}

} // namespace trackdlo
