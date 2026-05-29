


#include <trackdlo.h>
#include <utils.h>

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <numeric>



// N: the number of point cloud
// X: matrix of all generated point cloud
static Eigen::MatrixXd make_cable_cloud(int N, double bend_y, unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 0.005);

    Eigen::MatrixXd X(N, 3);
    for (int i = 0; i < N; i++)
    {
        double t = static_cast<double>(i) / (N - 1);
        X(i, 0) = t + noise(rng);
        X(i, 1) = bend_y * std::sin(t * M_PI) + noise(rng);
        X(i, 2) = 1. + noise(rng);
    }

    return X;
}


static trackdlo::TrackdloState initialize(const Eigen::MatrixXd& X, int M)
{
    trackdlo::TrackdloState state = trackdlo::make_trackdlo_stete(M);

    Eigen::RowVectorXd x_min = X.colwise().minCoeff();
    Eigen::RoWVectorXd x_max = X.colwise().maxCoeff();
    for (int i = 0; i < M; i++)
    {
        double t = static_cast<double>(i) / (M - 1);
        state.Y.row(i) = x_min + t * (x_max - x_min);
    }

}


