#include "bundle_adjustment.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

struct ReprojectionError {
    ReprojectionError(double x, double y)
        : observed_x(x), observed_y(y) {}

    template <typename T>
    bool operator()(const T* const q,
                    const T* const t,
                    const T* const point,
                    T* residuals) const {

        T p[3];

        ceres::QuaternionRotatePoint(q, point, p);

        p[0] += t[0];
        p[1] += t[1];
        p[2] += t[2];

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        residuals[0] = xp - T(observed_x);
        residuals[1] = yp - T(observed_y);

        return true;
    }

    double observed_x;
    double observed_y;
};

BAResult bundle_adjustment(
    std::vector<double> cameras,
    std::vector<double> points,
    const std::vector<Observation>& observations
) {
    ceres::Problem problem;

    for (const auto& obs : observations) {

        double* camera = cameras.data() + obs.camera_index * 7;

        double* q = camera;
        double* t = camera + 4;

        double* point = points.data() + obs.point_index * 3;

        auto* cost_function =
            new ceres::AutoDiffCostFunction<
                ReprojectionError,
                2,
                4,
                3,
                3>(
                new ReprojectionError(obs.x, obs.y));

        problem.AddResidualBlock(
            cost_function,
            nullptr,
            q,
            t,
            point
        );

        problem.SetManifold(
            q,
            new ceres::QuaternionManifold()
        );
    }

    // fix first camera
    problem.SetParameterBlockConstant(cameras.data());
    problem.SetParameterBlockConstant(cameras.data() + 4);

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    BAResult result;
    result.cameras = cameras;
    result.points = points;

    return result;
}