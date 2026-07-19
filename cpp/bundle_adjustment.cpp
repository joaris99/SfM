#include "bundle_adjustment.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

struct ReprojectionError {

    ReprojectionError(
        double x,
        double y,
        const std::vector<double>& K
    )
        :
        observed_x(x),
        observed_y(y),
        fx(K[0]),
        fy(K[4]),
        cx(K[2]),
        cy(K[5])
    {}

    template <typename T>
    bool operator()(
        const T* const q,
        const T* const t,
        const T* const point,
        T* residuals
    ) const {

        // Rotate point from world coordinates to camera coordinates
        T p[3];

        ceres::QuaternionRotatePoint(
            q,
            point,
            p
        );

        // Apply translation
        p[0] += t[0];
        p[1] += t[1];
        p[2] += t[2];


        // Normalize camera coordinates
        T x_normalized = p[0] / p[2];
        T y_normalized = p[1] / p[2];


        // Project to pixel coordinates using K
        T u = T(fx) * x_normalized + T(cx);
        T v = T(fy) * y_normalized + T(cy);


        // Pixel reprojection error
        residuals[0] = u - T(observed_x);
        residuals[1] = v - T(observed_y);

        return true;
    }


    double observed_x;
    double observed_y;

    double fx;
    double fy;
    double cx;
    double cy;
};

BAResult bundle_adjustment(
    std::vector<double> cameras,
    std::vector<double> points,
    const std::vector<Observation>& observations,
    const std::vector<double>& K,
    bool verbose
){
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
                new ReprojectionError(obs.x, obs.y, K));

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
    options.minimizer_progress_to_stdout = verbose;

    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
    if (verbose) {
        std::cout << summary.BriefReport() << std::endl;
    }

    BAResult result;
    result.cameras = cameras;
    result.points = points;

    return result;
}