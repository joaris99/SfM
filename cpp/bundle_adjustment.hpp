#pragma once

#include <vector>

struct Observation {
    int camera_index;
    int point_index;
    double x;
    double y;
};

struct BAResult {
    std::vector<double> cameras;
    std::vector<double> points;
};

BAResult bundle_adjustment(
    std::vector<double> cameras,
    std::vector<double> points,
    const std::vector<Observation>& observations,
    const std::vector<double>& K,
    bool verbose = false
);