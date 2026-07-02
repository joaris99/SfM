#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bundle_adjustment.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cpp_ba, m) {

    py::class_<Observation>(m, "Observation")
        .def(py::init<>())
        .def_readwrite("camera_index", &Observation::camera_index)
        .def_readwrite("point_index", &Observation::point_index)
        .def_readwrite("x", &Observation::x)
        .def_readwrite("y", &Observation::y);

    m.def("bundle_adjustment", &bundle_adjustment);

    py::class_<BAResult>(m, "BAResult")
    .def_readonly("cameras", &BAResult::cameras)
    .def_readonly("points", &BAResult::points);
}


