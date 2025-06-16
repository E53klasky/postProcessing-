#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gradient.h"

namespace py = pybind11;

PYBIND11_MODULE(mygrad , m) {
    m.def("gradient_1d_order2" , &gradient_1d_order2 ,
        "Compute 1D gradient (2nd order)" , py::arg("f") , py::arg("dx"));

    m.def("gradient_1d_order4" , &gradient_1d_order4 ,
        "Compute 1D gradient (4th order)" , py::arg("f") , py::arg("dx"));

    m.def("gradient_1d_order6" , &gradient_1d_order6 ,
        "Compute 1D gradient (6th order)" , py::arg("f") , py::arg("dx"));

    m.def("gradient_2d_order2" , &gradient_2d_order2 ,
        "Compute 2D gradient (2nd order)" , py::arg("f") , py::arg("dx") , py::arg("dy"));

    m.def("gradient_2d_order4" , &gradient_2d_order4 ,
        "Compute 2D gradient (4th order)" , py::arg("f") , py::arg("dx") , py::arg("dy"));

    m.def("gradient_2d_order6" , &gradient_2d_order6 ,
        "Compute 2D gradient (6th order)" , py::arg("f") , py::arg("dx") , py::arg("dy"));
}
