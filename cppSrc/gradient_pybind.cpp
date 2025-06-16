#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gradient.h"

namespace py = pybind11;

PYBIND11_MODULE(mygrad , m) {
    m.def("gradient_1d_order2" , &gradient_1d_order2 , "Compute gradient of 1D array (order 2)" ,
        py::arg("f") , py::arg("dx") = 1.0);

    m.def("gradient_1d_order4" , &gradient_1d_order4 , "Compute gradient of 1D array (order 4)" ,
        py::arg("f") , py::arg("dx") = 1.0);

    m.def("gradient_1d_order6" , &gradient_1d_order6 , "Compute gradient of 1D array (order 6)" ,
        py::arg("f") , py::arg("dx") = 1.0);
}
