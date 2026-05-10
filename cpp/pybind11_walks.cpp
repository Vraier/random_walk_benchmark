#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <cstdint>

namespace py = pybind11;

py::array_t<int64_t> walk_impl(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rowptr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> col,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> start_nodes,
    int64_t walk_length
) {
    auto rp = rowptr.unchecked<1>();
    auto co = col.unchecked<1>();
    auto st = start_nodes.unchecked<1>();
    int64_t n_walks = start_nodes.size();

    auto result = py::array_t<int64_t>({n_walks, walk_length});
    auto res = result.mutable_unchecked<2>();

    std::mt19937 rng(42);

    for (int64_t i = 0; i < n_walks; i++) {
        int64_t node = st(i);
        res(i, 0) = node;
        for (int64_t k = 1; k < walk_length; k++) {
            int64_t rs = rp(node);
            int64_t re = rp(node + 1);
            if (rs == re) {
                res(i, k) = node;
            } else {
                std::uniform_int_distribution<int64_t> dist(rs, re - 1);
                node = co(dist(rng));
                res(i, k) = node;
            }
        }
    }
    return result;
}

PYBIND11_MODULE(MODULE_NAME, m) {
    m.def("walk_impl", &walk_impl, "Sequential random walk via pybind11");
}
