#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <cstdint>

namespace py = pybind11;

py::array_t<int64_t> walk_impl(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rowptr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> col,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> start_nodes,
    int64_t walk_length,
    bool allow_backtrack
) {
    auto rp = rowptr.unchecked<1>();
    auto co = col.unchecked<1>();
    auto st = start_nodes.unchecked<1>();
    int64_t n_walks = start_nodes.size();

    auto result = py::array_t<int64_t>({n_walks, walk_length});
    auto res = result.mutable_unchecked<2>();

    std::mt19937 rng(42);

    for (int64_t i = 0; i < n_walks; i++) {
        int64_t curr = st(i);
        int64_t prev = -1;
        res(i, 0) = curr;
        for (int64_t k = 1; k < walk_length; k++) {
            int64_t rs = rp(curr);
            int64_t re = rp(curr + 1);
            int64_t degree = re - rs;
            int64_t nxt;

            if (degree == 0) {
                res(i, k) = curr;
                continue;
            } else if (!allow_backtrack && prev >= 0 && degree > 1) {
                std::uniform_int_distribution<int64_t> dist(0, degree - 2);
                int64_t idx = dist(rng);
                int64_t count = 0;
                nxt = curr;
                for (int64_t j = rs; j < re; j++) {
                    if (co(j) == prev) continue;
                    if (count == idx) { nxt = co(j); break; }
                    count++;
                }
            } else {
                std::uniform_int_distribution<int64_t> dist(rs, re - 1);
                nxt = co(dist(rng));
            }

            prev = curr;
            curr = nxt;
            res(i, k) = curr;
        }
    }
    return result;
}

PYBIND11_MODULE(MODULE_NAME, m) {
    m.def("walk_impl", &walk_impl, "Sequential random walk via pybind11");
}
