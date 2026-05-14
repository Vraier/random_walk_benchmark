#include <torch/extension.h>
#include <vector>
#include <random>
#include <omp.h>

torch::Tensor walk_cpp_impl(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor start_nodes,
    int walk_length,
    int seed,
    int num_threads,
    bool allow_backtrack
) {
    omp_set_num_threads(num_threads);

    auto rowptr_c = rowptr.contiguous();
    auto col_c = col.contiguous();
    auto start_nodes_c = start_nodes.contiguous();

    int64_t* rowptr_ptr = rowptr_c.data_ptr<int64_t>();
    int64_t* col_ptr = col_c.data_ptr<int64_t>();
    int64_t* starts_ptr = start_nodes_c.data_ptr<int64_t>();

    int64_t num_walks = start_nodes.size(0);

    auto options = torch::TensorOptions().dtype(torch::kInt64);
    torch::Tensor walks = torch::empty({num_walks, walk_length}, options);
    int64_t* walks_ptr = walks.data_ptr<int64_t>();

    #pragma omp parallel for
    for (int64_t i = 0; i < num_walks; i++) {
        std::mt19937 rng(seed + i);
        int64_t curr = starts_ptr[i];
        int64_t prev = -1;

        walks_ptr[i * walk_length] = curr;
        for (int k = 1; k < walk_length; k++) {
            int64_t rs = rowptr_ptr[curr];
            int64_t re = rowptr_ptr[curr + 1];
            int64_t degree = re - rs;
            int64_t nxt;

            if (degree == 0) {
                walks_ptr[i * walk_length + k] = curr;
                continue;
            } else if (!allow_backtrack && prev >= 0 && degree > 1) {
                std::uniform_int_distribution<int64_t> dist(0, degree - 2);
                int64_t idx = dist(rng);
                int64_t count = 0;
                nxt = curr;
                for (int64_t j = rs; j < re; j++) {
                    if (col_ptr[j] == prev) continue;
                    if (count == idx) { nxt = col_ptr[j]; break; }
                    count++;
                }
            } else {
                std::uniform_int_distribution<int64_t> dist(0, degree - 1);
                nxt = col_ptr[rs + dist(rng)];
            }

            prev = curr;
            curr = nxt;
            walks_ptr[i * walk_length + k] = curr;
        }
    }
    return walks;
}
