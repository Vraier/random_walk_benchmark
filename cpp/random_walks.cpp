#include <torch/extension.h>
#include <vector>
#include <random>
#include <omp.h>

torch::Tensor walk_cpp_impl(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor start_nodes,
    int walk_length,
    int seed
) {
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

        walks_ptr[i * walk_length + 0] = curr;
        for (int k = 1; k < walk_length; k++) {
            int64_t row_start = rowptr_ptr[curr];
            int64_t row_end = rowptr_ptr[curr + 1];
            int64_t degree = row_end - row_start;

            if (degree == 0) {
                walks_ptr[i * walk_length + k] = curr;
                continue;
            }

            std::uniform_int_distribution<int64_t> dist(0, degree - 1);
            int64_t offset = dist(rng);
            curr = col_ptr[row_start + offset];
            walks_ptr[i * walk_length + k] = curr;
        }
    }
    return walks;
}
