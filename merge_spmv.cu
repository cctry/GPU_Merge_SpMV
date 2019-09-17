#include "sparse_matrix.h"
#include "utils.h"
#include <cstdio>
#include <string>
#include <algorithm>

// Compute reference SpMV y = Ax
template <typename ValueT, typename OffsetT>
void SpmvGold(CsrMatrix<ValueT, OffsetT> &a, ValueT *vector_x,
              ValueT *vector_y_in, ValueT *vector_y_out) {
    for (OffsetT row = 0; row < a.num_rows; ++row) {
        ValueT partial = 0;
        for (OffsetT offset = a.row_offsets[row];
             offset < a.row_offsets[row + 1]; ++offset) {
            partial += a.values[offset] * vector_x[a.column_indices[offset]];
        }
        vector_y_out[row] = partial;
    }
}

/**
 * Display performance
 */
template <typename ValueT, typename OffsetT>
void DisplayPerf(float device_giga_bandwidth, double setup_ms, double avg_ms,
                 CsrMatrix<ValueT, OffsetT> &csr_matrix) {
    double nz_throughput, effective_bandwidth;
    size_t total_bytes =
        (csr_matrix.num_nonzeros * (sizeof(ValueT) * 2 + sizeof(OffsetT))) +
        (csr_matrix.num_rows) * (sizeof(OffsetT) + sizeof(ValueT));

    nz_throughput = double(csr_matrix.num_nonzeros) / avg_ms / 1.0e6;
    effective_bandwidth = double(total_bytes) / avg_ms / 1.0e6;

    printf("fp%d: %.4f setup ms, %.4f avg ms, %.5f gflops, %.3lf effective "
           "GB/s (%.2f%% peak)\n",
           sizeof(ValueT) * 8, setup_ms, avg_ms, 2 * nz_throughput,
           effective_bandwidth,
           effective_bandwidth / device_giga_bandwidth * 100);
}

template <typename ValueT, typename OffsetT>
CsrMatrix<ValueT, OffsetT> init_mat(const std::string &mtx_filename) {
    // Initialize matrix in COO form
    CooMatrix<ValueT, OffsetT> coo_matrix;

    if (!mtx_filename.empty()) {
        // Parse matrix market file
        coo_matrix.InitMarket(mtx_filename, 1.0, true);

        if ((coo_matrix.num_rows == 1) || (coo_matrix.num_cols == 1) ||
            (coo_matrix.num_nonzeros == 1)) {
            printf("Trivial dataset\n");
            exit(0);
        }
        printf("%s\n, ", mtx_filename.c_str());
    } else {
        fprintf(stderr, "No graph type specified.\n");
        exit(1);
    }
    CsrMatrix<ValueT, OffsetT> csr_matrix(coo_matrix);
    coo_matrix.Clear();
    return csr_matrix;
}

/**
 * Computes the begin offsets into A and B for the specific diagonal
 */
template <typename AIteratorT, typename BIteratorT, typename OffsetT,
          typename CoordinateT>
__device__ void MergePathSearch(
    OffsetT diagonal,             ///< [in]The diagonal to search
    AIteratorT a,                 ///< [in]List A
    BIteratorT b,                 ///< [in]List B
    OffsetT a_len,                ///< [in]Length of A
    OffsetT b_len,                ///< [in]Length of B
    CoordinateT &path_coordinate) ///< [out] (x,y) coordinate where diagonal
                                  ///< intersects the merge path
{
    OffsetT x_min = std::max(diagonal - b_len, 0);
    OffsetT x_max = std::min(diagonal, a_len);

    while (x_min < x_max) {
        OffsetT x_pivot = (x_min + x_max) >> 1;
        if (a[x_pivot] <= b[diagonal - x_pivot - 1])
            x_min = x_pivot + 1; // Contract range up A (down B)
        else
            x_max = x_pivot; // Contract range down A (up B)
    }

    path_coordinate.x = std::min(x_min, a_len);
    path_coordinate.y = diagonal - x_min;
}

/**
 * Kernal
 */
template <typename ValueT, typename OffsetT>
__global__ void merge_spmv_kernal(OffsetT num_rows, OffsetT num_cols,
                                  OffsetT num_nonzeros, OffsetT *d_row_offsets,
                                  OffsetT *d_column_indices, ValueT *d_values) {
}

template <typename ValueT, typename OffsetT>
void lauch(OffsetT num_rows, OffsetT num_cols, OffsetT num_nonzeros,
           OffsetT *d_row_offsets, OffsetT *d_column_indices,
           ValueT *d_values) {}

/**
 * Run tests
 */
template <typename ValueT, typename OffsetT>
void RunTest(const std::string &mtx_filename, CommandLineArgs &args) {
    bool g_quiet = false;
    auto mat = init_mat(mtx_filename);
    mat.Stats().Display(!g_quiet);
    mat.DisplayHistogram();
    printf("\n");
    // Allocate input and output vectors
    ValueT *vector_x = new ValueT[mat.num_cols];
    ValueT *vector_y_in = new ValueT[mat.num_rows];
    ValueT *vector_y_out = new ValueT[mat.num_rows];
    ValueT *reference_vector_y_out = new ValueT[mat.num_rows];
    // Init vertor x and y
    for (int col = 0; col < mat.num_cols; ++col)
        vector_x[col] = 1.0;
    for (int row = 0; row < mat.num_rows; ++row)
        vector_y_in[row] = 1.0;
    // Compute reference answer
    SpmvGold(mat, vector_x, vector_y_in, reference_vector_y_out);
    float setup_ms = 0.0;
    printf("%s, %s\n\n, Merge CsrMV, ", args.deviceProp.name,
           (sizeof(ValueT) > 4) ? "fp64" : "fp32");
    // Get GPU device bandwidth (GB/s)
    float device_giga_bandwidth = args.device_giga_bandwidth;

    memset(vector_y_out, -1, sizeof(ValueT) * mat.num_rows);
    // Set GPU memory
    OffsetT *d_row_end_offsets;
    OffsetT *d_column_indices;
    ValueT *d_value;
    cudaMalloc((void **)&d_row_end_offsets,
               sizeof(OffsetT) * mat.num_cols); // except 0
    cudaMemcpy(d_row_end_offsets, mat.row_offsets++,
               sizeof(OffsetT) * mat.num_cols, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_column_indices, sizeof(ValueT) * mat.num_nonzeros);
    cudaMemcpy(d_column_indices,
               mat.column_indices sizeof(ValueT) * mat.num_nonzeros,
               cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_value, sizeof(ValueT) * mat.num_nonzeros);
    cudaMemcpy(d_value, mat.value, sizeof(ValueT) * mat.num_nonzeros,
               cudaMemcpyHostToDevice);

    // Timing
    float elapsed_ms = 0.0;
    GpuTimer timer;
    timer.Start();
    lauch();
    timer.Stop();
    elapsed_ms += timer.ElapsedMillis();
    // Check answer
    int compare = CompareResults(reference_vector_y_out, vector_y_out,
                                 csr_matrix.num_rows, true);
    printf("\t%s\n", compare ? "FAIL" : "PASS");
    floag avg_ms = elapsed_ms / timing_iterations;

    DisplayPerf(setup_ms, avg_ms, csr_matrix);
    csr_matrix.clear();
}

/**
 * Main
 */
int main(int argc, char **argv) {
    CommandLineArgs args(argc, argv);
    std::string mtx_filename;
    args.GetCmdLineArgument("mtx", mtx_filename);
    RunTest<double, int>(mtx_filename, args);
    printf("\n");
    return 0;
}
