/*
  This is the baseline implementation of a 1D Stencil operation.

  Parameters:

  m0 > 0: dimension of the original input and output vector(array) size
  k0 > 0: dimesnion of the original weights vector(array)

  float* input_sequential: pointer to original input data
  float* input_distributed: pointer to the input data that you have distributed
  across the system

  float* output_sequential:  pointer to original output data
  float* output_distributed: pointer to the output data that you have
  distributed across the system

  float* weights_sequential:  pointer to original weights data
  float* weights_distributed: pointer to the weights data that you have
  distributed across the system

  Functions: Modify these however you please.

  DISTRIBUTED_ALLOCATE_NAME(...): Allocate the distributed buffers.
  DISTRIBUTE_DATA_NAME(...): takes the sequential data and distributes it across
  the system. COMPUTE_NAME(...): Performs the stencil computation.
  COLLECT_DATA_NAME(...): Collect the distributed output and combine it back to
  the sequential one for testing. DISTRIBUTED_FREE_NAME(...): Free the
  distributed buffers that were allocated


  - richard.m.veras@ou.edu

*/

#include <mpi.h>
#include <stdlib.h>

#include <immintrin.h>

#ifndef COMPUTE_NAME
#define COMPUTE_NAME baseline
#endif

#ifndef DISTRIBUTE_DATA_NAME
#define DISTRIBUTE_DATA_NAME baseline_distribute
#endif

#ifndef COLLECT_DATA_NAME
#define COLLECT_DATA_NAME baseline_collect
#endif

#ifndef DISTRIBUTED_ALLOCATE_NAME
#define DISTRIBUTED_ALLOCATE_NAME baseline_allocate
#endif

#ifndef DISTRIBUTED_FREE_NAME
#define DISTRIBUTED_FREE_NAME baseline_free
#endif

#define AVX2_FLOAT_N 8

void COMPUTE_NAME(int m0, int k0, float *input_distributed,
                  float *weights_distributed, float *output_distributed)

{
  /*
    STUDENT_TODO: Modify as you please.
  */
  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status status;
  int root_rid = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if (rid == root_rid) {
    /* This block will only run on the node that matches root_rid .*/

    __m256i rotate_indices = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
    int before_wrap = m0 - k0;

    for (int i0 = 0; i0 <= before_wrap; i0 += AVX2_FLOAT_N) {
      __m256 input_reg_1 = _mm256_loadu_ps(&input_distributed[i0]);
      __m256 input_reg_2 =
          _mm256_loadu_ps(&input_distributed[i0 + AVX2_FLOAT_N]);
      __m256 res_reg = _mm256_setzero_ps();
      for (int j = 0; j < k0; j++) {
        __m256 current_weight_reg =
            _mm256_broadcast_ss(&weights_distributed[j]);
        res_reg = _mm256_fmadd_ps(input_reg_1, current_weight_reg, res_reg);

        // Rotate input registers individually
        input_reg_1 = _mm256_permutevar8x32_ps(input_reg_1, rotate_indices);
        input_reg_2 = _mm256_permutevar8x32_ps(input_reg_2, rotate_indices);
        __m256 temp = input_reg_1;
        // Swap end pieces between registers to complete rotation
        input_reg_1 = _mm256_blend_ps(input_reg_1, input_reg_2, 0b10000000);
        input_reg_2 = _mm256_blend_ps(input_reg_2, temp, 0b10000000);
      }
      _mm256_storeu_ps(&output_distributed[i0], res_reg);
    }

    // do the part that wraps around
    for (int i0 = before_wrap; i0 < m0; i0++) {
      float res = 0.0f;
      int unwrapped_n = m0 - i0;
      int wrapped_n = k0 - unwrapped_n;
      for (int j = 0; j < unwrapped_n; j++) {
        res += input_distributed[j + i0] * weights_distributed[j];
      }
      for (int j = 0; j < wrapped_n; j++) {
        res += input_distributed[j] * weights_distributed[j + unwrapped_n];
      }
      output_distributed[i0] = res;
    }
  } else {
    /* This will run on all other nodes whose rid is not root_rid. */
  }
}

// Create the buffers on each node
void DISTRIBUTED_ALLOCATE_NAME(int m0, int k0, float **input_distributed,
                               float **weights_distributed,
                               float **output_distributed) {
  /*
    STUDENT_TODO: Modify as you please.
  */

  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status status;
  int root_rid = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if (rid == root_rid) {
    /* This block will only run on the node that matches root_rid .*/

    *input_distributed = (float *)malloc(sizeof(float) * m0);
    *output_distributed = (float *)malloc(sizeof(float) * m0);
    *weights_distributed = (float *)malloc(sizeof(float) * k0);
  } else {
    /* This will run on all other nodes whose rid is not root_rid. */
  }
}

void DISTRIBUTE_DATA_NAME(int m0, int k0, float *input_sequential,
                          float *weights_sequential, float *input_distributed,
                          float *weights_distributed) {
  /*
    STUDENT_TODO: Modify as you please.
  */

  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status status;
  int root_rid = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if (rid == root_rid) {
    /* This block will only run on the node that matches root_rid .*/

    // Distribute the inputs
    for (int i0 = 0; i0 < m0; ++i0)
      input_distributed[i0] = input_sequential[i0];

    // Distribute the weights
    for (int p0 = 0; p0 < k0; ++p0)
      weights_distributed[p0] = weights_sequential[p0];
  } else {
    /* This will run on all other nodes whose rid is not root_rid. */
  }
}

void COLLECT_DATA_NAME(int m0, int k0, float *output_distributed,
                       float *output_sequential) {
  /*
    STUDENT_TODO: Modify as you please.
  */

  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status status;
  int root_rid = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if (rid == root_rid) {
    /* This block will only run on the node that matches root_rid .*/

    // Collect the output
    for (int i0 = 0; i0 < m0; ++i0)
      output_sequential[i0] = output_distributed[i0];
  } else {
    /* This will run on all other nodes whose rid is not root_rid. */
  }
}

void DISTRIBUTED_FREE_NAME(int m0, int k0, float *input_distributed,
                           float *weights_distributed,
                           float *output_distributed) {
  /*
    STUDENT_TODO: Modify as you please.
  */

  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status status;
  int root_rid = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if (rid == root_rid) {
    /* This block will only run on the node that matches root_rid .*/

    free(input_distributed);
    free(weights_distributed);
    free(output_distributed);

  } else {
    /* This will run on all other nodes whose rid is not root_rid. */
  }
}
