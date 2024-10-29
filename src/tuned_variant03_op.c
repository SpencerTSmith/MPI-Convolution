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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

#define TILE_SIZE_DISTRIBUTED 4

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

  //Number of elements that will be computed for each process
  int elems_per_proc = m0/num_ranks;
  int total_elems_needed = elems_per_proc + (k0-1);



  if (rid == root_rid) {
    /* This block will only run on the node that matches root_rid .*/
    for(int i1 = 0; i1 < elems_per_proc; i1++){
      float res = 0.0;
      for(int j1 = 0; j1 < k0; j1++){
        res += input_distributed[j1+i1]*weights_distributed[j1];
      }
      output_distributed[i1] = res;
    }
    // printf("Rank %d given: [",rid);
    // for(int i = 0; i < total_elems_needed; i++){
    //   printf("%.2f, ",input_distributed[i]);
    // }
    // printf("]\n");
    // printf("Rank %d created: [",rid);
    // for(int i = 0; i < elems_per_proc; i++){
    //   printf("%.2f, ",output_distributed[i]);
    // }
    // printf("]\n");
    // padding to register size
    // float padded_weights[8] = {0};
    // memcpy(padded_weights, weights_distributed, sizeof(float) * k0);
    // __m256 weights = _mm256_loadu_ps(padded_weights);

    // int before_wrap = m0 - k0;
    // for (int i0 = 0; i0 <= before_wrap; ++i0) {
    //   __m256 input = _mm256_loadu_ps(&input_distributed[i0]);
    //   __m256 mults = _mm256_mul_ps(weights, input);
    //   float to_sum[8] = {0};
    //   _mm256_storeu_ps(to_sum, mults);
    //   float sum = 0.0f;
    //   for (int j = 0; j < 8; j++) {
    //     sum += to_sum[j];
    //   }
    //   output_distributed[i0] = sum;
    // }
    // // do the part that wraps around
    // for (int i0 = before_wrap + 1; i0 < m0; i0++) {
    //   float res = 0.0f;
    //   int unwrapped_n = m0 - i0;
    //   int wrapped_n = k0 - unwrapped_n;
    //   for (int j = 0; j < unwrapped_n; j++) {
    //     res += input_distributed[j + i0] * weights_distributed[j];
    //   }
    //   for (int j = 0; j < wrapped_n; j++) {
    //     res += input_distributed[j] * weights_distributed[j + unwrapped_n];
    //   }
    //   output_distributed[i0] = res;
  } 
  else {
    /* This will run on all other nodes whose rid is not root_rid. */
    for(int i1 = 0; i1 < elems_per_proc; i1++){
      float res = 0.0;
      for(int j1 = 0; j1 < k0; j1++){
        res += input_distributed[j1+i1]*weights_distributed[j1];
      }
      output_distributed[i1] = res;
    }
    // printf("Rank %d given: [",rid);
    // for(int i = 0; i < total_elems_needed; i++){
    //   printf("%.2f, ",input_distributed[i]);
    // }
    // printf("]\n");
    // printf("Rank %d created: [",rid);
    // for(int i = 0; i < elems_per_proc; i++){
    //   printf("%.2f, ",output_distributed[i]);
    // }
    // printf("]\n");
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

  int elems_per_proc = m0/num_ranks;
  //Find total elements needed for each nodes computations
  int total_elems_needed = elems_per_proc + (k0-1);


  if (rid == root_rid) {
    /* This block will only run on the node that matches root_rid .*/

    *input_distributed = (float *)malloc(sizeof(float) * total_elems_needed);
    *output_distributed = (float *)malloc(sizeof(float) * elems_per_proc);
    *weights_distributed = (float *)malloc(sizeof(float) * k0);
  } else {
    *input_distributed = (float *)malloc(sizeof(float) * total_elems_needed);
    *output_distributed = (float *)malloc(sizeof(float) * elems_per_proc);
    *weights_distributed = (float *)malloc(sizeof(float) * k0);
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

  int elems_per_proc = m0/num_ranks;
  //Find total elements needed for each nodes computations
  int total_elems_needed = elems_per_proc + (k0-1);



  if (rid == root_rid) {
    /* This block will only run on the node that matches root_rid .*/
    // Distribute the weights
    for (int p0 = 0; p0 < k0; ++p0)
      weights_distributed[p0] = weights_sequential[p0];

    for(int dst_rank = 0; dst_rank < num_ranks; dst_rank++){
      if(dst_rank != root_rid){
    //Find where that node should start
    int start = elems_per_proc*dst_rank;

    if(start + total_elems_needed > m0){
      //Before wrap
      int after_wrap = (start+total_elems_needed) - m0;
      int before_wrap = total_elems_needed-after_wrap;
      for(int i0 = 0; i0 < before_wrap; i0++){
        input_distributed[i0] = input_sequential[start+i0];
      }
      //After wrap
      for(int i0 = 0; i0 < after_wrap; i0++){
        input_distributed[before_wrap+i0] = input_sequential[i0];
      }
    }
    else{
      //If it does not need to wrap around any elements
    for (int i0 = 0; i0 < total_elems_needed; i0++)
      input_distributed[i0] = input_sequential[start+i0];
    }
    MPI_Send(input_distributed,
    total_elems_needed,
    MPI_FLOAT,
    dst_rank,
    tag,
    MPI_COMM_WORLD
    );
    MPI_Send(weights_distributed,
    k0,
    MPI_FLOAT,
    dst_rank,
    tag,
    MPI_COMM_WORLD
    );
    }
  }
  //Send the data to the root nodes' distributed buffer
  int start = 0;
    if(start + total_elems_needed > m0){
      //Before wrap
      int before_wrap = (start+total_elems_needed) - m0;
      for(int i0 = 0; i0 < before_wrap; i0++){
        input_distributed[i0] = input_sequential[start+i0];
      }
      //After wrap
      for(int i0 = 0; i0 < (total_elems_needed - before_wrap); i0++){
        input_distributed[before_wrap+i0] = input_sequential[i0];
      }
    }
    else{
      //If it does not need to wrap around any elements
    for (int i0 = 0; i0 < total_elems_needed; i0++)
      input_distributed[i0] = input_sequential[start+i0];
    }
  }
  else{
    /* This will run on all other nodes whose rid is not root_rid. */
    MPI_Recv(input_distributed,
    total_elems_needed,
    MPI_FLOAT,
    root_rid,
    tag,
    MPI_COMM_WORLD,
    &status);

    MPI_Recv(weights_distributed,
    k0,
    MPI_FLOAT,
    root_rid,
    tag,
    MPI_COMM_WORLD,
    &status);
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

  int elems_per_proc = m0/num_ranks;


  if (rid == root_rid) {
    /* This block will only run on the node that matches root_rid .*/
  MPI_Gather(
      output_distributed,
      elems_per_proc,
      MPI_FLOAT,
      output_sequential,
      elems_per_proc,
      MPI_FLOAT,
      root_rid,
      MPI_COMM_WORLD);
  } else {
    /* This will run on all other nodes whose rid is not root_rid. */
  MPI_Gather(
      output_distributed,
      elems_per_proc,
      MPI_FLOAT,
      NULL,
      elems_per_proc,
      MPI_FLOAT,
      root_rid,
      MPI_COMM_WORLD);
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
    free(input_distributed);
    free(weights_distributed);
    free(output_distributed);
  }
}
