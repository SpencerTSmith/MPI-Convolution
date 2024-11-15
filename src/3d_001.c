/*
  This is the 3D tiling implementation of a 1D Stencil operation.

  Parameters:

  m0 > 0: dimension of the original input and output vector(array) size
  k0 > 0: dimesnion of the original weights vector(array)

  float* input_sequential: pointer to original input data
  float* input_distributed: pointer to the input data that you have distributed across
  the system

  float* output_sequential:  pointer to original output data
  float* output_distributed: pointer to the output data that you have distributed across
  the system

  float* weights_sequential:  pointer to original weights data
  float* weights_distributed: pointer to the weights data that you have distributed across
  the system

  Functions: Modify these however you please.

  DISTRIBUTED_ALLOCATE_NAME(...): Allocate the distributed buffers.
  DISTRIBUTE_DATA_NAME(...): takes the sequential data and distributes it across the system.
  COMPUTE_NAME(...): Performs the stencil computation.
  COLLECT_DATA_NAME(...): Collect the distributed output and combine it back to the sequential
  one for testing.
  DISTRIBUTED_FREE_NAME(...): Free the distributed buffers that were allocated


  - richard.m.veras@ou.edu

*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef COMPUTE_NAME
#define COMPUTE_NAME baseline
#endif

#ifndef DISTRIBUTE_DATA_NAME
#define DISTRIBUTE_DATA_NAME baseline_distribute
#endif

#ifndef COLLECT_DATA_NAME
#define  COLLECT_DATA_NAME baseline_collect
#endif  


#ifndef DISTRIBUTED_ALLOCATE_NAME
#define DISTRIBUTED_ALLOCATE_NAME baseline_allocate
#endif


#ifndef DISTRIBUTED_FREE_NAME
#define DISTRIBUTED_FREE_NAME baseline_free
#endif


#define TILE_SIZE_WEIGHTS 8 //CHANGEME -- This is the tile size for weights.
#define TILE_SIZE_INNER 8 //CHANGEME -- This is the tile size for Inner Tiles for the dataset
#define TILE_SIZE_OUTER  //CHANGEME -- This is the tile size for the FIRST Loop


void COMPUTE_NAME( int m0, int k0,
		   float *input_distributed,
		   float *weights_distributed,
		   float *output_distributed )

{
  /*
    This version is for 3D Tiling
  */
  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status  status;
  int root_rid = 0;
  int d0 = m0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if (rid == root_rid)
    {
        // Outer loop for dataset tiles (loop 1)
        for (int outer_tile = 0; outer_tile < m0; outer_tile += TILE_SIZE_OUTER)
        {
            int outer_tile_end = outer_tile + TILE_SIZE_OUTER;
            if (outer_tile_end > m0)
            {
               outer_tile_end = m0;
            }

            // Middle loop for weights tiles (loop 2)
            for (int p_tile = 0; p_tile < k0; p_tile += TILE_SIZE_WEIGHTS)
            {
                int p_tile_end = p_tile + TILE_SIZE_WEIGHTS;
                if (p_tile_end > k0)
                {
                  p_tile_end = k0;
                }

                // Inner loop for subtiles of dataset. This is a tile inside a tile.  (loop 3)
                for (int inner_tile = 0; inner_tile < outer_tile_end; inner_tile += TILE_SIZE_INNER)
                {
                    int inner_tile_end = inner_tile + TILE_SIZE_INNER;
                    if (inner_tile_end > outer_tile_end)
                    {
                        inner_tile_end = outer_tile_end;
                    }

                    //go throguh tile for dataset traversal
                    for (int i0 = inner_tile; i0 < inner_tile_end; ++i0)
                    {
                        float res = 0.0f;

                        //go through each tile for weights traversal
                        for (int p0 = p_tile; p0 < p_tile_end; ++p0)
                        {
                          res += input_distributed[(p0 + i0) % m0] * weights_distributed[p0];

                        }

                        output_distributed[i0] = res;
                    }
                }
            }
        }
    }
  else
    {
      /* This will run on all other nodes whose rid is not root_rid. */
    }

}


// Create the buffers on each node
void DISTRIBUTED_ALLOCATE_NAME( int m0, int k0,
				float **input_distributed,
				float **weights_distributed,
				float **output_distributed )
{
  /*
    STUDENT_TODO: Modify as you please.
  */

  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status  status;
  int root_rid = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if(rid == root_rid )
    {
      /* This block will only run on the node that matches root_rid .*/

      *input_distributed=(float *)malloc(sizeof(float)*m0);
      *output_distributed=(float *)malloc(sizeof(float)*m0);
      *weights_distributed=(float *)malloc(sizeof(float)*k0);
    }
  else
    {
      /* This will run on all other nodes whose rid is not root_rid. */
    }
}

void DISTRIBUTE_DATA_NAME( int m0, int k0,
			   float *input_sequential,
			   float *weights_sequential,
			   float *input_distributed,
			   float *weights_distributed )
{
  /*
    STUDENT_TODO: Modify as you please.
  */

  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status  status;
  int root_rid = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if(rid == root_rid )
    {
      /* This block will only run on the node that matches root_rid .*/

      // Distribute the inputs
      for( int i0 = 0; i0 < m0; ++i0 )
	input_distributed[i0] = input_sequential[i0];
  
      // Distribute the weights
      for( int p0 = 0; p0 < k0; ++p0 )
	weights_distributed[p0] = weights_sequential[p0];
    }
  else
    {
      /* This will run on all other nodes whose rid is not root_rid. */      
    }

}



void COLLECT_DATA_NAME( int m0, int k0,
			float *output_distributed,
			float *output_sequential )
{
    /*
      STUDENT_TODO: Modify as you please.
    */

    int rid;
    int num_ranks;
    int tag = 0;
    MPI_Status  status;
    int root_rid = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if(rid == root_rid )
      {
	/* This block will only run on the node that matches root_rid .*/
  
	// Collect the output
	for( int i0 = 0; i0 < m0; ++i0 )
	  output_sequential[i0] = output_distributed[i0];
      }
    else
      {
	/* This will run on all other nodes whose rid is not root_rid. */      
      } 
}




void DISTRIBUTED_FREE_NAME( int m0, int k0,
			    float *input_distributed,
			    float *weights_distributed,
			    float *output_distributed )
{
  /*
    STUDENT_TODO: Modify as you please.
  */
    
  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status  status;
  int root_rid = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if(rid == root_rid )
    {
      /* This block will only run on the node that matches root_rid .*/

      free(input_distributed);
      free(weights_distributed);
      free(output_distributed);

    }
  else
    {
      /* This will run on all other nodes whose rid is not root_rid. */  
    }
}


