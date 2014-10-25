/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/

#include "utils.h"
#include <stdio.h>

__global__
void histo(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  extern __shared__ unsigned int s_histo[];
  int pos = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos >= numVals)
      return;

  // Zero all shared
  s_histo[threadIdx.x] = 0;
  __syncthreads();

  atomicAdd(&s_histo[vals[pos]], 1);

  __syncthreads();

  atomicAdd(&histo[threadIdx.x], s_histo[threadIdx.x]);
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  
  const int threadsPerBlock = 1024;
  int numBlocks = ceil(((float)numElems)/threadsPerBlock);
  histo<<<numBlocks, threadsPerBlock, threadsPerBlock*sizeof(unsigned int)>>>
      (d_vals, d_histo, numElems);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
