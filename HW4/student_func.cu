
//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <cstdio>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__
void histogramize(unsigned int* d_inputVals,
                  unsigned int* d_binHistogram,
                  const size_t numElems,
                  unsigned int mask, 
                  unsigned int i,
                  unsigned int* d_binByPos)
{
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId >= numElems)
    return;

  unsigned int bin = (d_inputVals[threadId] & mask) >> i;

  d_binByPos[threadId] = bin;
  atomicAdd(&(d_binHistogram[bin]), 1);
}

__global__
void blelloch_scan_sum(unsigned int* const d_histo,
                       unsigned int* d_cdf,
                       const size_t numBins)
{
  extern __shared__ unsigned int shdata[];
  int threadId = threadIdx.x;
  int offset = 1;

  shdata[2*threadId] = d_histo[2*threadId];
  shdata[2*threadId+1] = d_histo[2*threadId+1];

  //Reduce
  for (int d = numBins>>1; d > 0; d >>= 1)
  {
    __syncthreads();
    if (threadId < d)
    {
      int ai = offset*(2*threadId+1)-1;
      int bi = offset*(2*threadId+2)-1;

      shdata[bi] += shdata[ai];
    }
    offset *= 2;
  }

  if (threadId == 0)
    shdata[numBins-1] = 0; // Set last element to identity element

  //Downsweep
  for (int d = 1; d < numBins; d *= 2)
  {
    offset >>= 1;
    __syncthreads();
    if (threadId < d)
    {
      int ai = offset*(2*threadId+1)-1;
      int bi = offset*(2*threadId+2)-1;

      float t = shdata[ai];
      shdata[ai] = shdata[bi];
      shdata[bi] += t;
    }
  }

  __syncthreads();

  // write results to device memory
  d_cdf[2*threadId] = shdata[2*threadId];
  d_cdf[2*threadId+1] = shdata[2*threadId+1];
}

__global__
void blelloch_scan_sum_large(unsigned int* const d_histo,
                       unsigned int* d_cdf,
                       const size_t numBins,
                       unsigned int* sums)
{
  int threadId = threadIdx.x;
  int offset = 1;

  unsigned int start_index = blockIdx.x*blockDim.x*2;

  d_cdf[start_index + 2*threadId] = d_histo[start_index + 2*threadId];
  d_cdf[start_index + 2*threadId+1] = d_histo[start_index + 2*threadId+1];

  //Reduce
  for (int d = numBins>>1; d > 0; d >>= 1)
  {
    __syncthreads();
    if (threadId < d)
    {
      int ai = offset*(2*threadId+1)-1;
      int bi = offset*(2*threadId+2)-1;

      d_cdf[start_index + bi] += d_cdf[start_index + ai];
    }
    offset *= 2;
  }

  if (threadId == 0) {
    sums[blockIdx.x] = d_cdf[start_index + numBins-1];
    d_cdf[start_index + numBins-1] = 0; // Set last element to identity element
  }

  //Downsweep
  for (int d = 1; d < numBins; d *= 2)
  {
    offset >>= 1;
    __syncthreads();
    if (threadId < d)
    {
      int ai = offset*(2*threadId+1)-1;
      int bi = offset*(2*threadId+2)-1;

      float t = d_cdf[start_index + ai];
      d_cdf[start_index + ai] = d_cdf[start_index + bi];
      d_cdf[start_index + bi] += t;
    }
  }

  __syncthreads();
}

__global__
void get_bin_mask(unsigned int* d_binByPos,
                  unsigned int targetBin,
                  unsigned int* d_binMask,
                  const size_t numElems)
{
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    
  if (threadId >= numElems)
    return;
    
  if (d_binByPos[threadId] == targetBin)
    d_binMask[threadId] = 1;
  else
    d_binMask[threadId] = 0;
}

__global__
void add_constant(unsigned int* d_cdf,
                  unsigned int* incr,
                  const size_t numElems)
{
  int pos = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos >= numElems)
    return;

  d_cdf[pos] = d_cdf[pos] + incr[blockIdx.x];
}

unsigned long upper_power_of_two(unsigned long v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

void large_scan_sum(unsigned int* const d_histo,
                    unsigned int* d_cdf,
                    const size_t numElems)
{
  const int B = 256;
  int numBlocks = ceil(((float)numElems)/B);
  int numBlocks2 = upper_power_of_two(numBlocks);

  unsigned int* sums;
  checkCudaErrors(cudaMalloc(&sums, numBlocks*sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(sums, 0, numBlocks*sizeof(unsigned int)));
  blelloch_scan_sum_large<<<numBlocks, B/2>>>(d_histo, d_cdf, B, sums);

  // Pad array by creating new array, zeroing all elements, and copying values
  unsigned int* padded_sums;
  checkCudaErrors(cudaMalloc(&padded_sums, numBlocks2*sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(padded_sums, 0, numBlocks2*sizeof(unsigned int)));
  checkCudaErrors(cudaMemcpy(padded_sums, sums, numBlocks*sizeof(unsigned int), cudaMemcpyDeviceToDevice));

  unsigned int* incr;
  checkCudaErrors(cudaMalloc(&incr, numBlocks2*sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(incr, 0, numBlocks2*sizeof(unsigned int)));
  blelloch_scan_sum<<<1, numBlocks2/2, numBlocks2*sizeof(unsigned int)>>>(padded_sums, incr, numBlocks2);

  add_constant<<<numBlocks, B>>>(d_cdf, incr, numElems);

  checkCudaErrors(cudaFree(sums));
  checkCudaErrors(cudaFree(padded_sums));
  checkCudaErrors(cudaFree(incr));
}

__global__
void set_relative_offsets(unsigned int* d_binMask,
                          unsigned int* d_binRelOffsets,
                          unsigned int* d_relativeOffsets,
                          const size_t numElems)
{
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId >= numElems)
    return;

  if (d_binMask[threadId] == 1) {
    d_relativeOffsets[threadId] = d_binRelOffsets[threadId];
  }
}

__global__
void set_positions_and_values(unsigned int* d_binByPos,
                              unsigned int* d_relativeOffsets,
                              unsigned int* d_binScan,
                              unsigned int* d_inputVals,
                              unsigned int* d_inputPos,
                              unsigned int* d_outputVals,
                              unsigned int* d_outputPos,
                              const size_t numElems)
{
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId >= numElems)
    return;

  unsigned int pos = d_relativeOffsets[threadId] + d_binScan[d_binByPos[threadId]];

  d_outputPos[pos] = d_inputPos[threadId];
  d_outputVals[pos] = d_inputVals[threadId];
}

__global__
void swap(unsigned int* first,
          unsigned int* second,
          unsigned int* temp,
          const size_t numElems)
{
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId >= numElems)
      return;

  temp[threadId] = first[threadId];
  first[threadId] = second[threadId];
  second[threadId] = temp[threadId];
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  int threadsPerBlock = 1024;
  int numBlocks = ceil(((float)numElems)/threadsPerBlock);

  const int numBits = 4;
  const int numBins = 1 << numBits;

  unsigned int* d_binHistogram;
  unsigned int* d_binScan;
  unsigned int* d_binByPos;
  unsigned int* d_binMask;
  unsigned int* d_binRelOffsets;
  unsigned int* d_relativeOffsets;
  unsigned int* d_temp;

  checkCudaErrors(cudaMalloc(&d_binHistogram, numBins*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_binScan, numBins*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_binByPos, numElems*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_binMask, numElems*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_binRelOffsets, numElems*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_relativeOffsets, numElems*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_temp, numElems*sizeof(unsigned int)));

  for (unsigned int i = 0; i < 8*sizeof(unsigned int); i += numBits) {
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    unsigned int mask = (numBins - 1) << i;

    checkCudaErrors(cudaMemset(d_binHistogram, 0, numBins*sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_binScan, 0, numBins*sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_binByPos, 0, numElems*sizeof(unsigned int)));

    // 1) Histogram of the number of occurrences of each digits
    histogramize<<<numBlocks, threadsPerBlock>>>(d_inputVals, d_binHistogram, numElems, mask, i, d_binByPos);

    // 2) Exclusive Prefix Sum of Histogram
    blelloch_scan_sum<<<1, numBins/2, numBins*sizeof(unsigned int)>>>
      (d_binHistogram, d_binScan, numBins);

    // 3) Determine relative offset of each digit
    //      For example [0 0 1 1 0 0 1]
    //              ->  [0 1 0 1 2 3 2]
    
    checkCudaErrors(cudaMemset(d_relativeOffsets, 0, numElems*sizeof(unsigned int)));

    for (unsigned int j = 0; j < numBins; j++) {
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        
      checkCudaErrors(cudaMemset(d_binMask, 0, numElems*sizeof(unsigned int)));
      checkCudaErrors(cudaMemset(d_binRelOffsets, 0, numElems*sizeof(unsigned int)));

      get_bin_mask<<<numBlocks, threadsPerBlock>>>(d_binByPos, j, d_binMask, numElems);

      large_scan_sum(d_binMask, d_binRelOffsets, numElems);

      set_relative_offsets<<<numBlocks, threadsPerBlock>>>
          (d_binMask, d_binRelOffsets, d_relativeOffsets, numElems);
    }

    // 4) Combine the results of steps 2 & 3 to determine the final
    //    output location for each element and move it there
    set_positions_and_values<<<numBlocks, threadsPerBlock>>>
        (d_binByPos, d_relativeOffsets, d_binScan, d_inputVals, d_inputPos, d_outputVals,
         d_outputPos, numElems);

    //swap the buffers
    swap<<<numBlocks, threadsPerBlock>>>(d_outputPos, d_inputPos, d_temp, numElems);
    swap<<<numBlocks, threadsPerBlock>>>(d_outputVals, d_inputVals, d_temp, numElems);
  }

  checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, numElems*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems*sizeof(unsigned int), cudaMemcpyDeviceToDevice));

  checkCudaErrors(cudaFree(d_binHistogram));
  checkCudaErrors(cudaFree(d_binScan));
  checkCudaErrors(cudaFree(d_binByPos));
  checkCudaErrors(cudaFree(d_binMask));
  checkCudaErrors(cudaFree(d_binRelOffsets));
  checkCudaErrors(cudaFree(d_relativeOffsets));
  checkCudaErrors(cudaFree(d_temp));
}

