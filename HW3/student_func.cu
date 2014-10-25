/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

__global__
void reduce_min(const float* d_in,
                float* d_out,
                const size_t numRows,
                const size_t numCols)
{
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int threadId = threadIdx.x;

  extern __shared__ float shdata_min[];

  if (myId >= numRows*numCols)
    shdata_min[threadId] = 0x7f800000;  // Infinity
  else
    shdata_min[threadId] = d_in[myId];

  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (threadId < s)
    {
      shdata_min[threadId] = min(shdata_min[threadId], shdata_min[threadId + s]);
    }
    __syncthreads();
  }

  if (threadId == 0)
  {
    d_out[blockIdx.x] = shdata_min[0];
  }
}

__global__
void reduce_max(const float* d_in,
                float* d_out,
                const size_t numRows,
                const size_t numCols)
{
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int threadId = threadIdx.x;

  extern __shared__ float shdata_max[];

  if (myId >= numRows*numCols)
    shdata_max[threadId] = 0xff800000; // Negative infinity
  else
    shdata_max[threadId] = d_in[myId];
    
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (threadId < s)
    {
      shdata_max[threadId] = max(shdata_max[threadId], shdata_max[threadId + s]);
    }
    __syncthreads();
  }

  if (threadId == 0)
  {
    d_out[blockIdx.x] = shdata_max[0];
  }
}

__global__
void histogramize(const float* const d_logLuminance,
                  unsigned int* d_histo,
                  const float logLumMin,
                  const float logLumRange,
                  const size_t numBins,
                  const size_t numRows,
                  const size_t numCols)
{
  int myId = threadIdx.x + blockDim.x * blockIdx.x;

  if (myId >= numRows*numCols)
    return;

  unsigned int bin = min(static_cast<unsigned int>(numBins - 1),
                         static_cast<unsigned int>((d_logLuminance[myId] - logLumMin) / logLumRange * numBins));

  atomicAdd(&(d_histo[bin]), 1);
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

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  // 1) find the minimum and maximum value in the input logLuminance channel
  //    store in min_logLum and max_logLum
  int threadsPerBlock = 1024;
  const dim3 blockSize(threadsPerBlock, 1, 1);
  int numBlocks = ceil(((float)(numRows*numCols)/threadsPerBlock));
  const dim3 gridSize(numBlocks, 1,1);

  float* d_min_intermediate;
  float* d_min_out;
  checkCudaErrors(cudaMalloc(&d_min_intermediate, threadsPerBlock*sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_min_out, threadsPerBlock*sizeof(float)));

  reduce_min<<<gridSize, blockSize, threadsPerBlock*sizeof(float)>>>(d_logLuminance,
    d_min_intermediate, numRows, numCols);

  int numBlocks2 = ceil(((float)numBlocks)/threadsPerBlock);
  assert(numBlocks2 == 1);
  const dim3 gridSize2(numBlocks2, 1, 1);

  reduce_min<<<gridSize2, blockSize, threadsPerBlock*sizeof(float)>>>(d_min_intermediate,
    d_min_out, numRows, numCols);
    
  checkCudaErrors(cudaMemcpy(&min_logLum, &(d_min_out[0]), sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_min_intermediate));
  checkCudaErrors(cudaFree(d_min_out));

  float* d_max_intermediate;
  float* d_max_out;
  checkCudaErrors(cudaMalloc(&d_max_intermediate, threadsPerBlock*sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_max_out, threadsPerBlock*sizeof(float)));
    
  reduce_max<<<gridSize, blockSize, threadsPerBlock*sizeof(float)>>>(d_logLuminance,
    d_max_intermediate, numRows, numCols);

  reduce_max<<<gridSize2, blockSize, threadsPerBlock*sizeof(float)>>>(d_max_intermediate,
    d_max_out, numRows, numCols);

  checkCudaErrors(cudaMemcpy(&max_logLum, &(d_max_out[0]), sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_max_intermediate));
  checkCudaErrors(cudaFree(d_max_out));

  // 2) subtract them to find the range
  float lumRange = max_logLum - min_logLum;

  // 3) generate a histogram of all the values in the logLuminance channel using
  //    the formula: bin = (lum[i] - lumMin) / lumRange * numBins
  unsigned int* d_histo;
  checkCudaErrors(cudaMalloc(&d_histo, numBins*sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(d_histo, 0, numBins*sizeof(unsigned int)));
  histogramize<<<gridSize, blockSize>>>(d_logLuminance, d_histo, min_logLum, lumRange, numBins,
      numRows, numCols);

  // 4) Perform an exclusive scan (prefix sum) on the histogram to get
  //    the cumulative distribution of luminance values (this should go in the
  //    incoming d_cdf pointer which already has been allocated for you)
  const dim3 blockSize3(numBins/2, 1, 1);
  assert(numBins <= 2048);
  const dim3 gridSize3(1, 1, 1);
  blelloch_scan_sum<<<gridSize3, blockSize3, numBins*sizeof(unsigned int)>>>(d_histo, d_cdf, numBins);

  checkCudaErrors(cudaFree(d_histo));
}
