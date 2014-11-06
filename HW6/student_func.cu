//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */

#include "utils.h"
#include <thrust/host_vector.h>
#include "reference_calc.cpp"

__global__
void computeMask(const uchar4* d_sourceImg,
                 unsigned char* d_mask,
                 const size_t numRowsSource,
                 const size_t numColsSource)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i > numRowsSource*numColsSource)
        return;
    
    d_mask[i] = (d_sourceImg[i].x + d_sourceImg[i].y + d_sourceImg[i].z < 3 * 255) ? 1 : 0;
}

__global__
void interiorBorder(unsigned char* borderPixels,
                    unsigned char* strictInteriorPixels,
                    unsigned char* mask,
                    const size_t numRowsSource,
                    const size_t numColsSource)
{
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;

    if ((r == 0) || (r >= numRowsSource-1) || (c == 0) || (c >= numColsSource-1))
        return;

    unsigned int offset = r * numColsSource + c;
    
    if (mask[offset]) {
      if (mask[offset - numColsSource] && mask[offset + numColsSource] &&
          mask[offset - 1] && mask[offset + 1]) {
        strictInteriorPixels[offset] = 1;
        borderPixels[offset] = 0;
      }
      else {
        strictInteriorPixels[offset] = 0;
        borderPixels[offset] = 1;
      }
    }
    else {
        strictInteriorPixels[offset] = 0;
        borderPixels[offset] = 0;
    }
}

__global__
void separateChannels(const uchar4* const d_sourceImg,
                            const uchar4* const d_destImg,
                            unsigned char* red_src,
                            unsigned char* blue_src,
                            unsigned char* green_src,
                            unsigned char* red_dst,
                            unsigned char* blue_dst,
                            unsigned char* green_dst,
                            const int srcSize)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (threadId >= srcSize)
        return;
    
    red_dst[threadId] = d_destImg[threadId].x;
    blue_dst[threadId] = d_destImg[threadId].y;
    green_dst[threadId] = d_destImg[threadId].z;

    red_src[threadId] = d_sourceImg[threadId].x;
    blue_src[threadId] = d_sourceImg[threadId].y;
    green_src[threadId] = d_sourceImg[threadId].z;
}

__global__
void initializeBlended(float* blendedValsRed_1,
                       float* blendedValsRed_2,
                       float* blendedValsBlue_1,
                       float* blendedValsBlue_2,
                       float* blendedValsGreen_1,
                       float* blendedValsGreen_2,
                       unsigned char* red_src,
                       unsigned char* blue_src,
                       unsigned char* green_src,
                       int srcSize)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i >= srcSize)
        return;

    blendedValsRed_1[i] = red_src[i];
    blendedValsRed_2[i] = red_src[i];
    blendedValsBlue_1[i] = blue_src[i];
    blendedValsBlue_2[i] = blue_src[i];
    blendedValsGreen_1[i] = green_src[i];
    blendedValsGreen_2[i] = green_src[i];
}

__global__
void computeG(const unsigned char* const channel,
              float* const g,
              const unsigned char* const strictInteriorPixels,
              const unsigned char* const borderPixels,
              const size_t numRowsSource,
              const size_t numColsSource)
{
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ((r == 0) || (r >= numRowsSource-1) || (c == 0) || (c >= numColsSource-1))
        return;
    
    unsigned int offset = r * numColsSource + c;
    
    if ((strictInteriorPixels[offset] == 1)&& (borderPixels[offset] == 0)) {
        
        float sum = 4.f * channel[offset];

        sum -= (float)channel[offset - 1] + (float)channel[offset + 1];
        sum -= (float)channel[offset + numColsSource] + (float)channel[offset - numColsSource];

        g[offset] = sum;
    }
}

__global__
void computeIteration(const unsigned char* const dstImg,
                      const unsigned char* const strictInteriorPixels,
                      const unsigned char* const borderPixels,
                      const size_t numRowsSource,
                      const size_t numColsSource,
                      const float* const f,
                      const float* const g,
                      float* const f_next)
{
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ((r == 0) || (r >= numRowsSource-1) || (c == 0) || (c >= numColsSource-1))
        return;
    
    unsigned int offset = r * numColsSource + c;
    
    if ((strictInteriorPixels[offset] == 1) && (borderPixels[offset] == 0)) {
        
        float blendedSum = 0.f;
        float borderSum  = 0.f;

        //process all 4 neighbor pixels
        //for each pixel if it is an interior pixel
        //then we add the previous f, otherwise if it is a
        //border pixel then we add the value of the destination
        //image at the border.  These border values are our boundary
        //conditions.
        if (strictInteriorPixels[offset - 1]) {
            blendedSum += f[offset - 1];
        }
        else {
            borderSum += dstImg[offset - 1];
        }

        if (strictInteriorPixels[offset + 1]) {
            blendedSum += f[offset + 1];
        }
        else {
            borderSum += dstImg[offset + 1];
        }

        if (strictInteriorPixels[offset - numColsSource]) {
            blendedSum += f[offset - numColsSource];
        }
        else {
            borderSum += dstImg[offset - numColsSource];
        }

        if (strictInteriorPixels[offset + numColsSource]) {
            blendedSum += f[offset + numColsSource];
        }
        else {
            borderSum += dstImg[offset + numColsSource];
        }

        float f_next_val = (blendedSum + borderSum + g[offset]) / 4.f;

        f_next[offset] = min(255.f, max(0.f, f_next_val)); //clip to [0, 255]
    }
}

__global__
void copyInterior(uchar4* d_blendedImg,
                  unsigned char* strictInteriorPixels,
                  unsigned char* borderPixels,
                  const size_t numRowsSource,
                  const size_t numColsSource,
                  float* blendedValsRed_2,
                  float* blendedValsBlue_2,
                  float* blendedValsGreen_2)
{
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;

    if ((r == 0) || (r >= numRowsSource-1) || (c == 0) || (c >= numColsSource-1))
        return;
    
    unsigned int offset = r * numColsSource + c;
    
    if ((strictInteriorPixels[offset] == 1) && (borderPixels[offset] == 0)) {
        d_blendedImg[offset].x = blendedValsRed_2[offset];
        d_blendedImg[offset].y = blendedValsBlue_2[offset];
        d_blendedImg[offset].z = blendedValsGreen_2[offset];
    }
}

__global__
void separateChannels(const uchar4* const d_sourceImg,
                            const uchar4* const d_destImg,
                            unsigned char* red_src,
                            unsigned char* blue_src,
                            unsigned char* green_src,
                            unsigned char* red_dst,
                            unsigned char* blue_dst,
                            unsigned char* green_dst,
                            const int srcSize)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadId >= srcSize)
        return;

    red_dst[threadId] = d_destImg[threadId].x;
    blue_dst[threadId] = d_destImg[threadId].y;
    green_dst[threadId] = d_destImg[threadId].z;

    red_src[threadId] = d_sourceImg[threadId].x;
    blue_src[threadId] = d_sourceImg[threadId].y;
    green_src[threadId] = d_sourceImg[threadId].z;
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
    cudaStream_t red_stream, blue_stream, green_stream;
    cudaStreamCreate(&red_stream);
    cudaStreamCreate(&blue_stream);
    cudaStreamCreate(&green_stream);
    
    uchar4* d_sourceImg;
    uchar4* d_destImg;
    uchar4* d_blendedImg;
    
    const int srcSize = numRowsSource*numColsSource;
    cudaMalloc(&d_sourceImg, srcSize*sizeof(uchar4));
    cudaMalloc(&d_destImg, srcSize*sizeof(uchar4));
    cudaMalloc(&d_blendedImg, srcSize*sizeof(uchar4));
    
    cudaMemcpy(d_sourceImg, h_sourceImg, srcSize*sizeof(uchar4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_destImg, h_destImg, srcSize*sizeof(uchar4), cudaMemcpyHostToDevice);
    
  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied. */
    unsigned char* d_mask;
    cudaMalloc(&d_mask, srcSize*sizeof(unsigned char));

    const int threadsPerBlock = 1024;
    int K = ceil(((float)srcSize)/threadsPerBlock);
    computeMask<<<K, threadsPerBlock>>>(d_sourceImg, d_mask, numRowsSource, numColsSource);

    /*
     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't. */
    unsigned char *d_borderPixels;
    unsigned char *d_strictInteriorPixels;
    cudaMalloc(&d_borderPixels, srcSize*sizeof(unsigned char));
    cudaMalloc(&d_strictInteriorPixels, srcSize*sizeof(unsigned char));
    
    const dim3 squareBlock(32, 32, 1);
    const dim3 squareGrid(ceil(((float)numColsSource)/32), ceil(((float)numRowsSource)/32), 1);
    interiorBorder<<<squareGrid, squareBlock>>>
        (d_borderPixels, d_strictInteriorPixels, d_mask, numRowsSource, numColsSource);
    
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    /*
     3) Separate out the incoming image into three separate channels */
    unsigned char* red_src;
    unsigned char* blue_src;
    unsigned char* green_src;
    unsigned char* red_dst;
    unsigned char* blue_dst;
    unsigned char* green_dst;
    cudaMalloc(&red_src, srcSize*sizeof(unsigned char));
    cudaMalloc(&blue_src, srcSize*sizeof(unsigned char));
    cudaMalloc(&green_src, srcSize*sizeof(unsigned char));
    cudaMalloc(&red_dst, srcSize*sizeof(unsigned char));
    cudaMalloc(&blue_dst, srcSize*sizeof(unsigned char));
    cudaMalloc(&green_dst, srcSize*sizeof(unsigned char));

    /*
     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess. */

      //for each color channel we'll need two buffers and we'll ping-pong between them
      float *blendedValsRed_1;
      float *blendedValsRed_2;
      float *blendedValsBlue_1;
      float *blendedValsBlue_2;
      float *blendedValsGreen_1;
      float *blendedValsGreen_2;
      checkCudaErrors(cudaMalloc(&blendedValsRed_1, srcSize*sizeof(float)));
      checkCudaErrors(cudaMalloc(&blendedValsRed_2, srcSize*sizeof(float)));
      checkCudaErrors(cudaMalloc(&blendedValsBlue_1, srcSize*sizeof(float)));
      checkCudaErrors(cudaMalloc(&blendedValsBlue_2, srcSize*sizeof(float)));
      checkCudaErrors(cudaMalloc(&blendedValsGreen_1, srcSize*sizeof(float)));
      checkCudaErrors(cudaMalloc(&blendedValsGreen_2, srcSize*sizeof(float)));

      initializeBlended<<<K, threadsPerBlock>>>(blendedValsRed_1, blendedValsRed_2, blendedValsBlue_1,
                        blendedValsBlue_2, blendedValsGreen_1, blendedValsGreen_2,
                        red_src, blue_src, green_src, srcSize);

    /*
     5) For each color channel perform the Jacobi iteration described 
        above 800 times. */
    float *g_red;
    float *g_blue;
    float *g_green;
    cudaMalloc(&g_red, srcSize * sizeof(float));
    cudaMalloc(&g_blue, srcSize * sizeof(float));
    cudaMalloc(&g_green, srcSize * sizeof(float));
    
    computeG<<<squareGrid, squareBlock, 0, red_stream>>>
        (red_src, g_red, d_strictInteriorPixels, d_borderPixels, numRowsSource, numColsSource);
    computeG<<<squareGrid, squareBlock, 0, blue_stream>>>
        (blue_src, g_blue, d_strictInteriorPixels, d_borderPixels, numRowsSource, numColsSource);
    computeG<<<squareGrid, squareBlock, 0, green_stream>>>
        (green_src, g_green, d_strictInteriorPixels, d_borderPixels, numRowsSource, numColsSource);
    
    const size_t numIterations = 800;

    for (size_t i = 0; i < numIterations; ++i) {
        computeIteration<<<squareGrid, squareBlock, 0, red_stream>>>
            (red_dst, d_strictInteriorPixels, d_borderPixels, numRowsSource, numColsSource,
             blendedValsRed_1, g_red, blendedValsRed_2);
        
        computeIteration<<<squareGrid, squareBlock, 0, blue_stream>>>
            (blue_dst, d_strictInteriorPixels, d_borderPixels, numRowsSource, numColsSource,
             blendedValsBlue_1, g_blue, blendedValsBlue_2);
        
        computeIteration<<<squareGrid, squareBlock, 0, green_stream>>>
            (green_dst, d_strictInteriorPixels, d_borderPixels, numRowsSource, numColsSource,
             blendedValsGreen_1, g_green, blendedValsGreen_2);
 
        std::swap(blendedValsRed_1, blendedValsRed_2);
        std::swap(blendedValsBlue_1, blendedValsBlue_2);
        std::swap(blendedValsGreen_1, blendedValsGreen_2);
    }
    
    std::swap(blendedValsRed_1, blendedValsRed_2);
    std::swap(blendedValsBlue_1, blendedValsBlue_2);
    std::swap(blendedValsGreen_1, blendedValsGreen_2);
    
    /*
     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range. */
    cudaMemcpy(d_blendedImg, d_destImg, sizeof(uchar4) * srcSize, cudaMemcpyDeviceToDevice);
    copyInterior<<<squareGrid, squareBlock>>>(d_blendedImg, d_strictInteriorPixels,
        d_borderPixels, numRowsSource, numColsSource, blendedValsRed_2, blendedValsBlue_2,
        blendedValsGreen_2);
        
    cudaMemcpy(h_blendedImg, d_blendedImg, sizeof(uchar4) * srcSize, cudaMemcpyDeviceToHost);
    /*
      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */

  checkCudaErrors(cudaFree(d_sourceImg));
  checkCudaErrors(cudaFree(d_destImg));
  checkCudaErrors(cudaFree(d_blendedImg));
  checkCudaErrors(cudaFree(d_borderPixels));
  checkCudaErrors(cudaFree(d_strictInteriorPixels));
  checkCudaErrors(cudaFree(red_src));
  checkCudaErrors(cudaFree(blue_src));
  checkCudaErrors(cudaFree(green_src));
  checkCudaErrors(cudaFree(red_dst));
  checkCudaErrors(cudaFree(blue_dst));
  checkCudaErrors(cudaFree(green_dst));
  checkCudaErrors(cudaFree(blendedValsRed_1));
  checkCudaErrors(cudaFree(blendedValsRed_2));
  checkCudaErrors(cudaFree(blendedValsBlue_1));
  checkCudaErrors(cudaFree(blendedValsBlue_2));
  checkCudaErrors(cudaFree(blendedValsGreen_1));
  checkCudaErrors(cudaFree(blendedValsGreen_2));
  checkCudaErrors(cudaFree(g_red));
  checkCudaErrors(cudaFree(g_blue));
  checkCudaErrors(cudaFree(g_green));

  /* The reference calculation is provided below, feel free to use it
     for debugging purposes. 
   */

  /*
    uchar4* h_reference = new uchar4[srcSize];
    reference_calc(h_sourceImg, numRowsSource, numColsSource,
                   h_destImg, h_reference);

    checkResultsEps((unsigned char *)h_reference, (unsigned char *)h_blendedImg, 4 * srcSize, 2, .01);
    delete[] h_reference; */
}

