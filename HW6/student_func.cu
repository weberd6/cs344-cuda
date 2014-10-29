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

    if ((r == 0) || (r >= numRowsSource-1) || (c == 0) || (c >= numRowsSource-1))
        return;

    if (mask[r * numColsSource + c]) {
      if (mask[(r -1) * numColsSource + c] && mask[(r + 1) * numColsSource + c] &&
          mask[r * numColsSource + c - 1] && mask[r * numColsSource + c + 1]) {
        strictInteriorPixels[r * numColsSource + c] = 1;
        borderPixels[r * numColsSource + c] = 0;
        //interiorPixelList.push_back(make_uint2(r, c));
      }
      else {
        strictInteriorPixels[r * numColsSource + c] = 0;
        borderPixels[r * numColsSource + c] = 1;
      }
    }
    else {
        strictInteriorPixels[r * numColsSource + c] = 0;
        borderPixels[r * numColsSource + c] = 0;
    }
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
    const uchar4* d_sourceImg;
    const uchar4* d_destImg;
    const uchar4* d_blendedImg;

    const int srcSize = numRowsSource*numColsSource;
    cudaMalloc(&d_sourceImg, srcSize*sizeof(unsigned char));
    cudaMalloc(&d_destImg, srcSize*sizeof(unsigned char));
    cudaMalloc(&d_blendedImg, srcSize*sizeof(unsigned char));

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

    /*
     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */



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

