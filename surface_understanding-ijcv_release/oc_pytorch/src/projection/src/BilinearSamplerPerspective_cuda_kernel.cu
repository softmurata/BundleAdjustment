#include <stdbool.h>
#include <stdio.h>
#include "BilinearSamplerPerspective_cuda_kernel.h"

#define real float

// code adapted from github repo
// implemented by Yijie Guo (guoyijie@umich.edu) and Xinchen Yan (skywalkeryxc@gmail.com)

// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids

__device__ void getTopLeftFront(float x, int width, int& point, float& weight)
{
   /* for interpolation :
      stores in point and weight :
      - the x-coordinate of the pixel on the left (or y-coordinate of the upper pixel)
      - the weight for interpolating
   */

   float xcoord = (x + 1) * (width - 1) / 2;
   point = floor(xcoord);
   weight = 1 - (xcoord - point);
}

__device__ bool between(int value, int lowerBound, int upperBound)
{
   return (value >= lowerBound && value <= upperBound);
}

__device__ void sumReduceShMem(volatile float s[])
{
   /* obviously only works for 32 elements */
   /* sums up a shared memory array of 32 elements, stores it in s[0] */
   /* whole warp can then read first element (broadcasting) */
   //if(threadIdx.x<32) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+32]; }
   if(threadIdx.x<16) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+16] + s[threadIdx.x+32]; }
   if(threadIdx.x<8) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+8]; }
   if(threadIdx.x<4) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+4]; }
   if(threadIdx.x<2) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+2]; }
   if(threadIdx.x<1) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+1]; }
}

__device__ void sumReduceShMemPerspective(volatile float s[])
{
   /* obviously only works for 32 elements */
   /* sums up a shared memory array of 32 elements, stores it in s[0] */
   /* whole warp can then read first element (broadcasting) */
   if(threadIdx.x<16) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+16]; }
   if(threadIdx.x<8) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+8]; }
   if(threadIdx.x<4) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+4]; }
   if(threadIdx.x<2) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+2]; }
   if(threadIdx.x<1) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+1]; }
}


// Affine Transformation

__global__ void bilinearSamplingFromGrid(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, int inputImages_strideDepth, int inputImages_strideHeight, int inputImages_strideWidth,
                                         float* grids_data, int grids_strideBatch, int grids_strideYX, int grids_strideDepth, int grids_strideHeight, int grids_strideWidth,
                                         float* output_data, int output_strideBatch, int output_strideChannels, int output_strideDepth, int output_strideHeight, int output_strideWidth,
                                         int inputImages_channels, int inputImages_depth, int inputImages_height, int inputImages_width, int output_depth, int output_width)
{
   // each (32,16) block 16 output pixels (for coalescing the grid read)
   // x,y = coordinates (xOut = blockIdx.x*16+blockDim.y+threadIdx.y)
   // z = batch index
   // threadIdx.x : used for features (coalescing is trivial)
      
   const int xOut = blockIdx.x*blockDim.y+threadIdx.y;
   const bool withinImageBounds = xOut < output_width;
   const bool withinGridBounds = blockIdx.x*blockDim.y + threadIdx.x / 3 < output_width;
   const int yOut = blockIdx.y;
   const int width = inputImages_width;
   const int height = inputImages_height;
   const int depth = output_depth;

   const int b = blockIdx.z/depth;
   const int zOut = blockIdx.z%depth;

   float zf, yf,xf;

   
   __shared__ float gridData[32];
   if (threadIdx.y==0 && withinGridBounds)
   {
      gridData[threadIdx.x] = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + zOut*grids_strideDepth + threadIdx.x];
   }
   __syncthreads();
   if(!withinImageBounds) return;
   zf = gridData[threadIdx.y*3];
   yf = gridData[threadIdx.y*3+1];
   xf = gridData[threadIdx.y*3+2];
   // printf("%.3f %.3f %.3f\n", blockIdx.x, blockIdx.y, blockIdx.z);  
   //printf("%.3f %.3f %.3f\n", zf, yf, xf);
  
   int yInTopLeftFront, xInTopLeftFront, zInTopLeftFront;
   float yWeightTopLeftFront, xWeightTopLeftFront, zWeightTopLeftFront;
   getTopLeftFront(zf, inputImages_depth, zInTopLeftFront, zWeightTopLeftFront);
   getTopLeftFront(yf, inputImages_height, yInTopLeftFront, yWeightTopLeftFront);
   getTopLeftFront(xf, inputImages_width, xInTopLeftFront, xWeightTopLeftFront);
  
   //printf("GPU y[%.3f] x[%.3f] z[%.3f] WeightTopLeftFront\n",yWeightTopLeftFront, xWeightTopLeftFront, zWeightTopLeftFront);
   // printf("GPU y[%d] x[%d] z[%d] InTopLeftFront\n",yInTopLeftFront, xInTopLeftFront, zInTopLeftFront);
   
   const int outAddress = output_strideBatch * b + output_strideHeight * yOut + output_strideWidth * xOut + output_strideDepth * zOut;
   
   const int inTopLeftFrontAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeftFront
  + inputImages_strideWidth * xInTopLeftFront + inputImages_strideDepth * zInTopLeftFront;

   const int inTopLeftBackAddress = inTopLeftFrontAddress + inputImages_strideDepth;

   const int inTopRightFrontAddress = inTopLeftFrontAddress + inputImages_strideWidth;
   const int inTopRightBackAddress = inTopRightFrontAddress + inputImages_strideDepth;

   const int inBottomLeftFrontAddress = inTopLeftFrontAddress + inputImages_strideHeight;
   const int inBottomLeftBackAddress = inBottomLeftFrontAddress + inputImages_strideDepth;

   const int inBottomRightFrontAddress = inBottomLeftFrontAddress + inputImages_strideWidth;
   const int inBottomRightBackAddress = inBottomRightFrontAddress + inputImages_strideDepth;

   float v=0;
   float inTopLeftFront=0;
   float inTopLeftBack=0;
   float inTopRightFront=0;
   float inTopRightBack=0;
   float inBottomLeftFront=0;
   float inBottomLeftBack=0;
   float inBottomRightFront=0;
   float inBottomRightBack=0;

   bool topLeftFrontIsIn = between(xInTopLeftFront, 0, inputImages_width-1) 
      && between(yInTopLeftFront, 0, inputImages_height-1) && between(zInTopLeftFront, 0, inputImages_depth-1);
      
   bool topLeftBackIsIn = between(xInTopLeftFront, 0, inputImages_width-1) 
      && between(yInTopLeftFront, 0, inputImages_height-1) && between(zInTopLeftFront+1, 0, inputImages_depth-1);
      
   bool topRightFrontIsIn = between(xInTopLeftFront+1, 0, inputImages_width-1) 
      && between(yInTopLeftFront, 0, inputImages_height-1) && between(zInTopLeftFront, 0, inputImages_depth-1);
      
   bool topRightBackIsIn = between(xInTopLeftFront+1, 0, inputImages_width-1) 
      && between(yInTopLeftFront, 0, inputImages_height-1) && between(zInTopLeftFront+1, 0, inputImages_depth-1);
      
   bool bottomLeftFrontIsIn = between(xInTopLeftFront, 0, inputImages_width-1) 
      && between(yInTopLeftFront+1, 0, inputImages_height-1) && between(zInTopLeftFront, 0, inputImages_depth-1);
      
   bool bottomLeftBackIsIn = between(xInTopLeftFront, 0, inputImages_width-1) 
      && between(yInTopLeftFront+1, 0, inputImages_height-1) && between(zInTopLeftFront+1, 0, inputImages_depth-1);
      
   bool bottomRightFrontIsIn = between(xInTopLeftFront+1, 0, inputImages_width-1) 
      && between(yInTopLeftFront+1, 0, inputImages_height-1) && between(zInTopLeftFront, 0, inputImages_depth-1);
      
   bool bottomRightBackIsIn = between(xInTopLeftFront+1, 0, inputImages_width-1) 
      && between(yInTopLeftFront+1, 0, inputImages_height-1) && between(zInTopLeftFront+1, 0, inputImages_depth-1);


// interpolation happens here
   for(int t=threadIdx.x; t<inputImages_channels; t+= blockDim.x)
   {
      if(topLeftFrontIsIn) inTopLeftFront = inputImages_data[inTopLeftFrontAddress + t];
      if(topLeftBackIsIn) inTopLeftBack = inputImages_data[inTopLeftBackAddress + t];

      if(topRightFrontIsIn) inTopRightFront = inputImages_data[inTopRightFrontAddress + t];
      if(topRightBackIsIn) inTopRightBack = inputImages_data[inTopRightBackAddress + t];

      if(bottomLeftFrontIsIn) inBottomLeftFront = inputImages_data[inBottomLeftFrontAddress + t];
      if(bottomLeftBackIsIn) inBottomLeftBack = inputImages_data[inBottomLeftBackAddress + t];

      if(bottomRightFrontIsIn) inBottomRightFront = inputImages_data[inBottomRightFrontAddress + t];
      if(bottomRightBackIsIn) inBottomRightBack = inputImages_data[inBottomRightBackAddress + t];


      v = xWeightTopLeftFront * yWeightTopLeftFront * zWeightTopLeftFront * inTopLeftFront
        + xWeightTopLeftFront * yWeightTopLeftFront * (1-zWeightTopLeftFront) * inTopLeftBack
        + (1 - xWeightTopLeftFront) * yWeightTopLeftFront * zWeightTopLeftFront * inTopRightFront
        + (1 - xWeightTopLeftFront) * yWeightTopLeftFront * (1-zWeightTopLeftFront) * inTopRightBack
        + xWeightTopLeftFront * (1 - yWeightTopLeftFront) * zWeightTopLeftFront * inBottomLeftFront
        + xWeightTopLeftFront * (1 - yWeightTopLeftFront) * (1-zWeightTopLeftFront) * inBottomLeftBack
        + (1 - xWeightTopLeftFront) * (1 - yWeightTopLeftFront) * zWeightTopLeftFront * inBottomRightFront
        + (1 - xWeightTopLeftFront) * (1 - yWeightTopLeftFront) * (1-zWeightTopLeftFront) * inBottomRightBack;

      output_data[outAddress + t] = v;
   }
}



template<bool onlyGrid> __global__ void backwardBilinearSampling(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, int inputImages_strideDepth, int inputImages_strideHeight, int inputImages_strideWidth,
                                         float* gradInputImages_data, int gradInputImages_strideBatch, int gradInputImages_strideChannels, int gradInputImages_strideDepth, int gradInputImages_strideHeight, int gradInputImages_strideWidth,
                                         float* grids_data, int grids_strideBatch, int grids_strideYX, int grids_strideDepth, int grids_strideHeight, int grids_strideWidth,
                                         float* gradGrids_data, int gradGrids_strideBatch, int gradGrids_strideYX, int gradGrids_strideDepth, int gradGrids_strideHeight, int gradGrids_strideWidth,
                                         float* gradOutput_data, int gradOutput_strideBatch, int gradOutput_strideChannels, int gradOutput_strideDepth, int gradOutput_strideHeight, int gradOutput_strideWidth,
                                         int inputImages_channels, int inputImages_depth, int inputImages_height, int inputImages_width, int gradOutput_depth, int gradOutput_width)
{
   // each (32,16) block 16 output pixels (for coalescing the grid read)
   // x,y = coordinates
   // z = batch index
   // threads : used for features
      
   const int xOut = blockIdx.x*blockDim.y+threadIdx.y;
   const bool withinImageBounds = xOut < gradOutput_width;
   const bool withinGridBounds = blockIdx.x*blockDim.y + threadIdx.x / 3 < gradOutput_width;

   const int yOut = blockIdx.y;
   const int width = inputImages_width;
   const int height = inputImages_height;

   const int depth = gradOutput_depth;

   const int b = blockIdx.z/depth;
   const int zOut = blockIdx.z%depth;
   
   float yf,xf, zf;

   __shared__ float gridData[48];
   if (threadIdx.y==0 && withinGridBounds)
   {
      gridData[threadIdx.x] = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + zOut*grids_strideDepth + threadIdx.x];
   }
   __syncthreads();

   if(withinImageBounds)
   {
      zf = gridData[threadIdx.y*3];
      yf = gridData[threadIdx.y*3+1];
      xf = gridData[threadIdx.y*3+2];

      int yInTopLeftFront, xInTopLeftFront, zInTopLeftFront;
      float yWeightTopLeftFront, xWeightTopLeftFront, zWeightTopLeftFront;
      getTopLeftFront(zf, inputImages_depth, zInTopLeftFront, zWeightTopLeftFront);
      getTopLeftFront(yf, inputImages_height, yInTopLeftFront, yWeightTopLeftFront);
      getTopLeftFront(xf, inputImages_width, xInTopLeftFront, xWeightTopLeftFront);

      const int inTopLeftFrontAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeftFront
  + inputImages_strideWidth * xInTopLeftFront + inputImages_strideDepth * zInTopLeftFront;

      const int inTopLeftBackAddress = inTopLeftFrontAddress + inputImages_strideDepth;

      const int inTopRightFrontAddress = inTopLeftFrontAddress + inputImages_strideWidth;
      const int inTopRightBackAddress = inTopRightFrontAddress + inputImages_strideDepth;

      const int inBottomLeftFrontAddress = inTopLeftFrontAddress + inputImages_strideHeight;
      const int inBottomLeftBackAddress = inBottomLeftFrontAddress + inputImages_strideDepth;

      const int inBottomRightFrontAddress = inBottomLeftFrontAddress + inputImages_strideWidth;
      const int inBottomRightBackAddress = inBottomRightFrontAddress + inputImages_strideDepth;

      const int gradInputImagesTopLeftFrontAddress = gradInputImages_strideBatch * b + gradInputImages_strideHeight * yInTopLeftFront + gradInputImages_strideWidth * xInTopLeftFront + gradInputImages_strideDepth * zInTopLeftFront;
      const int gradInputImagesTopLeftBackAddress = gradInputImagesTopLeftFrontAddress + gradInputImages_strideDepth;
      const int gradInputImagesTopRightFrontAddress = gradInputImagesTopLeftFrontAddress + gradInputImages_strideWidth;
      const int gradInputImagesTopRightBackAddress = gradInputImagesTopRightFrontAddress + gradInputImages_strideDepth;

      const int gradInputImagesBottomLeftFrontAddress = gradInputImagesTopLeftFrontAddress + gradInputImages_strideHeight;
      const int gradInputImagesBottomLeftBackAddress = gradInputImagesBottomLeftFrontAddress + gradInputImages_strideDepth;
      const int gradInputImagesBottomRightFrontAddress = gradInputImagesBottomLeftFrontAddress + gradInputImages_strideWidth;
      const int gradInputImagesBottomRightBackAddress = gradInputImagesBottomRightFrontAddress + gradInputImages_strideDepth;


      const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideHeight * yOut + gradOutput_strideWidth * xOut + gradOutput_strideDepth * zOut;

      float topLeftFrontDotProduct = 0;
      float topLeftBackDotProduct = 0;
      float topRightFrontDotProduct = 0;
      float topRightBackDotProduct = 0;

      float bottomLeftFrontDotProduct = 0;
      float bottomLeftBackDotProduct = 0;
      float bottomRightFrontDotProduct = 0;
      float bottomRightBackDotProduct = 0;

      bool topLeftFrontIsIn = between(xInTopLeftFront, 0, inputImages_width-1) 
         && between(yInTopLeftFront, 0, inputImages_height-1) && between(zInTopLeftFront, 0, inputImages_depth-1);
      bool topLeftBackIsIn = between(xInTopLeftFront, 0, inputImages_width-1) 
         && between(yInTopLeftFront, 0, inputImages_height-1) && between(zInTopLeftFront+1, 0, inputImages_depth-1);

      bool topRightFrontIsIn = between(xInTopLeftFront+1, 0, inputImages_width-1) 
         && between(yInTopLeftFront, 0, inputImages_height-1) && between(zInTopLeftFront, 0, inputImages_depth-1);
      bool topRightBackIsIn = between(xInTopLeftFront+1, 0, inputImages_width-1) 
         && between(yInTopLeftFront, 0, inputImages_height-1) && between(zInTopLeftFront+1, 0, inputImages_depth-1);

      bool bottomLeftFrontIsIn = between(xInTopLeftFront, 0, inputImages_width-1) 
         && between(yInTopLeftFront+1, 0, inputImages_height-1) && between(zInTopLeftFront, 0, inputImages_depth-1);
      bool bottomLeftBackIsIn = between(xInTopLeftFront, 0, inputImages_width-1) 
         && between(yInTopLeftFront+1, 0, inputImages_height-1) && between(zInTopLeftFront+1, 0, inputImages_depth-1);

      bool bottomRightFrontIsIn = between(xInTopLeftFront+1, 0, inputImages_width-1) 
         && between(yInTopLeftFront+1, 0, inputImages_height-1) && between(zInTopLeftFront, 0, inputImages_depth-1);
      bool bottomRightBackIsIn = between(xInTopLeftFront+1, 0, inputImages_width-1) 
         && between(yInTopLeftFront+1, 0, inputImages_height-1) && between(zInTopLeftFront+1, 0, inputImages_depth-1);

      /*
         In that loop we accumulate
         - gradients into the gradInputImages array with atomic adds
         - we compute the dot product that we need for the grid gradient
      */

      for(int t=threadIdx.x; t<inputImages_channels; t+= blockDim.x)
      {
         float gradOutValue = gradOutput_data[gradOutputAddress + t];
         // bool between(int value, int lowerBound, int upperBound)
         if(topLeftFrontIsIn)
         {
            float inTopLeftFront = inputImages_data[inTopLeftFrontAddress + t];
            topLeftFrontDotProduct += inTopLeftFront * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesTopLeftFrontAddress + t], xWeightTopLeftFront * yWeightTopLeftFront * zWeightTopLeftFront * gradOutValue);
         }

        if(topLeftBackIsIn)
         {
            float inTopLeftBack = inputImages_data[inTopLeftBackAddress + t];
            topLeftBackDotProduct += inTopLeftBack * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesTopLeftBackAddress + t], xWeightTopLeftFront * yWeightTopLeftFront * (1-zWeightTopLeftFront) * gradOutValue);
         }

         if(topRightFrontIsIn)
         {
            float inTopRightFront = inputImages_data[inTopRightFrontAddress + t];
            topRightFrontDotProduct += inTopRightFront * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesTopRightFrontAddress + t], (1 - xWeightTopLeftFront) * yWeightTopLeftFront * zWeightTopLeftFront * gradOutValue);
         }

         if(topRightBackIsIn)
         {
            float inTopRightBack = inputImages_data[inTopRightBackAddress + t];
            topRightBackDotProduct += inTopRightBack * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesTopRightBackAddress + t], (1 - xWeightTopLeftFront) * yWeightTopLeftFront * (1-zWeightTopLeftFront) * gradOutValue);
          }

         if(bottomLeftFrontIsIn)
         {
            float inBottomLeftFront = inputImages_data[inBottomLeftFrontAddress + t];
            bottomLeftFrontDotProduct += inBottomLeftFront * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesBottomLeftFrontAddress + t], xWeightTopLeftFront * (1 - yWeightTopLeftFront) * zWeightTopLeftFront * gradOutValue);
         }

         if(bottomLeftBackIsIn)
         {
            float inBottomLeftBack = inputImages_data[inBottomLeftBackAddress + t];
            bottomLeftBackDotProduct += inBottomLeftBack * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesBottomLeftBackAddress + t], xWeightTopLeftFront * (1 - yWeightTopLeftFront) * (1-zWeightTopLeftFront) * gradOutValue);
          }

         if(bottomRightFrontIsIn)
         {
            float inBottomRightFront = inputImages_data[inBottomRightFrontAddress + t];
            bottomRightFrontDotProduct += inBottomRightFront * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesBottomRightFrontAddress + t], (1 - xWeightTopLeftFront) * (1 - yWeightTopLeftFront) * zWeightTopLeftFront * gradOutValue);
         }

         if(bottomRightBackIsIn)
         {
            float inBottomRightBack = inputImages_data[inBottomRightBackAddress + t];
            bottomRightBackDotProduct += inBottomRightBack * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesBottomRightBackAddress + t], (1 - xWeightTopLeftFront) * (1 - yWeightTopLeftFront) * (1-zWeightTopLeftFront) * gradOutValue);
          }


      }

      /*
         Here we reduce the dot product and compute the grid gradient before writing it.
      */

      /* could do shuffles and use no shmem at all but cuda arch is 2.0 */
/*
      __shared__ volatile float __shmem[16][32];
      __shmem[threadIdx.y][threadIdx.x] = topLeftDotProduct;
      sumReduceShMem(__shmem[threadIdx.y]);
      topLeftDotProduct = __shmem[threadIdx.y][0];
      __shmem[threadIdx.y][threadIdx.x] = topRightDotProduct;
      sumReduceShMem(__shmem[threadIdx.y]);
      topRightDotProduct = __shmem[threadIdx.y][0];
      __shmem[threadIdx.y][threadIdx.x] = bottomLeftDotProduct;
      sumReduceShMem(__shmem[threadIdx.y]);
      bottomLeftDotProduct = __shmem[threadIdx.y][0];
      __shmem[threadIdx.y][threadIdx.x] = bottomRightDotProduct;
      sumReduceShMem(__shmem[threadIdx.y]);
      bottomRightDotProduct = __shmem[threadIdx.y][0];
*/

__shared__ volatile float __shmem[16][48];
__shmem[threadIdx.y][threadIdx.x] = topLeftFrontDotProduct;
sumReduceShMem(__shmem[threadIdx.y]);
topLeftFrontDotProduct = __shmem[threadIdx.y][0];
//__syncthreads();

__shmem[threadIdx.y][threadIdx.x] = topLeftBackDotProduct;
sumReduceShMem(__shmem[threadIdx.y]);
topLeftBackDotProduct = __shmem[threadIdx.y][0];
//__syncthreads();


__shmem[threadIdx.y][threadIdx.x] = topRightFrontDotProduct;
sumReduceShMem(__shmem[threadIdx.y]);
topRightFrontDotProduct = __shmem[threadIdx.y][0];
//__syncthreads();

__shmem[threadIdx.y][threadIdx.x] = topRightBackDotProduct;
sumReduceShMem(__shmem[threadIdx.y]);
topRightBackDotProduct = __shmem[threadIdx.y][0];
//__syncthreads();

__shmem[threadIdx.y][threadIdx.x] = bottomLeftFrontDotProduct;
sumReduceShMem(__shmem[threadIdx.y]);
bottomLeftFrontDotProduct = __shmem[threadIdx.y][0];
//__syncthreads();

__shmem[threadIdx.y][threadIdx.x] = bottomLeftBackDotProduct;
sumReduceShMem(__shmem[threadIdx.y]);
bottomLeftBackDotProduct = __shmem[threadIdx.y][0];
//__syncthreads();

__shmem[threadIdx.y][threadIdx.x] = bottomRightFrontDotProduct;
sumReduceShMem(__shmem[threadIdx.y]);
bottomRightFrontDotProduct = __shmem[threadIdx.y][0];
//__syncthreads();

__shmem[threadIdx.y][threadIdx.x] = bottomRightBackDotProduct;
sumReduceShMem(__shmem[threadIdx.y]);
bottomRightBackDotProduct = __shmem[threadIdx.y][0];
//__syncthreads();


      yf = topLeftFrontDotProduct * xWeightTopLeftFront * zWeightTopLeftFront * (-1)
         + topLeftBackDotProduct * xWeightTopLeftFront * (1-zWeightTopLeftFront) * (-1)
         + topRightFrontDotProduct * (1-xWeightTopLeftFront) * zWeightTopLeftFront * (-1)
         + topRightBackDotProduct * (1-xWeightTopLeftFront) * (1-zWeightTopLeftFront) *(-1)
         + bottomLeftFrontDotProduct * xWeightTopLeftFront * zWeightTopLeftFront * (1)
         + bottomLeftBackDotProduct * xWeightTopLeftFront * (1-zWeightTopLeftFront) * (1)
         + bottomRightFrontDotProduct * (1-xWeightTopLeftFront) * zWeightTopLeftFront * (1)
         + bottomRightBackDotProduct * (1-xWeightTopLeftFront) * (1-zWeightTopLeftFront) *(1);

      xf = topLeftFrontDotProduct * yWeightTopLeftFront * zWeightTopLeftFront *(-1)
         + topLeftBackDotProduct * yWeightTopLeftFront * (1-zWeightTopLeftFront) *(-1)
         + topRightFrontDotProduct * yWeightTopLeftFront * zWeightTopLeftFront * 1
         + topRightBackDotProduct * yWeightTopLeftFront * (1-zWeightTopLeftFront) * 1
         + bottomLeftFrontDotProduct * (1-yWeightTopLeftFront) * zWeightTopLeftFront * (-1)
         + bottomLeftBackDotProduct * (1-yWeightTopLeftFront) * (1-zWeightTopLeftFront) * (-1)
         + bottomRightFrontDotProduct * (1-yWeightTopLeftFront) * zWeightTopLeftFront * (1)
         + bottomRightBackDotProduct * (1-yWeightTopLeftFront) *(1-zWeightTopLeftFront) * (1);

      zf = topLeftFrontDotProduct * yWeightTopLeftFront * xWeightTopLeftFront * (-1)
         + topLeftBackDotProduct * yWeightTopLeftFront * xWeightTopLeftFront *(1)
         + topRightFrontDotProduct * yWeightTopLeftFront * (1-xWeightTopLeftFront) *(-1)
         + topRightBackDotProduct * yWeightTopLeftFront * (1-xWeightTopLeftFront) *(1)
         + bottomLeftFrontDotProduct * (1-yWeightTopLeftFront) * xWeightTopLeftFront * (-1)
         + bottomLeftBackDotProduct * (1-yWeightTopLeftFront) * xWeightTopLeftFront * (1)
         + bottomRightFrontDotProduct * (1-yWeightTopLeftFront) * (1-xWeightTopLeftFront) *(-1)
         + bottomRightBackDotProduct * (1-yWeightTopLeftFront) * (1-xWeightTopLeftFront) * 1;



      if(threadIdx.x==0)
      {
         gridData[threadIdx.y*3] = zf * (inputImages_depth-1) / 2;
         gridData[threadIdx.y*3+1] = yf * (inputImages_height-1) / 2;
         gridData[threadIdx.y*3+2] = xf * (inputImages_width-1) / 2;
      }
   }// must put a big if condition in order not to hang at __syncthreads()...
   __syncthreads();

   if(threadIdx.y==0 && withinGridBounds)      
       gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + zOut*gradGrids_strideDepth + threadIdx.x] = gridData[threadIdx.x];
}


#ifdef __cplusplus
extern "C" {
#endif

int BilinearSamplerBZHWD_updateOutput_cuda_kernel(/*output->size[3]*/int sz1,
                                                 /*output->size[2]*/int sz2,
                                                 /*output->size[0]*/int sz3,
                                                 /*output->size[1]*/int sz4,
                                                 /*THCudaTensor_size(state, inputImages, 4)*/int ic,
                                                 /*THCudaTensor_size(state, inputImages, 1)*/int id,
                                                 /*THCudaTensor_size(state, inputImages, 2)*/int ih,
                                                 /*THCudaTensor_size(state, inputImages, 3)*/int iw,
                                                 /*THCudaTensor_size(state, output, 1)*/int od,
                                                 /*THCudaTensor_size(state, output, 3)*/int ow,
                                                 /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int isd, int ish, int isw,
                                                 /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsd, int gsh, int gsw,
                                                 /*THCudaTensor *output*/float *output, int osb, int osc, int osd, int osh, int osw,
                                                 /*THCState_getCurrentStream(state)*/cudaStream_t stream)
{
   // printf("%d", sz3*sz4);
   dim3 blocks((sz1+15)/16, sz2, sz3*sz4);
   dim3 threads(32,16);
/* assume BHWD */
   bilinearSamplingFromGrid <<< blocks, threads, 0, /*THCState_getCurrentStream(state)*/stream >>> (/*THCudaTensor_data(state, inputImages)*/inputImages,
                                                      /*THCudaTensor_stride(state, inputImages, 0)*/isb,
                                                      /*THCudaTensor_stride(state, inputImages, 4)*/isc,
                                                      /*THCudaTensor_stride(state, inputImages, 1)*/isd,
                                                      /*THCudaTensor_stride(state, inputImages, 2)*/ish,
                                                      /*THCudaTensor_stride(state, inputImages, 3)*/isw,

                                                      /*THCudaTensor_data(state, grids)*/grids,
                                                      /*THCudaTensor_stride(state, grids, 0)*/gsb,
                                                      /*THCudaTensor_stride(state, grids, 4)*/gsc,
                                                      /*THCudaTensor_stride(state, grids, 1)*/gsd,
                                                      /*THCudaTensor_stride(state, grids, 2)*/gsh,
                                                      /*THCudaTensor_stride(state, grids, 3)*/gsw,

                                                      /*THCudaTensor_data(state, output)*/output,
                                                      /*THCudaTensor_stride(state, output, 0)*/osb,
                                                      /*THCudaTensor_stride(state, output, 4)*/osc,
                                                      /*THCudaTensor_stride(state, output, 1)*/osd,
                                                      /*THCudaTensor_stride(state, output, 2)*/osh,
                                                      /*THCudaTensor_stride(state, output, 3)*/osw,

                                                      /*THCudaTensor_size(state, inputImages, 4)*/ic,
                                                      /*THCudaTensor_size(state, inputImages, 1)*/id,
                                                      /*THCudaTensor_size(state, inputImages, 2)*/ih,
                                                      /*THCudaTensor_size(state, inputImages, 3)*/iw,
                                                      /*THCudaTensor_size(state, output, 1)*/od,
                                                      /*THCudaTensor_size(state, output, 3)*/ow);


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return 0;
  }
  //printf("GPU forward end!\n");
  return 1;
}

int BilinearSamplerBZHWD_updateGradInput_cuda_kernel(/*output->size[3]*/int sz1,
                                                 /*output->size[2]*/int sz2,
                                                 /*output->size[0]*/int sz3,
                                                 /*output->size[1]*/int sz4,
                                                 /*THCudaTensor_size(state, inputImages, 4)*/int ic,
                                                 /*THCudaTensor_size(state, inputImages, 1)*/int id,
                                                 /*THCudaTensor_size(state, inputImages, 2)*/int ih,
                                                 /*THCudaTensor_size(state, inputImages, 3)*/int iw,
                                                 /*THCudaTensor_size(state, gradOutput, 1)*/int god,
                                                 /*THCudaTensor_size(state, gradOutput, 3)*/int gow,
                                                 /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int isd, int ish, int isw, 
                                                 /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsd, int gsh, int gsw, 
                                                /*THCudaTensor *gradInputImages*/float *gradInputImages, int gisb, int gisc, int gisd, int gish, int gisw, 
						/*THCudaTensor *gradGrids*/float *gradGrids, int ggsb, int ggsc, int ggsd, int ggsh, int ggsw,
                                                    /*THCudaTensor *gradOutput*/float *gradOutput, int gosb, int gosc, int gosd, int gosh, int gosw,
                                                 /*THCState_getCurrentStream(state)*/cudaStream_t stream)
{
   dim3 blocks((sz1+15)/16, sz2, sz3*sz4);
   dim3 threads(48,16);

   backwardBilinearSampling <false> <<< blocks, threads, 0, /*THCState_getCurrentStream(state)*/stream >>> (
                                                      /*THCudaTensor_data(state, inputImages)*/inputImages, 
                                                      /*THCudaTensor_stride(state, inputImages, 0)*/isb,
                                                      /*THCudaTensor_stride(state, inputImages, 4)*/isc,
                                                      /*THCudaTensor_stride(state, inputImages, 1)*/isd,
                                                      /*THCudaTensor_stride(state, inputImages, 2)*/ish,
                                                      /*THCudaTensor_stride(state, inputImages, 3)*/isw,

                                                      /*THCudaTensor_data(state, gradInputImages)*/gradInputImages,
                                                      /*THCudaTensor_stride(state, gradInputImages, 0)*/gisb,
                                                      /*THCudaTensor_stride(state, gradInputImages, 4)*/gisc,
                                                      /*THCudaTensor_stride(state, gradInputImages, 1)*/gisd,
                                                      /*THCudaTensor_stride(state, gradInputImages, 2)*/gish,
                                                      /*THCudaTensor_stride(state, gradInputImages, 3)*/gisw,

                                                      /*THCudaTensor_data(state, grids)*/grids, 
                                                      /*THCudaTensor_stride(state, grids, 0)*/gsb,
                                                      /*THCudaTensor_stride(state, grids, 4)*/gsc,
                                                      /*THCudaTensor_stride(state, grids, 1)*/gsd,
                                                      /*THCudaTensor_stride(state, grids, 2)*/gsh,
                                                      /*THCudaTensor_stride(state, grids, 3)*/gsw,

                                                      /*THCudaTensor_data(state, gradGrids)*/gradGrids, 
                                                      /*THCudaTensor_stride(state, gradGrids, 0)*/ggsb,
                                                      /*THCudaTensor_stride(state, gradGrids, 4)*/ggsc,
                                                      /*THCudaTensor_stride(state, gradGrids, 1)*/ggsd,
                                                      /*THCudaTensor_stride(state, gradGrids, 2)*/ggsh,
                                                      /*THCudaTensor_stride(state, gradGrids, 3)*/ggsw,

                                                      /*THCudaTensor_data(state, gradOutput)*/gradOutput,
                                                      /*THCudaTensor_stride(state, gradOutput, 0)*/gosb,
                                                      /*THCudaTensor_stride(state, gradOutput, 4)*/gosc,
                                                      /*THCudaTensor_stride(state, gradOutput, 1)*/gosd,
                                                      /*THCudaTensor_stride(state, gradOutput, 2)*/gosh,
                                                      /*THCudaTensor_stride(state, gradOutput, 3)*/gosw,

                                                      /*THCudaTensor_size(state, inputImages, 4)*/ic,
                                                      /*THCudaTensor_size(state, inputImages, 1)*/id, 
                                                      /*THCudaTensor_size(state, inputImages, 2)*/ih,
                                                      /*THCudaTensor_size(state, inputImages, 3)*/iw,
                                                      /*THCudaTensor_size(state, gradOutput, 1)*/god,
                                                      /*THCudaTensor_size(state, gradOutput, 3))*/gow);



  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
  printf("error in BilinearSampler.updateGradInput: %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return 0;
  }
  return 1;
}


// Perspective Transformation

__global__ void bilinearSamplingFromGridPerspective(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, int inputImages_strideDepth, int inputImages_strideHeight, int inputImages_strideWidth,
float* grids_data, int grids_strideBatch, int grids_strideYX, int grids_strideDepth, int grids_strideHeight, int grids_strideWidth,
float* output_data, int output_strideBatch, int output_strideChannels, int output_strideDist, int output_strideHeight, int output_strideWidth,
int inputImages_channels, int inputImages_depth, int inputImages_height, int inputImages_width, int output_dist, int output_width, float focal_length)
{
   // each (32,16) block 16 output pixels (for coalescing the grid read)
   // x,y = coordinates (xOut = blockIdx.x*16+blockDim.y+threadIdx.y)
   // z = batch index
   // threadIdx.x : used for features (coalescing is trivial)
      
   const int xOut = blockIdx.x*blockDim.y+threadIdx.y;
   const bool withinImageBounds = xOut < output_width;
   const bool withinGridBounds = blockIdx.x*blockDim.y + threadIdx.x / 4 < output_width;
   const int yOut = blockIdx.y;
   const int width = inputImages_width;
   const int height = inputImages_height;
   const int dist = output_dist;

   const int b = blockIdx.z/dist;
   const int disOut = blockIdx.z%dist;
   
   float zf,yf,xf, disf;

   __shared__ float gridData[32];
   if (threadIdx.y==0 && withinGridBounds)
   {
      gridData[threadIdx.x] = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + disOut*grids_strideDepth + threadIdx.x];
   }
   __syncthreads();
   if(!withinImageBounds) return;
   zf = gridData[threadIdx.y*4];
   yf = gridData[threadIdx.y*4+1];
   xf = gridData[threadIdx.y*4+2];
   disf = gridData[threadIdx.y*4+3];

   //yf = yf / disf;
   //xf = xf / disf;
   //zf = zf / disf - (focal_length + 0.5);

   const int outAddress = output_strideBatch * b + output_strideDist * disOut + output_strideHeight * yOut + output_strideWidth * xOut;

   int zInFrontTopLeft, yInFrontTopLeft, xInFrontTopLeft;
   float yWeightFrontTopLeft, xWeightFrontTopLeft, zWeightFrontTopLeft;
   getTopLeftFront(zf, inputImages_depth, zInFrontTopLeft, zWeightFrontTopLeft);
   getTopLeftFront(yf, inputImages_height, yInFrontTopLeft, yWeightFrontTopLeft);
   getTopLeftFront(xf, inputImages_width, xInFrontTopLeft, xWeightFrontTopLeft);

   const int inFrontTopLeftAddress = inputImages_strideBatch * b + inputImages_strideDepth * zInFrontTopLeft + inputImages_strideHeight * yInFrontTopLeft + inputImages_strideWidth * xInFrontTopLeft;
   const int inFrontTopRightAddress = inFrontTopLeftAddress + inputImages_strideWidth;
   const int inFrontBottomLeftAddress = inFrontTopLeftAddress + inputImages_strideHeight;
   const int inFrontBottomRightAddress = inFrontBottomLeftAddress + inputImages_strideWidth;

   const int inBackTopLeftAddress = inFrontTopLeftAddress + inputImages_strideDepth;
   const int inBackTopRightAddress = inBackTopLeftAddress + inputImages_strideWidth;
   const int inBackBottomLeftAddress = inBackTopLeftAddress + inputImages_strideHeight;
   const int inBackBottomRightAddress = inBackBottomLeftAddress + inputImages_strideWidth;

   float v=0;
   float inFrontTopLeft=0;
   float inFrontTopRight=0;
   float inFrontBottomLeft=0;
   float inFrontBottomRight=0;
   float inBackTopLeft=0;
   float inBackTopRight=0;
   float inBackBottomLeft=0;
   float inBackBottomRight=0;

   bool frontTopLeftIsIn = between(xInFrontTopLeft, 0, inputImages_width-1)
    && between(yInFrontTopLeft, 0, inputImages_height-1) && between(zInFrontTopLeft, 0, inputImages_depth-1);

   bool backTopLeftIsIn = between(xInFrontTopLeft, 0, inputImages_width-1)
    && between(yInFrontTopLeft, 0, inputImages_height-1) && between(zInFrontTopLeft+1, 0, inputImages_depth-1);

   bool frontTopRightIsIn = between(xInFrontTopLeft+1, 0, inputImages_width-1)
    && between(yInFrontTopLeft, 0, inputImages_height-1) && between(zInFrontTopLeft, 0, inputImages_depth-1);

   bool backTopRightIsIn = between(xInFrontTopLeft+1, 0, inputImages_width-1)
    && between(yInFrontTopLeft, 0, inputImages_height-1) && between(zInFrontTopLeft+1, 0, inputImages_depth-1);

   bool frontBottomLeftIsIn = between(xInFrontTopLeft, 0, inputImages_width-1)
    && between(yInFrontTopLeft+1, 0, inputImages_height-1) && between(zInFrontTopLeft, 0, inputImages_depth-1);

   bool backBottomLeftIsIn = between(xInFrontTopLeft, 0, inputImages_width-1)
    && between(yInFrontTopLeft+1, 0, inputImages_height-1) && between(zInFrontTopLeft+1, 0, inputImages_depth-1);

   bool frontBottomRightIsIn = between(xInFrontTopLeft+1, 0, inputImages_width-1)
    && between(yInFrontTopLeft+1, 0, inputImages_height-1) && between(zInFrontTopLeft, 0, inputImages_depth-1);

   bool backBottomRightIsIn = between(xInFrontTopLeft+1, 0, inputImages_width-1)
    && between(yInFrontTopLeft+1, 0, inputImages_height-1) && between(zInFrontTopLeft+1, 0, inputImages_depth-1);

   // interpolation happens here
   for(int t=threadIdx.x; t<inputImages_channels; t+= blockDim.x)
   {
      if(frontTopLeftIsIn) inFrontTopLeft = inputImages_data[inFrontTopLeftAddress + t];
      if(frontTopRightIsIn) inFrontTopRight = inputImages_data[inFrontTopRightAddress + t];
      if(frontBottomLeftIsIn) inFrontBottomLeft = inputImages_data[inFrontBottomLeftAddress + t];
      if(frontBottomRightIsIn) inFrontBottomRight = inputImages_data[inFrontBottomRightAddress + t];

      if(backTopLeftIsIn) inBackTopLeft = inputImages_data[inBackTopLeftAddress + t];
      if(backTopRightIsIn) inBackTopRight = inputImages_data[inBackTopRightAddress + t];
      if(backBottomLeftIsIn) inBackBottomLeft = inputImages_data[inBackBottomLeftAddress + t];
      if(backBottomRightIsIn) inBackBottomRight = inputImages_data[inBackBottomRightAddress + t];

      v = xWeightFrontTopLeft * yWeightFrontTopLeft * zWeightFrontTopLeft * inFrontTopLeft
        + (1 - xWeightFrontTopLeft) * yWeightFrontTopLeft * zWeightFrontTopLeft * inFrontTopRight
        + xWeightFrontTopLeft * (1 - yWeightFrontTopLeft) * zWeightFrontTopLeft * inFrontBottomLeft
        + (1 - xWeightFrontTopLeft) * (1 - yWeightFrontTopLeft) * zWeightFrontTopLeft * inFrontBottomRight
        + xWeightFrontTopLeft * yWeightFrontTopLeft * (1 - zWeightFrontTopLeft) * inBackTopLeft
        + (1 - xWeightFrontTopLeft) * yWeightFrontTopLeft * (1-zWeightFrontTopLeft) * inBackTopRight
        + xWeightFrontTopLeft * (1 - yWeightFrontTopLeft) * (1-zWeightFrontTopLeft) * inBackBottomLeft
        + (1 - xWeightFrontTopLeft) * (1 - yWeightFrontTopLeft) * (1-zWeightFrontTopLeft) * inBackBottomRight;

      output_data[outAddress + t] = v;
   }
}


#ifdef __cplusplus
}
#endif
