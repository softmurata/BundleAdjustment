// # Adapted from stn example on github : https://github.com/fxia22/stn.pytorch/blob/master/script/build.py
// and from ptnbhwd: https://github.com/xcyan/ptnbhwd/blob/master/generic/BilinearSamplerPerspective.c

#include <TH/TH.h>
#include <stdbool.h>
#include <stdio.h>

#define real float

int BilinearSamplerBZHWD_updateOutput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output)
{

  int batchsize = inputImages->size[0];
  int inputImages_depth = inputImages->size[1];
  int inputImages_height = inputImages->size[2];
  int inputImages_width = inputImages->size[3];
  int output_height = output->size[2];
  int output_width = output->size[3];
  int output_depth = output->size[1];
  int inputImages_channels = inputImages->size[4];

  int output_strideBatch = output->stride[0];
  int output_strideHeight = output->stride[2];
  int output_strideWidth = output->stride[3];
  int output_strideDepth = output->stride[1];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideHeight = inputImages->stride[2];
  int inputImages_strideWidth = inputImages->stride[3];
  int inputImages_strideDepth = inputImages->stride[1];
    
  int grids_strideBatch = grids->stride[0];
  int grids_strideHeight = grids->stride[2];
  int grids_strideWidth = grids->stride[3];
  int grids_strideDepth = grids->stride[1];

  real *inputImages_data, *output_data, *grids_data;
  inputImages_data = THFloatTensor_data(inputImages);
  output_data = THFloatTensor_data(output);
  grids_data = THFloatTensor_data(grids);

  int b, yOut, xOut, zOut;

  for(b=0; b < batchsize; b++)
  {
    for(zOut = 0; zOut < output_depth; zOut++)
    {    
      for(yOut=0; yOut < output_height; yOut++) 
      {
        for(xOut=0; xOut < output_width; xOut++) 
        {
           //read the grid
        real yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + zOut*grids_strideDepth + 1];
        real xf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + zOut*grids_strideDepth + 2];
        real zf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + zOut*grids_strideDepth];

        // get the weights for interpolation
        int yInTopLeftFront, xInTopLeftFront, zInTopLeftFront;
        real yWeightTopLeftFront, xWeightTopLeftFront, zWeightTopLeftFront;
 
        real xcoord = (xf + 1) * (inputImages_width - 1) / 2;
        xInTopLeftFront = floor(xcoord);
        xWeightTopLeftFront = 1 - (xcoord - xInTopLeftFront);

        real ycoord = (yf + 1) * (inputImages_height - 1) / 2;
        yInTopLeftFront = floor(ycoord);
        yWeightTopLeftFront = 1 - (ycoord - yInTopLeftFront);
            
        real zcoord = (zf + 1) * (inputImages_depth - 1) / 2;
        zInTopLeftFront = floor(zcoord);
        zWeightTopLeftFront = 1 - (zcoord - zInTopLeftFront);

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
            
        real v=0;
        real inTopLeftFront=0;
        real inTopLeftBack=0;
        real inTopRightFront=0;
        real inTopRightBack=0;
        real inBottomLeftFront=0;
        real inBottomLeftBack=0;
        real inBottomRightFront=0;
        real inBottomRightBack=0;

        // we are careful with the boundaries
        bool topLeftFrontIsIn = (xInTopLeftFront >= 0 && xInTopLeftFront <= inputImages_width-1 
          && yInTopLeftFront >= 0 && yInTopLeftFront<= inputImages_height-1 
          && zInTopLeftFront >=0 && zInTopLeftFront <= inputImages_depth-1);
            
        bool topLeftBackIsIn = (xInTopLeftFront >= 0 && xInTopLeftFront <= inputImages_width-1 
          && yInTopLeftFront >= 0 && yInTopLeftFront<= inputImages_height-1 
          && zInTopLeftFront+1 >=0 && zInTopLeftFront+1<= inputImages_depth-1);
            
        bool topRightFrontIsIn = xInTopLeftFront+1 >= 0 && xInTopLeftFront+1 <= inputImages_width-1 
          && yInTopLeftFront >= 0 && yInTopLeftFront <= inputImages_height-1 
          && zInTopLeftFront >=0 && zInTopLeftFront <= inputImages_depth - 1;

        bool topRightBackIsIn = (xInTopLeftFront+1 >= 0 && xInTopLeftFront+1 <= inputImages_width-1 
          && yInTopLeftFront >= 0 && yInTopLeftFront <= inputImages_height-1 
          && zInTopLeftFront+1 >=0 && zInTopLeftFront+1 <= inputImages_depth-1);
        
        bool bottomLeftFrontIsIn = (xInTopLeftFront >= 0 && xInTopLeftFront <= inputImages_width-1 
          && yInTopLeftFront+1 >= 0 && yInTopLeftFront+1 <= inputImages_height-1 
          && zInTopLeftFront >=0 && zInTopLeftFront <= inputImages_depth-1);
            
        bool bottomLeftBackIsIn = (xInTopLeftFront >= 0 && xInTopLeftFront <= inputImages_width-1 
          && yInTopLeftFront+1 >= 0 && yInTopLeftFront+1 <= inputImages_height-1 
          && zInTopLeftFront+1 >=0 && zInTopLeftFront+1 <= inputImages_depth-1);
            
        bool bottomRightFrontIsIn = (xInTopLeftFront+1 >= 0 && xInTopLeftFront+1 <= inputImages_width-1 
          && yInTopLeftFront+1 >= 0 && yInTopLeftFront+1 <= inputImages_height-1 
          && zInTopLeftFront >=0 && zInTopLeftFront <= inputImages_depth-1);
    
        bool bottomRightBackIsIn = (xInTopLeftFront+1 >= 0 && xInTopLeftFront+1 <= inputImages_width-1 
          && yInTopLeftFront+1 >= 0 && yInTopLeftFront+1 <= inputImages_height-1 
          && zInTopLeftFront+1 >=0 && zInTopLeftFront+1 <= inputImages_depth-1);

        int t;
        // interpolation happens here
        for(t=0; t<inputImages_channels; t++)
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
      }
    }
  }

  return 1;
}




int BilinearSamplerBZHWD_updateGradInput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
                                        THFloatTensor *gradGrids, THFloatTensor *gradOutput)
{

  bool onlyGrid=false;

  int batchsize = inputImages->size[0];
  int inputImages_height = inputImages->size[2];
  int inputImages_width = inputImages->size[3];
  int inputImages_depth = inputImages->size[1];
    
  int gradOutput_height = gradOutput->size[2];
  int gradOutput_width = gradOutput->size[3];
  int gradOutput_depth = gradOutput->size[1];
  int inputImages_channels = inputImages->size[4];

  int gradOutput_strideBatch = gradOutput->stride[0];
  int gradOutput_strideHeight = gradOutput->stride[2];
  int gradOutput_strideWidth = gradOutput->stride[3];
  int gradOutput_strideDepth = gradOutput->stride[1];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideHeight = inputImages->stride[2];
  int inputImages_strideWidth = inputImages->stride[3];
  int inputImages_strideDepth = inputImages->stride[1];

  int gradInputImages_strideBatch = gradInputImages->stride[0];
  int gradInputImages_strideHeight = gradInputImages->stride[2];
  int gradInputImages_strideWidth = gradInputImages->stride[3];
  int gradInputImages_strideDepth = gradInputImages->stride[1];

  int grids_strideBatch = grids->stride[0];
  int grids_strideHeight = grids->stride[2];
  int grids_strideWidth = grids->stride[3];
  int grids_strideDepth = grids->stride[1];

  int gradGrids_strideBatch = gradGrids->stride[0];
  int gradGrids_strideHeight = gradGrids->stride[2];
  int gradGrids_strideWidth = gradGrids->stride[3];
  int gradGrids_strideDepth = gradGrids->stride[1];

  real *inputImages_data, *gradOutput_data, *grids_data, *gradGrids_data, *gradInputImages_data;
  inputImages_data = THFloatTensor_data(inputImages);
  gradOutput_data = THFloatTensor_data(gradOutput);
  grids_data = THFloatTensor_data(grids);
  gradGrids_data = THFloatTensor_data(gradGrids);
  gradInputImages_data = THFloatTensor_data(gradInputImages);

    int b, yOut, xOut, zOut;

  for(b=0; b < batchsize; b++)
  {
    for(zOut=0; zOut < gradOutput_depth; zOut++)
    {
      for(yOut=0; yOut < gradOutput_height; yOut++)
      {
        for(xOut=0; xOut < gradOutput_width; xOut++)
        {
          //read the grid
        real yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + zOut*grids_strideDepth + 1];
        real xf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + zOut*grids_strideDepth + 2];
        real zf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + zOut*grids_strideDepth];
        
        // get the weights for interpolation
        int yInTopLeftFront, xInTopLeftFront, zInTopLeftFront;
        real yWeightTopLeftFront, xWeightTopLeftFront, zWeightTopLeftFront;
 
        real xcoord = (xf + 1) * (inputImages_width - 1) / 2;
        xInTopLeftFront = floor(xcoord);
        xWeightTopLeftFront = 1 - (xcoord - xInTopLeftFront);

        real ycoord = (yf + 1) * (inputImages_height - 1) / 2;
        yInTopLeftFront = floor(ycoord);
        yWeightTopLeftFront = 1 - (ycoord - yInTopLeftFront);
            
        real zcoord = (zf + 1) * (inputImages_depth - 1) / 2;
        zInTopLeftFront = floor(zcoord);
        zWeightTopLeftFront = 1 - (zcoord - zInTopLeftFront);
            
        const int inTopLeftFrontAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeftFront 
          + inputImages_strideWidth * xInTopLeftFront + inputImages_strideDepth * zInTopLeftFront;
  
        const int inTopLeftBackAddress = inTopLeftFrontAddress + inputImages_strideDepth;
            
        const int inTopRightFrontAddress = inTopLeftFrontAddress + inputImages_strideWidth;
        const int inTopRightBackAddress = inTopRightFrontAddress + inputImages_strideDepth;
            
        const int inBottomLeftFrontAddress = inTopLeftFrontAddress + inputImages_strideHeight;
        const int inBottomLeftBackAddress = inBottomLeftFrontAddress + inputImages_strideDepth;
            
        const int inBottomRightFrontAddress = inBottomLeftFrontAddress + inputImages_strideWidth;
        const int inBottomRightBackAddress = inBottomRightFrontAddress + inputImages_strideDepth;

        const int gradInputImagesTopLeftFrontAddress = gradInputImages_strideBatch * b + gradInputImages_strideHeight * yInTopLeftFront 
          + gradInputImages_strideWidth * xInTopLeftFront + gradInputImages_strideDepth * zInTopLeftFront;
        const int gradInputImagesTopLeftBackAddress = gradInputImagesTopLeftFrontAddress + gradInputImages_strideDepth;
            
        const int gradInputImagesTopRightFrontAddress = gradInputImagesTopLeftFrontAddress + gradInputImages_strideWidth;
        const int gradInputImagesTopRightBackAddress = gradInputImagesTopRightFrontAddress + gradInputImages_strideDepth;
        
        const int gradInputImagesBottomLeftFrontAddress = gradInputImagesTopLeftFrontAddress + gradInputImages_strideHeight;
        const int gradInputImagesBottomLeftBackAddress = gradInputImagesBottomLeftFrontAddress +gradInputImages_strideDepth;
            
        const int gradInputImagesBottomRightFrontAddress = gradInputImagesBottomLeftFrontAddress + gradInputImages_strideWidth;
        const int gradInputImagesBottomRightBackAddress = gradInputImagesBottomRightFrontAddress + gradInputImages_strideDepth;

        const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideHeight * yOut 
          + gradOutput_strideWidth * xOut + gradOutput_strideDepth * zOut;

        real topLeftFrontDotProduct = 0;
        real topLeftBackDotProduct = 0;
        real topRightFrontDotProduct = 0;
        real topRightBackDotProduct = 0;
            
        real bottomLeftFrontDotProduct = 0;
        real bottomLeftBackDotProduct = 0;
        real bottomRightFrontDotProduct = 0;
        real bottomRightBackDotProduct = 0;

        real v=0;
        real inTopLeftFront=0;
        real inTopLeftBack=0;
        real inTopRightFront=0;
        real inTopRightBack=0;

        real inBottomLeftFront=0;
        real inBottomLeftBack=0;
        real inBottomRightFront=0;
        real inBottomRightBack=0;
      
        // we are careful with the boundaries
        bool topLeftFrontIsIn = (xInTopLeftFront >= 0 && xInTopLeftFront <= inputImages_width-1 
          && yInTopLeftFront >= 0 && yInTopLeftFront<= inputImages_height-1 
          && zInTopLeftFront >=0 && zInTopLeftFront <= inputImages_depth-1);
            
        bool topLeftBackIsIn = (xInTopLeftFront >= 0 && xInTopLeftFront <= inputImages_width-1 
          && yInTopLeftFront >= 0 && yInTopLeftFront<= inputImages_height-1 
          && zInTopLeftFront+1 >=0 && zInTopLeftFront+1<= inputImages_depth-1);
            
        bool topRightFrontIsIn = (xInTopLeftFront+1 >= 0 && xInTopLeftFront+1 <= inputImages_width-1 
          && yInTopLeftFront >= 0 && yInTopLeftFront <= inputImages_height-1 
          && zInTopLeftFront >=0 && zInTopLeftFront <= inputImages_depth-1);
            
        bool topRightBackIsIn = (xInTopLeftFront+1 >= 0 && xInTopLeftFront+1 <= inputImages_width-1 
          && yInTopLeftFront >= 0 && yInTopLeftFront <= inputImages_height-1 
          && zInTopLeftFront+1 >=0 && zInTopLeftFront+1 <= inputImages_depth-1);
            
        bool bottomLeftFrontIsIn = (xInTopLeftFront >= 0 && xInTopLeftFront <= inputImages_width-1 
          && yInTopLeftFront+1 >= 0 && yInTopLeftFront+1 <= inputImages_height-1 
          && zInTopLeftFront >=0 && zInTopLeftFront <= inputImages_depth-1);
            
        bool bottomLeftBackIsIn = (xInTopLeftFront >= 0 && xInTopLeftFront <= inputImages_width-1 
          && yInTopLeftFront+1 >= 0 && yInTopLeftFront+1 <= inputImages_height-1 
          && zInTopLeftFront+1 >=0 && zInTopLeftFront+1 <= inputImages_depth-1);
            
        bool bottomRightFrontIsIn = (xInTopLeftFront+1 >= 0 && xInTopLeftFront+1 <= inputImages_width-1 
          && yInTopLeftFront+1 >= 0 && yInTopLeftFront+1 <= inputImages_height-1 
          && zInTopLeftFront >=0 && zInTopLeftFront <= inputImages_depth-1);
 
        bool bottomRightBackIsIn = (xInTopLeftFront+1 >= 0 && xInTopLeftFront+1 <= inputImages_width-1 
          && yInTopLeftFront+1 >= 0 && yInTopLeftFront+1 <= inputImages_height-1 
          && zInTopLeftFront+1 >=0 && zInTopLeftFront+1 <= inputImages_depth-1);
        int t;
        
        for(t=0; t<inputImages_channels; t++)
        {
           real gradOutValue = gradOutput_data[gradOutputAddress + t];
           if(topLeftFrontIsIn)
           {
              real inTopLeftFront = inputImages_data[inTopLeftFrontAddress + t];
              topLeftFrontDotProduct += inTopLeftFront * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopLeftFrontAddress + t] += 
                xWeightTopLeftFront * yWeightTopLeftFront * zWeightTopLeftFront * gradOutValue;
           }
           if(topLeftBackIsIn)
           {
              real inTopLeftBack = inputImages_data[inTopLeftBackAddress + t];
              topLeftBackDotProduct += inTopLeftBack * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopLeftBackAddress + t] += 
                xWeightTopLeftFront * yWeightTopLeftFront * (1-zWeightTopLeftFront) * gradOutValue;
           }

           if(topRightFrontIsIn)
           {
              real inTopRightFront = inputImages_data[inTopRightFrontAddress + t];
              topRightFrontDotProduct += inTopRightFront * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopRightFrontAddress + t] += 
                (1 - xWeightTopLeftFront) * yWeightTopLeftFront * zWeightTopLeftFront * gradOutValue;
           }
           if(topRightBackIsIn)
           {
              real inTopRightBack = inputImages_data[inTopRightBackAddress + t];
              topRightBackDotProduct += inTopRightBack * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopRightBackAddress + t] += 
                (1 - xWeightTopLeftFront) * yWeightTopLeftFront * (1-zWeightTopLeftFront) * gradOutValue;
           }
           
           if(bottomLeftFrontIsIn)
           {
              real inBottomLeftFront = inputImages_data[inBottomLeftFrontAddress + t];
              bottomLeftFrontDotProduct += inBottomLeftFront * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomLeftFrontAddress + t] += 
                xWeightTopLeftFront * (1 - yWeightTopLeftFront) * zWeightTopLeftFront * gradOutValue;
           }
           if(bottomLeftBackIsIn)
           {
              real inBottomLeftBack = inputImages_data[inBottomLeftBackAddress + t];
              bottomLeftBackDotProduct += inBottomLeftBack * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomLeftBackAddress + t] += 
                xWeightTopLeftFront * (1 - yWeightTopLeftFront) * (1-zWeightTopLeftFront) * gradOutValue;
           }
      
            if(bottomRightFrontIsIn)
           {
              real inBottomRightFront = inputImages_data[inBottomRightFrontAddress + t];
              bottomRightFrontDotProduct += inBottomRightFront * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomRightFrontAddress + t] += 
                (1 - xWeightTopLeftFront) * (1 - yWeightTopLeftFront) * zWeightTopLeftFront * gradOutValue;
           }

           if(bottomRightBackIsIn)
           {
              real inBottomRightBack = inputImages_data[inBottomRightBackAddress + t];
              bottomRightBackDotProduct += inBottomRightBack * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomRightBackAddress + t] += 
                (1 - xWeightTopLeftFront) * (1 - yWeightTopLeftFront) * (1-zWeightTopLeftFront) * gradOutValue;
           }
        }

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

        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + zOut*gradGrids_strideDepth + 1] = yf * (inputImages_height-1) / 2;
        
        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + zOut*gradGrids_strideDepth + 2] = xf * (inputImages_width-1) / 2;
            
        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + zOut*gradGrids_strideDepth] = zf * (inputImages_depth-1) / 2;
      }
      }
    }
  }
  return 1;
}
