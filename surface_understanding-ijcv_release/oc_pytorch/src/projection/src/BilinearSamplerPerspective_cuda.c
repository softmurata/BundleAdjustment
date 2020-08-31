#include <THC/THC.h>
#include <stdbool.h>
#include <stdio.h>
#include "BilinearSamplerPerspective_cuda_kernel.h"

#define real float

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids

int BilinearSamplerBZHWD_updateOutput_cuda(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *output)
{
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
  // printf("%2f", output->size[0]);
  int success = 0;
  success = BilinearSamplerBZHWD_updateOutput_cuda_kernel(output->size[3],
					       output->size[2],
                                               output->size[0],
                                               output->size[1],
                                               THCudaTensor_size(state, inputImages, 4),
                                               THCudaTensor_size(state, inputImages, 1),
                                               THCudaTensor_size(state, inputImages, 2),
                                               THCudaTensor_size(state, inputImages, 3),
                                               THCudaTensor_size(state, output, 1),
                                               THCudaTensor_size(state, output, 3),
                                               THCudaTensor_data(state, inputImages),
                                               THCudaTensor_stride(state, inputImages, 0),
                                               THCudaTensor_stride(state, inputImages, 4),
                                               THCudaTensor_stride(state, inputImages, 1),
                                               THCudaTensor_stride(state, inputImages, 2),
                                               THCudaTensor_stride(state, inputImages, 3),
                                               THCudaTensor_data(state, grids), 
                                               THCudaTensor_stride(state, grids, 0),
                                               THCudaTensor_stride(state, grids, 4),
                                               THCudaTensor_stride(state, grids, 1),
                                               THCudaTensor_stride(state, grids, 2),
                                               THCudaTensor_stride(state, grids, 3),
                                               THCudaTensor_data(state, output), 
                                               THCudaTensor_stride(state, output, 0),
                                               THCudaTensor_stride(state, output, 4),
                                               THCudaTensor_stride(state, output, 1),
                                               THCudaTensor_stride(state, output, 2),
                                               THCudaTensor_stride(state, output, 3),
                                               THCState_getCurrentStream(state));

  //check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}

int BilinearSamplerBZHWD_updateGradInput_cuda(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *gradInputImages,
                                        THCudaTensor *gradGrids, THCudaTensor *gradOutput)
{
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *gradInputImages = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
//  THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
//  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");

  int success = 0;
  success = BilinearSamplerBZHWD_updateGradInput_cuda_kernel(gradOutput->size[3], gradOutput->size[2],
                                                  gradOutput->size[0],
                                                  gradOutput->size[1],
                                                  THCudaTensor_size(state, inputImages, 4),
                                                  THCudaTensor_size(state, inputImages, 1),
                                                  THCudaTensor_size(state, inputImages, 2),
                                                  THCudaTensor_size(state, inputImages, 3),
                                                  THCudaTensor_size(state, gradOutput, 1),
                                                  THCudaTensor_size(state, gradOutput, 3),
                                                  THCudaTensor_data(state, inputImages),
                                                  THCudaTensor_stride(state, inputImages, 0), 
                                                  THCudaTensor_stride(state, inputImages, 4),
                                                  THCudaTensor_stride(state, inputImages, 1),
                                                  THCudaTensor_stride(state, inputImages, 2),
                                                  THCudaTensor_stride(state, inputImages, 3),
                                                  THCudaTensor_data(state, grids), 
                                                  THCudaTensor_stride(state, grids, 0),
                                                  THCudaTensor_stride(state, grids, 4),
                                                  THCudaTensor_stride(state, grids, 1),
                                                  THCudaTensor_stride(state, grids, 2),
                                                  THCudaTensor_stride(state, grids, 3),
                                                  THCudaTensor_data(state, gradInputImages), 
                                                  THCudaTensor_stride(state, gradInputImages, 0),
                                                  THCudaTensor_stride(state, gradInputImages, 4),
                                                  THCudaTensor_stride(state, gradInputImages, 1),
                                                  THCudaTensor_stride(state, gradInputImages, 2),
                                                  THCudaTensor_stride(state, gradInputImages, 3),
                                                  THCudaTensor_data(state, gradGrids), 
                                                  THCudaTensor_stride(state, gradGrids, 0),
                                                  THCudaTensor_stride(state, gradGrids, 4),
                                                  THCudaTensor_stride(state, gradGrids, 1),
                                                  THCudaTensor_stride(state, gradGrids, 2),
                                                  THCudaTensor_stride(state, gradGrids, 3),
                                                  THCudaTensor_data(state, gradOutput), 
                                                  THCudaTensor_stride(state, gradOutput, 0),
                                                  THCudaTensor_stride(state, gradOutput, 4),
                                                  THCudaTensor_stride(state, gradOutput, 1),
                                                  THCudaTensor_stride(state, gradOutput, 2),
                                                  THCudaTensor_stride(state, gradOutput, 3),
                                                  THCState_getCurrentStream(state));

    if (!success) {
      THError("aborting");
    }
    return 1;

}

