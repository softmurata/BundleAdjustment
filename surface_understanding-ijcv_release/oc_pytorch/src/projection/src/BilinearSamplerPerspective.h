// # Adapted from stn example on github : https://github.com/fxia22/stn.pytorch/blob/master/script/build.py
// and from ptnbhwd: https://github.com/xcyan/ptnbhwd/blob/master/generic/BilinearSamplerPerspective.c

int BilinearSamplerBZHWD_updateOutput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output);

int BilinearSamplerBZHWD_updateGradInput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
                                        THFloatTensor *gradGrids, THFloatTensor *gradOutput);