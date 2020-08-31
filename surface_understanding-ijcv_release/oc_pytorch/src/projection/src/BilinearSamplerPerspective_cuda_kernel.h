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
                                                 /*THCState_getCurrentStream(state)*/cudaStream_t stream);

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
                                                    /*THCState_getCurrentStream(state)*/cudaStream_t stream);



#ifdef __cplusplus
}
#endif
