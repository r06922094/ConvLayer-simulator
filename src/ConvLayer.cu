#include "ConvLayer.h"
#include "tensor.h"
#include <cuda_fp16.h>
__global__ void g_ConvCFM_feedforward_fool(
	float*  inputs,
	float* ws,
	float* bs,
	float*  outputs,
	int inputDim,
	int kernelSize,
	int padding,
	int outputDim,
	int inputAmount,
	int outputAmount);
__global__ void g_ConvCFM_feedforward_IR(
	float*  inputs,
	float* ws,
	float* bs,
	float*  outputs,
	int inputDim,
	int kernelSize,
	int padding,
	int outputDim,
	int inputAmount,
	int outputAmount);
__global__ void g_ConvCFM_feedforward_FR(
	float*  inputs,
	float* ws,
	float* bs,
	float*  outputs,
	int inputDim,
	int kernelSize,
	int padding,
	int outputDim,
	int inputAmount,
	int outputAmount);
__global__ void g_ConvCFM_feedforward_full_IR(
	float*  inputs,
	float* ws,
	float* bs,
	float*  outputs,
	int inputDim,
	int kernelSize,
	int padding,
	int outputDim,
	int inputAmount,
	int outputAmount);
__global__ void g_ConvCFM_feedforward_row_IR(
	float*  inputs,
	float* ws,
	float* bs,
	float*  outputs,
	int inputDim,
	int kernelSize,
	int padding,
	int outputDim,
	int inputAmount,
	int outputAmount);

__global__ void g_ConvCFM_feedforward_mini_IR(
	float*  inputs,
	float* ws,
	float* bs,
	float*  outputs,
	int inputDim,
	int kernelSize,
	int padding,
	int outputDim,
	int inputAmount,
	int outputAmount);


__global__ void g_ConvCFM_feedforward_full_FR(
    float*  inputs,
    float* ws,
    float* bs,
    float*  outputs,
    int inputDim,
    int kernelSize,
    int padding,
    int outputDim,
    int inputAmount,
    int outputAmount);
__global__ void g_ConvCFM_feedforward_row_FR(
    float*  inputs,
    float* ws,
    float* bs,
    float*  outputs,
    int inputDim,
    int kernelSize,
    int padding,
    int outputDim,
    int inputAmount,
    int outputAmount);
__global__ void g_ConvCFM_feedforward_mini_FR(
    float*  inputs,
    float* ws,
    float* bs,
    float*  outputs,
    int inputDim,
    int kernelSize,
    int padding,
    int outputDim,
    int inputAmount,
    int outputAmount);

__global__ void g_ConvCFM_feedforward_full_FRIT(
    float*  inputs,
    float* ws,
    float* bs,
    float*  outputs,
    int inputDim,
    int kernelSize,
    int padding,
    int outputDim,
    int inputAmount,
    int outputAmount);
__global__ void g_ConvCFM_feedforward_row_FRIT(
    float*  inputs,
    float* ws,
    float* bs,
    float*  outputs,
    int inputDim,
    int kernelSize,
    int padding,
    int outputDim,
    int inputAmount,
    int outputAmount);
__global__ void g_ConvCFM_feedforward_mini_FRIT(
    float*  inputs,
    float* ws,
    float* bs,
    float*  outputs,
    int inputDim,
    int kernelSize,
    int padding,
    int outputDim,
    int inputAmount,
    int outputAmount);

void Launch_naive_kernel(ConvLayer *conv);
void Launch_IR_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize);
void Launch_FR_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize);

void Launch_mini_IR_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize);
void Launch_row_IR_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize);
void Launch_full_IR_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize);

void Launch_mini_FR_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize);
void Launch_row_FR_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize);
void Launch_full_FR_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize);

void Launch_mini_FRIT_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize);
void Launch_row_FRIT_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize);
void Launch_full_FRIT_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize);

static void feedforward(LayerBase *thiz)
{
    ConvLayer *conv = (ConvLayer *) thiz;
    /* prepare output on GPU memory */
    tensor_create(&conv->lb->output, conv->lb->batchSize, conv->lb->outputDim, conv->lb->outputDim, conv->lb->outputChannel);
    conv->lb->output->mallocDev(conv->lb->output);

    /* TODO: feedforward implementation */
    /**check shared memory size**/
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int sharedMemSize = (unsigned int)deviceProp.sharedMemPerBlock;

    /**check thread usage**/
    int threadRequire = conv->kernelDim * conv->kernelDim * conv->lb->inputChannel;
    int threadDim_channel = conv->lb->inputChannel;
    if (threadRequire > 1024){
        threadDim_channel = floor(1024/(conv->kernelDim * conv->kernelDim));
    }
    printf("require: %d ,assign thread: %d %d %d\n", threadRequire, conv->kernelDim ,conv->kernelDim , threadDim_channel);
    dim3 block = dim3(conv->lb->batchSize);
    dim3 thread= dim3(threadDim_channel, conv->kernelDim, conv->kernelDim);

    printf("choose policy: %d\n", conv->policy);
    switch(conv->policy){
        case 0:
            Launch_mini_IR_kernel(conv, block, thread, sharedMemSize);
            break;
        case 1:
            Launch_row_IR_kernel(conv, block, thread, sharedMemSize);
            break;
        case 2:
            Launch_full_IR_kernel(conv, block, thread, sharedMemSize);
            break;
        case 3:
            Launch_mini_FR_kernel(conv, block, thread, sharedMemSize);
            break;
        case 4:
            Launch_row_FR_kernel(conv, block, thread, sharedMemSize);
            break;
        case 5:
            Launch_full_FR_kernel(conv, block, thread, sharedMemSize);
            break;
        case 6:
            Launch_mini_FRIT_kernel(conv, block, thread, sharedMemSize);
            break;
        case 7:
            Launch_row_FRIT_kernel(conv, block, thread, sharedMemSize);
            break;
        case 8:
            Launch_full_FRIT_kernel(conv, block, thread, sharedMemSize);
            break;
        case 9:
            Launch_IR_kernel(conv, block, thread, sharedMemSize);
            break;
        case 10:
            Launch_FR_kernel(conv, block, thread, sharedMemSize);
            break;

        default:
            Launch_naive_kernel(conv);
    }
}
void Launch_mini_FR_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize){
    int filterSize = sizeof(half) * conv->kernelDim * conv->kernelDim * conv->lb->inputChannel;
    int suInput    = sizeof(half) * conv->kernelDim * conv->kernelDim * conv->lb->inputChannel;
    int suOutput   = sizeof(half) * conv->lb->outputDim * conv->lb->outputDim * conv->kernelAmount;
    int suFilter   = filterSize;
    int sharedMemRequire = suInput + suOutput + suFilter;
    printf("mini_FR: %d = %d + %d + %d / %d bytes needed\n",sharedMemRequire,suInput,suOutput,suFilter, sharedMemSize);
    if(sharedMemRequire > sharedMemSize){
        printf("mini FR init error\n");
		exit(0);
    }
    
    g_ConvCFM_feedforward_mini_FR<<<block, thread, sharedMemRequire, 0>>>(
        conv->lb->input->devData,
        conv->weight->devData,
        conv->bias->devData,
        conv->lb->output->devData,
        conv->lb->inputDim,
        conv->kernelDim,
        0,
        conv->lb->outputDim,
        conv->lb->inputChannel,
        conv->kernelAmount);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
}
void Launch_row_FR_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize){
    int filterSize = sizeof(half) * conv->kernelDim * conv->kernelDim * conv->lb->inputChannel;
    int suInput    = sizeof(half) * conv->kernelDim * conv->lb->inputDim * conv->lb->inputChannel;
    int suOutput   = sizeof(half) * conv->lb->outputDim * conv->lb->outputDim * conv->kernelAmount;
    int suFilter   = filterSize;
    int sharedMemRequire = suInput + suOutput + suFilter;
    printf("row_FR: %d = %d + %d + %d / %d bytes needed\n",sharedMemRequire,suInput,suOutput,suFilter, sharedMemSize);
    if(suInput + suOutput + suFilter > sharedMemSize){
        printf("row FR init error\n");
		exit(0);
    }
    g_ConvCFM_feedforward_row_FR<<<block, thread, sharedMemRequire, 0>>>(
        conv->lb->input->devData,
        conv->weight->devData,
        conv->bias->devData,
        conv->lb->output->devData,
        conv->lb->inputDim,
        conv->kernelDim,
        0,
        conv->lb->outputDim,
        conv->lb->inputChannel,
        conv->kernelAmount);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

}
void Launch_full_FR_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize){
    int filterSize = sizeof(half) * conv->kernelDim * conv->kernelDim * conv->lb->inputChannel;
    int suInput    = sizeof(half) * conv->lb->inputDim * conv->lb->inputDim * conv->lb->inputChannel;
    int suOutput   = sizeof(half) * conv->lb->outputDim * conv->kernelAmount;
    int suFilter   = filterSize;
    int sharedMemRequire = suInput + suOutput + suFilter;
    printf("full_FR: %d = %d + %d + %d / %d bytes needed\n",sharedMemRequire,suInput,suOutput,suFilter, sharedMemSize);
    if(sharedMemRequire > sharedMemSize){
        printf("full FR init error\n");
		exit(0);
    }
    
    g_ConvCFM_feedforward_full_FR<<<block, thread, sharedMemRequire, 0>>>(
        conv->lb->input->devData,
        conv->weight->devData,
        conv->bias->devData,
        conv->lb->output->devData,
        conv->lb->inputDim,
        conv->kernelDim,
        0,
        conv->lb->outputDim,
        conv->lb->inputChannel,
        conv->kernelAmount);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
}


void Launch_mini_FRIT_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize){
    int filterSize = sizeof(half) * conv->kernelDim * conv->kernelDim * conv->lb->inputChannel;
    int suInput    = filterSize;
    int suOutput   = sizeof(half) * conv->lb->outputDim * conv->kernelAmount;
    int suFilter   = filterSize;
    int sharedMemRequire = suInput + suOutput + suFilter;
    printf("mini_FRIT: %d = %d + %d + %d / %d bytes needed\n",sharedMemRequire,suInput,suOutput,suFilter, sharedMemSize);
    if(sharedMemRequire > sharedMemSize){
        printf("mini FRIT init error\n");
		exit(0);
    }
    g_ConvCFM_feedforward_mini_FRIT<<<block, thread, sharedMemRequire, 0>>>(
        conv->lb->input->devData,
        conv->weight->devData,
        conv->bias->devData,
        conv->lb->output->devData,
        conv->lb->inputDim,
        conv->kernelDim,
        0,
        conv->lb->outputDim,
        conv->lb->inputChannel,
        conv->kernelAmount);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
}
void Launch_row_FRIT_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize){
    int filterSize = sizeof(half) * conv->kernelDim * conv->kernelDim * conv->lb->inputChannel;
    int suInput    = sizeof(half) * conv->kernelDim * conv->lb->inputDim * conv->lb->inputChannel;
    int suOutput   = sizeof(half) * conv->lb->outputDim * conv->kernelAmount;
    int suFilter   = filterSize;
    int sharedMemRequire = suInput + suOutput + suFilter;
    printf("row_FRIT: %d = %d + %d + %d / %d bytes needed\n",sharedMemRequire,suInput,suOutput,suFilter, sharedMemSize);
    if(sharedMemRequire > sharedMemSize){
        printf("row FRIT init error\n");
		exit(0);
    }
    
    g_ConvCFM_feedforward_row_FRIT<<<block, thread, sharedMemRequire, 0>>>(
        conv->lb->input->devData,
        conv->weight->devData,
        conv->bias->devData,
        conv->lb->output->devData,
        conv->lb->inputDim,
        conv->kernelDim,
        0,
        conv->lb->outputDim,
        conv->lb->inputChannel,
        conv->kernelAmount);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
}
void Launch_full_FRIT_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize){
    int filterSize = sizeof(half) * conv->kernelDim * conv->kernelDim * conv->lb->inputChannel;
    int suInput    = sizeof(half) * conv->lb->inputDim * conv->lb->inputDim * conv->lb->inputChannel;
    int suOutput   = sizeof(half) * conv->lb->outputDim * conv->kernelAmount;
    int suFilter   = filterSize;
    int sharedMemRequire = suInput + suOutput + suFilter;
    printf("full_FRIT: %d = %d + %d + %d / %d bytes needed\n",sharedMemRequire,suInput,suOutput,suFilter, sharedMemSize);
    if(sharedMemRequire > sharedMemSize){
        printf("full FRIT init error\n");
		exit(0);
    }
    
    g_ConvCFM_feedforward_full_FRIT<<<block, thread, sharedMemRequire, 0>>>(
        conv->lb->input->devData,
        conv->weight->devData,
        conv->bias->devData,
        conv->lb->output->devData,
        conv->lb->inputDim,
        conv->kernelDim,
        0,
        conv->lb->outputDim,
        conv->lb->inputChannel,
        conv->kernelAmount);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
}


void Launch_mini_IR_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize){
    
    int filterSize = sizeof(half) * conv->kernelDim * conv->kernelDim * conv->lb->inputChannel;
    int suInput    = filterSize;
    int suOutput   = sizeof(half) * conv->kernelAmount;
    int suFilter   = filterSize;
    int sharedMemRequire = suInput + suOutput + suFilter;
    printf("mini_IR: %d = %d + %d + %d / %d bytes needed\n",sharedMemRequire,suInput,suOutput,suFilter, sharedMemSize);
    if(sharedMemRequire > sharedMemSize){
        printf("mini IR init error\n");
		exit(0);
    }
    /**launch kernel**/
    g_ConvCFM_feedforward_mini_IR<<<block, thread, sharedMemRequire>>>(
        conv->lb->input->devData,
        conv->weight->devData,
        conv->bias->devData,
        conv->lb->output->devData,
        conv->lb->inputDim,
        conv->kernelDim,
        0,
        conv->lb->outputDim,
        conv->lb->inputChannel,
        conv->kernelAmount);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    
}
void Launch_row_IR_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize){
    
    int filterSize = sizeof(half) * conv->kernelDim * conv->kernelDim * conv->lb->inputChannel;
    int suInput    = sizeof(half) * conv->kernelDim * conv->lb->inputDim * conv->lb->inputChannel;
    int suOutput   = sizeof(half) * conv->kernelAmount;
    int suFilter   = filterSize;
    int sharedMemRequire = suInput + suOutput + suFilter;
    printf("row_IR: %d = %d + %d + %d / %d bytes needed\n",sharedMemRequire,suInput,suOutput,suFilter, sharedMemSize);
    if(sharedMemRequire > sharedMemSize){
        printf("row IR init error\n");
		exit(0);
    }
    
    g_ConvCFM_feedforward_row_IR<<<block, thread, sharedMemRequire, 0>>>(
        conv->lb->input->devData,
        conv->weight->devData,
        conv->bias->devData,
        conv->lb->output->devData,
        conv->lb->inputDim,
        conv->kernelDim,
        0,
        conv->lb->outputDim,
        conv->lb->inputChannel,
        conv->kernelAmount);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
}
void Launch_full_IR_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize){
    int filterSize = sizeof(half) * conv->kernelDim * conv->kernelDim * conv->lb->inputChannel;
    int suInput    = sizeof(half) * conv->lb->inputDim * conv->lb->inputDim * conv->lb->inputChannel;
    int suOutput   = sizeof(half) * conv->kernelAmount;
    int suFilter   = filterSize;
    int sharedMemRequire = suInput + suOutput + suFilter;
    printf("full_IR: %d = %d + %d + %d / %d bytes needed\n",sharedMemRequire,suInput,suOutput,suFilter, sharedMemSize);
    if(sharedMemRequire > sharedMemSize){
        printf("full IR init error\n");
		exit(0);
    }
    
    g_ConvCFM_feedforward_full_IR<<<block, thread, sharedMemRequire, 0>>>(
        conv->lb->input->devData,
        conv->weight->devData,
        conv->bias->devData,
        conv->lb->output->devData,
        conv->lb->inputDim,
        conv->kernelDim,
        0,
        conv->lb->outputDim,
        conv->lb->inputChannel,
        conv->kernelAmount);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
}

void Launch_naive_kernel(ConvLayer *conv){
    dim3 block = dim3(conv->lb->batchSize);
    dim3 thread= dim3(1024);
    g_ConvCFM_feedforward_fool<<<block, thread>>>(
        conv->lb->input->devData,
        conv->weight->devData,
        conv->bias->devData,
        conv->lb->output->devData,
        conv->lb->inputDim,
        conv->kernelDim,
        0,
        conv->lb->outputDim,
        conv->lb->inputChannel,
        conv->kernelAmount);
}

void Launch_IR_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize){
    int filterSize = sizeof(half) * conv->kernelDim * conv->kernelDim * conv->lb->inputChannel;
    int suFilter   = filterSize;
    int sharedMemRequire = suFilter;
    printf("IR: %d / %d bytes needed\n",sharedMemRequire, sharedMemSize);
    if(sharedMemRequire > sharedMemSize){
        printf("IR init error\n");
		exit(0);
    }
    /**launch kernel**/
    g_ConvCFM_feedforward_IR<<<block, thread, sharedMemRequire>>>(
        conv->lb->input->devData,
        conv->weight->devData,
        conv->bias->devData,
        conv->lb->output->devData,
        conv->lb->inputDim,
        conv->kernelDim,
        0,
        conv->lb->outputDim,
        conv->lb->inputChannel,
        conv->kernelAmount);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    
}
void Launch_FR_kernel(ConvLayer *conv, dim3 block, dim3 thread, int sharedMemSize){
    int filterSize = sizeof(half) * conv->kernelDim * conv->kernelDim * conv->lb->inputChannel;
    int suFilter   = filterSize;
    int sharedMemRequire = suFilter;
    printf("FR: %d / %d bytes needed\n",sharedMemRequire, sharedMemSize);
    if(sharedMemRequire > sharedMemSize){
        printf("FR init error\n");
		exit(0);
    }
    /**launch kernel**/
    g_ConvCFM_feedforward_FR<<<block, thread, sharedMemRequire>>>(
        conv->lb->input->devData,
        conv->weight->devData,
        conv->bias->devData,
        conv->lb->output->devData,
        conv->lb->inputDim,
        conv->kernelDim,
        0,
        conv->lb->outputDim,
        conv->lb->inputChannel,
        conv->kernelAmount);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    
}
void ConvLayer_init(ConvLayer **thiz, int batchSize, \
                    int inputDim, int inputChannel, \
                    int kernelDim, int kernelAmount, \
                    LayerBase *preLayer, LayerBase *nextLayer, int policy)
{
    (*thiz) = (ConvLayer *) malloc(sizeof(ConvLayer));
    if (!(*thiz)) {
        printf("ConvLayer.cu: No available Memory\n");
        exit(0);
    }

    (*thiz)->lb = (LayerBase *) malloc(sizeof(LayerBase));
    if (!(*thiz)->lb) {
        printf("ConvLayer.cu: No availablle Memory\n");
        exit(0);
    }

    /* LayerBase */
    LayerBase *base = (*thiz)->lb;
    base->batchSize = batchSize;
    base->inputDim = inputDim;
    /* Padding*/
    base->outputDim = inputDim ;//- kernelDim + 1;
    base->inputChannel = inputChannel;
    base->outputChannel = kernelAmount;
    base->input = NULL;
    base->output = NULL;
    base->preLayer = preLayer;
    base->nextLayer = NULL;
    base->feedforward = feedforward;
    /* ConvLayer */
    (*thiz)->kernelDim = kernelDim;
    (*thiz)->kernelAmount = kernelAmount;
    (*thiz)->policy = policy;
    /* TODO: Initialize Weights and bias */
    ConvLayer_weight_init(*thiz);
    ConvLayer_bias_init(*thiz);
}

void ConvLayer_weight_init(ConvLayer *thiz)
{
    tensor_create(&thiz->weight, thiz->kernelAmount, thiz->kernelDim, thiz->kernelDim, thiz->lb->inputChannel);
    tensor *tzr = thiz->weight;
    float Gx_array[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};

    for (int i = 0; i < tzr->D0; i++) {
        for (int j = 0; j < tzr->D1; j++) {
            for (int k = 0; k < tzr->D2; k++) {
                for (int w = 0; w < tzr->D3; w++) {
                    //tzr->set(tzr, i, j, k, w, Gx_array[j][k]);
                    
                    if(w==i){
                        tzr->set(tzr, i, j, k, w, 1);
                    }
                    else{
                        tzr->set(tzr, i, j, k, w, 0);
                    }
                    
                    
                }
            }
        }
    }
    tzr->mallocDev(tzr);
    tzr->toGpu(tzr);
}

void ConvLayer_bias_init(ConvLayer *thiz)
{
    tensor_create(&thiz->bias, thiz->kernelAmount, 1, 1, 1);
    tensor *tzr = thiz->bias;
    for (int i = 0; i < tzr->D0; i++) {
        tzr->set(tzr, i, 0, 0, 0, 0);
    }
    tzr->mallocDev(tzr);
    tzr->toGpu(tzr);
}
__global__ void g_ConvCFM_feedforward_full_FR(
    float*  inputs,
    float* ws,
    float* bs,
    float*  outputs,
    int inputDim,
    int kernelSize,
    int padding,
    int outputDim,
    int inputAmount,
    int outputAmount)
{
    int sp = blockIdx.x;
    int inputSize2  = inputDim* inputDim;
    int outputSize2 = outputDim * outputDim;
    int kernelSize2 = kernelSize * kernelSize;
    int inputArea  = inputSize2 * inputAmount;
    int outputArea = outputSize2* outputAmount;
    int k_idx = threadIdx.z + threadIdx.y*kernelSize;
    int tid = threadIdx.z + threadIdx.y*kernelSize + threadIdx.x*kernelSize2;
    //declare shared mem
    extern __shared__ half bufferShared[];
    //point to diff addr
    int filterBufferSize = blockDim.x  * blockDim.y  * blockDim.z;
    int suInput = kernelSize * inputDim * inputAmount;
    int suOutput = outputDim * outputDim * outputAmount;
    half *inputShared  = bufferShared;
    half *outputShared = bufferShared + suInput;
    half *filterShared = bufferShared + suInput + suOutput;

    float t_w_array[5];
    float* curInput = inputs + sp * inputArea;
    float* curOutput = outputs + sp * outputArea;
    /**Buffer Loading Start**/
    for(int loadChunk=0; loadChunk < inputArea; loadChunk+=filterBufferSize){
        if(loadChunk + tid < inputArea){
            inputShared[loadChunk + tid] = __float2half(curInput[loadChunk + tid]);
        }
    }
    /**Buffer Loading End**/
    for(int ok=0; ok<outputAmount; ok++){//ok: filter
        /**Load Filter Start**/
        float b = bs[ok];
        float* w = ws + ok * kernelSize2 * inputAmount;

        for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
            int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
            if(t_ch_idx < inputAmount){
                t_w_array[ch_chunk_idx] = w[k_idx*inputAmount + t_ch_idx];
            }
        }
        /**Load Filter End**/
        for(int i=0; i<inputDim-kernelSize+1; i++){//i: input column
            for(int j=0; j<inputDim-kernelSize+1; j++){//j:row
                int xx = j+threadIdx.z;
                int yy = i+threadIdx.y;
                int inputBuffOffset_Idx = (xx + yy*inputDim)*inputAmount + threadIdx.x;

                /**Conv Start**/
                half tmp_val = 0;
                for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
                    int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
                    if(t_ch_idx < inputAmount){

                        filterShared[tid] = __hmul(inputShared[inputBuffOffset_Idx], __float2half(t_w_array[ch_chunk_idx]));
                        __syncthreads();
                        //reduction
                        int activeBuffSize = filterBufferSize;
                        for (int stride = ceil(filterBufferSize / 2.0); stride > 1; stride = ceil(stride/2.0)) {
                            if (tid < stride && (tid + stride) < activeBuffSize ) {
                                filterShared[tid] = __hadd(filterShared[tid], filterShared[tid + stride]);
                            }
                            activeBuffSize = stride;
                            __syncthreads();
                        }
                        if (tid == 0){// write result back to tmp_val
                            filterShared[0] = __hadd(filterShared[0], filterShared[1]);
                            tmp_val = __hadd(tmp_val, filterShared[0]);
                        }
                    }
                }
                /**Conv End**/
                if (tid == 0){// write result back to outputbuff
                    outputShared[(j+i*outputDim)*outputAmount + ok] = __hadd(tmp_val, __float2half(b));
                }
            }
        }
    }
    //write back global mem
    for(int loadChunk=0; loadChunk < outputArea; loadChunk += filterBufferSize){
        if(loadChunk + tid < outputArea){
            curOutput[loadChunk + tid] = __half2float(outputShared[loadChunk + tid]);
        }
    }
}
__global__ void g_ConvCFM_feedforward_row_FR(
    float*  inputs,
    float* ws,
    float* bs,
    float*  outputs,
    int inputDim,
    int kernelSize,
    int padding,
    int outputDim,
    int inputAmount,
    int outputAmount)
{
    int sp = blockIdx.x;
    int inputSize2  = inputDim* inputDim;
    int outputSize2 = outputDim * outputDim;
    int kernelSize2 = kernelSize * kernelSize;
    int inputArea  = inputSize2 * inputAmount;
    int outputArea = outputSize2* outputAmount;
    int k_idx = threadIdx.z + threadIdx.y*kernelSize;
    int tid = threadIdx.z + threadIdx.y*kernelSize + threadIdx.x*kernelSize2;
    //declare shared mem
    extern __shared__ half bufferShared[];
    //point to diff addr
    int filterBufferSize = blockDim.x  * blockDim.y  * blockDim.z;
    int suInput = kernelSize * inputDim * inputAmount;
    int suOutput = outputDim * outputDim * outputAmount;
    half *inputShared  = bufferShared;
    half *outputShared = bufferShared + suInput;
    half *filterShared = bufferShared + suInput + suOutput;

    float t_w_array[5];
    float* curInput = inputs + sp * inputArea;
    float* curOutput = outputs + sp * outputArea;
    
    for(int ok=0; ok<outputAmount; ok++){//ok: filter
        /**Load Filter Start**/
        float b = bs[ok];
        float* w = ws + ok * kernelSize2 * inputAmount;

        for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
            int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
            if(t_ch_idx < inputAmount){
                t_w_array[ch_chunk_idx] = w[k_idx*inputAmount + t_ch_idx];
            }
        }
        /**Load Filter End**/
        for(int i=0; i<inputDim-kernelSize+1; i++){//i: input column
            /**Buffer Loading Start**/
            int inputBuffOffset =   i    % kernelSize;
            int inputLoadOffset =  (i-1) % kernelSize;
            if(i==0){
                //naive version
                for(int load_idx=0; load_idx < inputDim; load_idx+=kernelSize){
                    int buff_x = load_idx + threadIdx.z;
                    int buff_y = threadIdx.y;
                    int buff_idx  = (buff_x+buff_y*inputDim)*inputAmount + threadIdx.x;
                    int input_idx = (buff_x + (i+buff_y)*inputDim)*inputAmount + threadIdx.x;
                    if(buff_x < inputDim){
                        inputShared[buff_idx] = __float2half(curInput[input_idx]);
                    }
                }
            }
            else{//just need to load last row, use as much threads as possible
                for(int loadChunk=0; loadChunk < inputDim; loadChunk += kernelSize2){
                    int input_x = loadChunk + k_idx;
                    int input_y = i+kernelSize-1;
                    int input_idx = (input_x + input_y*inputDim)*inputAmount + threadIdx.x;

                    int inputLoadOffset_idx = (input_x + ((input_y+inputLoadOffset)% kernelSize)*inputDim)*inputAmount + threadIdx.x;
                    if(input_x < inputDim){
                        inputShared[inputLoadOffset_idx] = __float2half(curInput[input_idx]);
                    }
                }
            }
            /**Buffer Loading End**/
            for(int j=0; j<inputDim-kernelSize+1; j++){//j:row
                int xx = j+threadIdx.z;
                int yy = i+threadIdx.y;
                int inputBuffOffset_Idx = (xx + ((yy+inputBuffOffset) % kernelSize)*inputDim)*inputAmount + threadIdx.x;

                /**Conv Start**/
                half tmp_val = 0;
                for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
                    int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
                    if(t_ch_idx < inputAmount){

                        filterShared[tid] = __hmul(inputShared[inputBuffOffset_Idx], __float2half(t_w_array[ch_chunk_idx]));
                        __syncthreads();
                        //reduction
                        int activeBuffSize = filterBufferSize;
                        for (int stride = ceil(filterBufferSize / 2.0); stride > 1; stride = ceil(stride/2.0)) {
                            if (tid < stride && (tid + stride) < activeBuffSize ) {
                                filterShared[tid] = __hadd(filterShared[tid], filterShared[tid + stride]);
                            }
                            activeBuffSize = stride;
                            __syncthreads();
                        }
                        if (tid == 0){// write result back to tmp_val
                            filterShared[0] = __hadd(filterShared[0], filterShared[1]);
                            tmp_val = __hadd(tmp_val, filterShared[0]);
                        }
                    }
                }
                /**Conv End**/
                if (tid == 0){// write result back to outputbuff
                    outputShared[(j+i*outputDim)*outputAmount + ok] = __hadd(tmp_val, __float2half(b));
                }
            }
        }
    }
    //write back global mem
    for(int loadChunk=0; loadChunk < outputArea; loadChunk += filterBufferSize){
        if(loadChunk + tid < outputArea){
            curOutput[loadChunk + tid] = __half2float(outputShared[loadChunk + tid]);
        }
    }
}
__global__ void g_ConvCFM_feedforward_mini_FR(
    float*  inputs,
    float* ws,
    float* bs,
    float*  outputs,
    int inputDim,
    int kernelSize,
    int padding,
    int outputDim,
    int inputAmount,
    int outputAmount)
{
    int sp = blockIdx.x;
    int inputSize2  = inputDim* inputDim;
    int outputSize2 = outputDim * outputDim;
    int kernelSize2 = kernelSize * kernelSize;
    int inputArea  = inputSize2 * inputAmount;
    int outputArea = outputSize2* outputAmount;
    int k_idx = threadIdx.z + threadIdx.y*kernelSize;
    int tid = threadIdx.z + threadIdx.y*kernelSize + threadIdx.x*kernelSize2;    
    //declare shared mem
    extern __shared__ half bufferShared[];
    //point to diff addr
    int filterBufferSize = blockDim.x  * blockDim.y  * blockDim.z;
    int suInput = filterBufferSize;
    int suOutput = outputDim * outputDim * outputAmount;
    half *inputShared  = bufferShared;
    half *outputShared = bufferShared + suInput;
    half *filterShared = bufferShared + suInput + suOutput;

    float t_w_array[5];
    float* curInput = inputs + sp * inputArea;
    float* curOutput = outputs + sp * outputArea;

    for(int ok=0; ok<outputAmount; ok++){//ok: filter
        /**Load Filter Start**/
            float b = bs[ok];
            float* w = ws + ok * kernelSize2 * inputAmount;
            for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
                int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
                if(t_ch_idx < inputAmount){
                    t_w_array[ch_chunk_idx] = w[k_idx*inputAmount + t_ch_idx];
                }
            }
        /**Load Filter End**/
        for(int i=0; i<inputDim-kernelSize+1; i++){//i: input column
            for(int j=0; j<inputDim-kernelSize+1; j++){//j:row
                /**Buffer Loading Start**/
                    //(xx, yy): threadId map to input coordinate
                    int xx = j + threadIdx.z;
                    int yy = i + threadIdx.y;

                    int inputBuffOffset =  (j    % kernelSize);
                    int inputLoadOffset =  (j-1) % kernelSize;
                    int inputLoadOffset_idx = ((inputLoadOffset + k_idx)%kernelSize)*inputAmount+ threadIdx.x;
                    int inputBuffOffset_Idx = ((inputBuffOffset + k_idx)%kernelSize)*inputAmount+ threadIdx.x;

                    int buff_idx  = k_idx*inputAmount + threadIdx.x;
                    int input_idx = (xx + yy*inputDim)*inputAmount + threadIdx.x;
                    if(j==0){
                        inputShared[buff_idx] = __float2half(curInput[input_idx]);
                    }
                    else{
                        //always right most column threads need to load
                        if( threadIdx.z == (kernelSize-1) ){
                            inputShared[inputLoadOffset_idx] = __float2half(curInput[input_idx]);
                        }
                    }
                /**Buffer Loading End**/
                /**Conv Start**/
                    half tmp_val = 0;
                    for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
                        int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
                        if(t_ch_idx < inputAmount){
                            filterShared[tid] = __hmul(inputShared[inputBuffOffset_Idx], __float2half(t_w_array[ch_chunk_idx]));
                            __syncthreads();
                            //reduction
                            int activeBuffSize = filterBufferSize;
                            for (int stride = ceil(filterBufferSize / 2.0); stride > 1; stride = ceil(stride/2.0)) {
                                if (tid < stride && (tid + stride) < activeBuffSize ) {
                                    filterShared[tid] = __hadd(filterShared[tid], filterShared[tid + stride]);
                                }
                                activeBuffSize = stride;
                                __syncthreads();
                            }
                            if (tid == 0){// write result back to tmp_val
                                filterShared[0] = __hadd(filterShared[0], filterShared[1]);
                                tmp_val = __hadd(tmp_val, filterShared[0]);
                            }
                        }
                    }
                /**Conv End**/
                //FR
                if (tid == 0){// write result back to outputbuff
                    outputShared[(j+i*outputDim)*outputAmount + ok] = __hadd(tmp_val, __float2half(b));
                }
                __syncthreads();
                
            }
        }
    }
    //write back global mem
    //FR
    for(int loadChunk=0; loadChunk < outputArea; loadChunk += filterBufferSize){
        if(loadChunk + tid < outputArea){
            curOutput[loadChunk + tid] = __half2float(outputShared[loadChunk + tid]);
        }
    }
    
}

__global__ void g_ConvCFM_feedforward_full_IR(
        float*  inputs,
        float* ws,
        float* bs,
        float*  outputs,
        int inputDim,
        int kernelSize,
        int padding,
        int outputDim,
        int inputAmount,
        int outputAmount)
{
    int sp = blockIdx.x;
    int outputSize2 = outputDim * outputDim;
    int inputSize2  = inputDim* inputDim;
    int kernelSize2 = kernelSize * kernelSize;
    int inputArea  = inputSize2 * inputAmount;
    int outputArea = outputSize2* outputAmount;
    int k_idx = threadIdx.z + threadIdx.y*kernelSize;
    int tid = threadIdx.z + threadIdx.y*kernelSize + threadIdx.x*kernelSize2;
    //declare shared mem
    extern __shared__ half bufferShared[];
    //point to diff addr
    int suInput = inputDim * inputDim * inputAmount;
    int suOutput = outputAmount;
    int filterBufferSize= blockDim.x  * blockDim.y  * blockDim.z;
    half *inputShared  = bufferShared;
    half *outputShared = bufferShared + suInput;
    half *filterShared = bufferShared + suInput + suOutput;

    float t_w_array[5];
    float* curInput = inputs + sp * inputArea;
    float* curOutput = outputs + sp * outputArea;
    
    
    /**Buffer Loading Start**/
    for(int loadChunk=0; loadChunk < inputArea; loadChunk+=filterBufferSize){
        if(loadChunk + tid < inputArea){
            inputShared[loadChunk + tid] = __float2half(curInput[loadChunk + tid]);
        }
    }
    /**Buffer Loading End**/
    for(int i=0; i<inputDim-kernelSize+1; i++){//i: column
        for(int j=0; j<inputDim-kernelSize+1; j++){//j:row
            int xx = j+threadIdx.z;
            int yy = i+threadIdx.y;
            int inputBuffOffset_Idx = (xx + yy*inputDim)*inputAmount + threadIdx.x;
            for(int ok=0; ok<outputAmount; ok++){//ok: filter
                /**Load Filter Start**/
                float b = bs[ok];
                float* w = ws + ok * kernelSize2 * inputAmount;
                for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
                    int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
                    if(t_ch_idx < inputAmount){
                        t_w_array[ch_chunk_idx] = w[k_idx*inputAmount + t_ch_idx];
                    }
                }
                /**Load Filter End**/
                /**Conv Start**/
                half tmp_val = 0;
                for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
                    int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
                    if(t_ch_idx < inputAmount){
                        
                        filterShared[tid] = __hmul(inputShared[inputBuffOffset_Idx], __float2half(t_w_array[ch_chunk_idx]));
                        __syncthreads();
                        //reduction
                        int activeBuffSize = filterBufferSize;
                        for (int stride = ceil(filterBufferSize / 2.0); stride > 1; stride = ceil(stride/2.0)) {
                            if (tid < stride && (tid + stride) < activeBuffSize ) {
                                filterShared[tid] = __hadd(filterShared[tid], filterShared[tid + stride]);
                            }
                            activeBuffSize = stride;
                            __syncthreads();
                        }
                        if (tid == 0){// write result back to tmp_val
                            filterShared[0] = __hadd(filterShared[0], filterShared[1]);
                            tmp_val = __hadd(tmp_val, filterShared[0]);
                        }
                    }
                }
                /**Conv End**/
                if (tid == 0){// write result back to outputbuff
                    outputShared[ok] = __hadd(tmp_val, __float2half(b));
                }

            }
            //save output buffer to global Mem
            for(int loadChunk=0; loadChunk<outputAmount; loadChunk+=filterBufferSize){
                if(loadChunk + tid < outputAmount){
                    curOutput[outputAmount*(i*outputDim + j) + loadChunk + tid] = __half2float(outputShared[loadChunk + tid]);
                }
            }
        }
    }
    
}

__global__ void g_ConvCFM_feedforward_row_IR(
        float*  inputs,
        float* ws,
        float* bs,
        float*  outputs,
        int inputDim,
        int kernelSize,
        int padding,
        int outputDim,
        int inputAmount,
        int outputAmount)
{
    int sp = blockIdx.x;
    int outputSize2 = outputDim * outputDim;
    int inputSize2  = inputDim* inputDim;
    int kernelSize2 = kernelSize * kernelSize;
    int inputArea  = inputSize2 * inputAmount;
    int outputArea = outputSize2* outputAmount;
    int k_idx = threadIdx.z + threadIdx.y*kernelSize;
    int tid = threadIdx.z + threadIdx.y*kernelSize + threadIdx.x*kernelSize2;
    //declare shared mem
    extern __shared__ half bufferShared[];
    //point to diff addr
    int suInput = kernelSize * inputDim * inputAmount;
    int suOutput = outputAmount;
    int filterBufferSize= blockDim.x  * blockDim.y  * blockDim.z;
    half *inputShared  = bufferShared;
    half *outputShared = bufferShared + suInput;
    half *filterShared = bufferShared + suInput + suOutput;
    
    float t_w_array[5];
    float* curInput = inputs + sp * inputArea;
    float* curOutput = outputs + sp * outputArea;

    for(int i=0; i<inputDim-kernelSize+1; i++){//i: input column
        /**Buffer Loading Start**/
        int inputBuffOffset =   i    % kernelSize;
        int inputLoadOffset =  (i-1) % kernelSize;

        if(i==0){
            //naive version
            for(int load_idx=0; load_idx < inputDim; load_idx+=kernelSize){
                int buff_x = load_idx + threadIdx.z;
                int buff_y = threadIdx.y;
                int buff_idx  = (buff_x+buff_y*inputDim)*inputAmount + threadIdx.x;
                int input_idx = (buff_x + (i+buff_y)*inputDim)*inputAmount + threadIdx.x;
                if(buff_x < inputDim){
                    inputShared[buff_idx] = __float2half(curInput[input_idx]);
                }
            }
        }
        else{//just need to load last row, use as much threads as possible
            for(int loadChunk=0; loadChunk < inputDim; loadChunk += kernelSize2){
                int input_x = loadChunk + k_idx;
                int input_y = i+kernelSize-1;
                int input_idx = (input_x + input_y*inputDim)*inputAmount + threadIdx.x;

                int inputLoadOffset_idx = (input_x + ((input_y+inputLoadOffset)% kernelSize)*inputDim)*inputAmount + threadIdx.x;
                if(input_x < inputDim){
                    inputShared[inputLoadOffset_idx] = __float2half(curInput[input_idx]);
                }
            }
        }
        /**Buffer Loading End**/
        for(int j=0; j<inputDim-kernelSize+1; j++){//j:row
            int xx = j+threadIdx.z;
            int yy = i+threadIdx.y;
            int inputBuffOffset_Idx = (xx + ((yy+inputBuffOffset) % kernelSize)*inputDim)*inputAmount + threadIdx.x;
            for(int ok=0; ok<outputAmount; ok++){//ok: filter
                /**Load Filter Start**/
                float b = bs[ok];
                float* w = ws + ok * kernelSize2 * inputAmount;
                for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
                    int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
                    if(t_ch_idx < inputAmount){
                        t_w_array[ch_chunk_idx] = w[k_idx*inputAmount + t_ch_idx];
                    }
                }
                /**Load Filter End**/
                /**Conv Start**/
                half tmp_val = 0;
                for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
                    int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
                    if(t_ch_idx < inputAmount){
                        filterShared[tid] = __hmul(inputShared[inputBuffOffset_Idx], __float2half(t_w_array[ch_chunk_idx]));
                        __syncthreads();
                        //reduction
                        int activeBuffSize = filterBufferSize;
                        for (int stride = ceil(filterBufferSize / 2.0); stride > 1; stride = ceil(stride/2.0)) {
                            if (tid < stride && (tid + stride) < activeBuffSize ) {
                                filterShared[tid] = __hadd(filterShared[tid], filterShared[tid + stride]);
                            }
                            activeBuffSize = stride;
                            __syncthreads();
                        }
                        if (tid == 0){// write result back to tmp_val
                            filterShared[0] = __hadd(filterShared[0], filterShared[1]);
                            tmp_val = __hadd(tmp_val, filterShared[0]);
                        }
                    }
                }
                /**Conv End**/
                if (tid == 0){// write result back to outputbuff
                    outputShared[ok] = __hadd(tmp_val, __float2half(b));
                }
            }
            //save output buffer to global Mem
            for(int loadChunk=0; loadChunk<outputAmount; loadChunk+=filterBufferSize){
                if(loadChunk + tid < outputAmount){
                    curOutput[outputAmount*(i*inputDim + j) + loadChunk + tid] = __half2float(outputShared[loadChunk + tid]);
                }
            }
        }
    }
    
}

__global__ void g_ConvCFM_feedforward_mini_IR(
        float*  inputs,
        float* ws,
        float* bs,
        float*  outputs,
        int inputDim,
        int kernelSize,
        int padding,
        int outputDim,
        int inputAmount,
        int outputAmount)
{
    int sp = blockIdx.x;
    int outputSize2 = outputDim * outputDim;
    int inputSize2  = inputDim* inputDim;
    int kernelSize2 = kernelSize * kernelSize;
    int inputArea  = inputSize2 * inputAmount;
    int outputArea = outputSize2* outputAmount;
    int tid = threadIdx.z + threadIdx.y*kernelSize + threadIdx.x*kernelSize2;
    int k_idx = threadIdx.z + threadIdx.y*kernelSize;
    //declare shared mem
    extern __shared__ half bufferShared[];
    //point to diff addr
    int suInput = kernelSize * kernelSize * inputAmount;
    int suOutput = outputAmount;
    int filterBufferSize= blockDim.x  * blockDim.y  * blockDim.z;
    half *inputShared  = bufferShared;
    half *outputShared = bufferShared + suInput;
    half *filterShared = bufferShared + suInput + suOutput;

    float t_w_array[5];
    float* curInput = inputs + sp * inputArea;
    float* curOutput = outputs + sp * outputArea;

    for(int i=0; i<inputDim-kernelSize+1; i++){//i: input column
        //j:row
        for(int j=0; j<inputDim-kernelSize+1; j++){
            /**Buffer Loading**/
            //(xx, yy): threadId map to input coordinate
            int xx = j + threadIdx.z;
            int yy = i + threadIdx.y;

            int inputBuffOffset =  (j    % kernelSize);
            int inputLoadOffset =  (j-1) % kernelSize;
            int inputLoadOffset_idx = ((inputLoadOffset + k_idx)%kernelSize)*inputAmount+ threadIdx.x;
            int inputBuffOffset_Idx = ((inputBuffOffset + k_idx)%kernelSize)*inputAmount+ threadIdx.x;

            int buff_idx  = k_idx*inputAmount + threadIdx.x;
            int input_idx = (xx + yy*inputDim)*inputAmount + threadIdx.x;
            if(j==0){
                inputShared[buff_idx] = __float2half(curInput[input_idx]);
            }
            else{
                //always right most column threads need to load
                if( threadIdx.z == (kernelSize-1) ){
                    inputShared[inputLoadOffset_idx] = __float2half(curInput[input_idx]);
                }
            }
            /**Buffer Loading End**/
            
            for(int ok=0; ok<outputAmount; ok++){//ok: filter
                /**Load Filter Start**/
                float b = bs[ok];
                float* w = ws + ok * kernelSize2 * inputAmount;
                for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
                    int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
                    if(t_ch_idx < inputAmount){
                        t_w_array[ch_chunk_idx] = w[k_idx*inputAmount + t_ch_idx];
                    }
                }
                /**Load Filter End**/
                /**Conv Start**/
                half tmp_val = 0;
                for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
                    int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
                    if(t_ch_idx < inputAmount){
                        
                        filterShared[tid] = __hmul(inputShared[inputBuffOffset_Idx], __float2half(t_w_array[ch_chunk_idx]));
                        __syncthreads();
                        //reduction
                        int activeBuffSize = filterBufferSize;
                        for (int stride = ceil(filterBufferSize / 2.0); stride > 1; stride = ceil(stride/2.0)) {
                            if (tid < stride && (tid + stride) < activeBuffSize ) {
                                filterShared[tid] = __hadd(filterShared[tid], filterShared[tid + stride]);
                            }
                            activeBuffSize = stride;
                            __syncthreads();
                        }
                        if (tid == 0){// write result back to tmp_val
                            filterShared[0] = __hadd(filterShared[0], filterShared[1]);
                            tmp_val = __hadd(tmp_val, filterShared[0]);
                        }
                    }
                }
                /**Conv End**/
                if (tid == 0){// write result back to outputbuff
                    outputShared[ok] = __hadd(tmp_val, __float2half(b));
                }
            }
            //save output buffer to global Mem
            for(int loadChunk=0; loadChunk<outputAmount; loadChunk+=filterBufferSize){
                if(loadChunk + tid < outputAmount){
                    curOutput[outputAmount*(i*inputDim + j) + loadChunk + tid] = __half2float(outputShared[loadChunk + tid]);
                }
            }
        }
    }
    
}
__global__ void g_ConvCFM_feedforward_full_FRIT(
        float*  inputs,
        float* ws,
        float* bs,
        float*  outputs,
        int inputDim,
        int kernelSize,
        int padding,
        int outputDim,
        int inputAmount,
        int outputAmount)
{
    int sp = blockIdx.x;
    int outputSize2 = outputDim * outputDim;
    int inputSize2  = inputDim* inputDim;
    int kernelSize2 = kernelSize * kernelSize;
    int inputArea  = inputSize2 * inputAmount;
    int outputArea = outputSize2* outputAmount;
    int tid = threadIdx.z + threadIdx.y*kernelSize + threadIdx.x*kernelSize2;
    int k_idx = threadIdx.z + threadIdx.y*kernelSize;
    //declare shared mem
    extern __shared__ half bufferShared[];
    //point to diff addr
    int suInput = inputDim * inputDim * inputAmount;
    int suOutput = outputDim * outputAmount;
    int filterBufferSize= blockDim.x  * blockDim.y  * blockDim.z;
    half *inputShared  = bufferShared;
    half *outputShared = bufferShared + suInput;
    half *filterShared = bufferShared + suInput + suOutput;

    float t_w_array[5];
    float* curInput = inputs + sp * inputArea;
    float* curOutput = outputs + sp * outputArea;
    /**Buffer Loading Start**/
    for(int loadChunk=0; loadChunk < inputArea; loadChunk+=filterBufferSize){
        if(loadChunk + tid < inputArea){
            inputShared[loadChunk + tid] = __float2half(curInput[loadChunk + tid]);
        }
    }
    /**Buffer Loading End**/
    for(int i=0; i<inputDim-kernelSize+1; i++){//i: input column
        
        for(int ok=0; ok<outputAmount; ok++){//ok: filter
            /**Load Filter Start**/
            float b = bs[ok];
            float* w = ws + ok * kernelSize2 * inputAmount;
            for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
                int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
                if(t_ch_idx < inputAmount){
                    t_w_array[ch_chunk_idx] = w[k_idx*inputAmount + t_ch_idx];
                }
            }
            /**Load Filter End**/
            for(int j=0; j<inputDim-kernelSize+1; j++){//j:row
                /**Conv Start**/
                int xx = j+threadIdx.z;
                int yy = i+threadIdx.y;
                int inputBuffOffset_Idx = (xx + yy*inputDim)*inputAmount + threadIdx.x;
                half tmp_val = 0;
                for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
                    int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
                    if(t_ch_idx < inputAmount){
                        
                        filterShared[tid] = __hmul(inputShared[inputBuffOffset_Idx], __float2half(t_w_array[ch_chunk_idx]));
                        __syncthreads();
                        //reduction
                        int activeBuffSize = filterBufferSize;
                        for (int stride = ceil(filterBufferSize / 2.0); stride > 1; stride = ceil(stride/2.0)) {
                            if (tid < stride && (tid + stride) < activeBuffSize ) {
                                filterShared[tid] = __hadd(filterShared[tid], filterShared[tid + stride]);
                            }
                            activeBuffSize = stride;
                            __syncthreads();
                        }
                        
                        if (tid == 0){// write result back to tmp_val
                            filterShared[0] = __hadd(filterShared[0], filterShared[1]);
                            tmp_val = __hadd(tmp_val, filterShared[0]);
                        }
                    }
                }
                /**Conv End**/
                
                if (tid == 0){// write result back to outputbuff
                    outputShared[j*outputAmount + ok] = __hadd(tmp_val, __float2half(b));
                }
            }
        }
        //save output buffer to global Mem
        for(int loadChunk=0; loadChunk< outputDim * outputAmount; loadChunk+=filterBufferSize){
            if(loadChunk + tid < outputDim * outputAmount){
                curOutput[outputAmount*(i*outputDim) + loadChunk + tid] = __half2float(outputShared[loadChunk + tid]);
            }
        }
    }
}
__global__ void g_ConvCFM_feedforward_row_FRIT(
        float*  inputs,
        float* ws,
        float* bs,
        float*  outputs,
        int inputDim,
        int kernelSize,
        int padding,
        int outputDim,
        int inputAmount,
        int outputAmount)
{
    int sp = blockIdx.x;
    int outputSize2 = outputDim * outputDim;
    int inputSize2  = inputDim* inputDim;
    int kernelSize2 = kernelSize * kernelSize;
    int inputArea  = inputSize2 * inputAmount;
    int outputArea = outputSize2* outputAmount;
    int tid = threadIdx.z + threadIdx.y*kernelSize + threadIdx.x*kernelSize2;
    int k_idx = threadIdx.z + threadIdx.y*kernelSize;
    //declare shared mem
    extern __shared__ half bufferShared[];
    //point to diff addr
    int suInput = kernelSize * inputDim * inputAmount;
    int suOutput = outputDim * outputAmount;
    int filterBufferSize= blockDim.x  * blockDim.y  * blockDim.z;
    half *inputShared  = bufferShared;
    half *outputShared = bufferShared + suInput;
    half *filterShared = bufferShared + suInput + suOutput;

    float t_w_array[5];
    float* curInput = inputs + sp * inputArea;
    float* curOutput = outputs + sp * outputArea;
    
    for(int i=0; i<inputDim-kernelSize+1; i++){//i: input column
        /**Buffer Loading Start**/
        int inputBuffOffset =   i    % kernelSize;
        int inputLoadOffset =  (i-1) % kernelSize;
        if(i==0){
            //naive version
            for(int load_idx=0; load_idx < inputDim; load_idx+=kernelSize){
                int buff_x = load_idx + threadIdx.z;
                int buff_y = threadIdx.y;
                int buff_idx  = (buff_x+buff_y*inputDim)*inputAmount + threadIdx.x;
                int input_idx = (buff_x + (i+buff_y)*inputDim)*inputAmount + threadIdx.x;
                if(buff_x < inputDim){
                    inputShared[buff_idx] = __float2half(curInput[input_idx]);
                }
            }
        }
        else{//just need to load last row, use as much threads as possible
            for(int loadChunk=0; loadChunk < inputDim; loadChunk += kernelSize2){
                int input_x = loadChunk + k_idx;
                int input_y = i+kernelSize-1;
                int input_idx = (input_x + input_y*inputDim)*inputAmount + threadIdx.x;

                int inputLoadOffset_idx = (input_x + ((input_y+inputLoadOffset)% kernelSize)*inputDim)*inputAmount + threadIdx.x;
                if(input_x < inputDim){
                    inputShared[inputLoadOffset_idx] = __float2half(curInput[input_idx]);
                }
            }
        }
        /**Buffer Loading End**/
        
        for(int ok=0; ok<outputAmount; ok++){//ok: filter
            /**Load Filter Start**/
            float b = bs[ok];
            float* w = ws + ok * kernelSize2 * inputAmount;
            for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
                int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
                if(t_ch_idx < inputAmount){
                    t_w_array[ch_chunk_idx] = w[k_idx*inputAmount + t_ch_idx];
                }
            }
            /**Load Filter End**/
            for(int j=0; j<inputDim-kernelSize+1; j++){//j:row
                /**Conv Start**/
                int xx = j+threadIdx.z;
                int yy = i+threadIdx.y;
                int inputBuffOffset_Idx = (xx + ((yy+inputBuffOffset) % kernelSize)*inputDim)*inputAmount + threadIdx.x;
                half tmp_val = 0;
                for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
                    int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
                    if(t_ch_idx < inputAmount){
                        
                        filterShared[tid] = __hmul(inputShared[inputBuffOffset_Idx], __float2half(t_w_array[ch_chunk_idx]));
                        __syncthreads();
                        //reduction
                        int activeBuffSize = filterBufferSize;
                        for (int stride = ceil(filterBufferSize / 2.0); stride > 1; stride = ceil(stride/2.0)) {
                            if (tid < stride && (tid + stride) < activeBuffSize ) {
                                filterShared[tid] = __hadd(filterShared[tid], filterShared[tid + stride]);
                            }
                            activeBuffSize = stride;
                            __syncthreads();
                        }
                        
                        if (tid == 0){// write result back to tmp_val
                            filterShared[0] = __hadd(filterShared[0], filterShared[1]);
                            tmp_val = __hadd(tmp_val, filterShared[0]);
                        }
                    }
                }
                /**Conv End**/
                
                if (tid == 0){// write result back to outputbuff
                    outputShared[j*outputAmount + ok] = __hadd(tmp_val, __float2half(b));
                }
            }
        }
        //save output buffer to global Mem
        for(int loadChunk=0; loadChunk< outputDim * outputAmount; loadChunk+=filterBufferSize){
            if(loadChunk + tid < outputDim * outputAmount){
                curOutput[outputAmount*(i*outputDim) + loadChunk + tid] = __half2float(outputShared[loadChunk + tid]);
            }
        }
    }
}
__global__ void g_ConvCFM_feedforward_mini_FRIT(
        float*  inputs,
        float* ws,
        float* bs,
        float*  outputs,
        int inputDim,
        int kernelSize,
        int padding,
        int outputDim,
        int inputAmount,
        int outputAmount)
{
    int sp = blockIdx.x;
    int outputSize2 = outputDim * outputDim;
    int inputSize2  = inputDim* inputDim;
    int kernelSize2 = kernelSize * kernelSize;
    int inputArea  = inputSize2 * inputAmount;
    int outputArea = outputSize2* outputAmount;
    int tid = threadIdx.z + threadIdx.y*kernelSize + threadIdx.x*kernelSize2;
    int k_idx = threadIdx.z + threadIdx.y*kernelSize;
    //declare shared mem
    extern __shared__ half bufferShared[];
    //point to diff addr
    int suInput = kernelSize * kernelSize * inputAmount;
    int suOutput = outputDim * outputAmount;
    int filterBufferSize= blockDim.x  * blockDim.y  * blockDim.z;
    half *inputShared  = bufferShared;
    half *outputShared = bufferShared + suInput;
    half *filterShared = bufferShared + suInput + suOutput;

    float t_w_array[5];
    float* curInput = inputs + sp * inputArea;
    float* curOutput = outputs + sp * outputArea;
    
    for(int i=0; i<inputDim-kernelSize+1; i++){//i: input column
        for(int ok=0; ok<outputAmount; ok++){//ok: filter
            /**Load Filter Start**/
            float b = bs[ok];
            float* w = ws + ok * kernelSize2 * inputAmount;
            for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
                int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
                if(t_ch_idx < inputAmount){
                    t_w_array[ch_chunk_idx] = w[k_idx*inputAmount + t_ch_idx];
                }
            }
            /**Load Filter End**/
            for(int j=0; j<inputDim-kernelSize+1; j++){//j:row
                /**Buffer Loading Start**/
                //(xx, yy): threadId map to input coordinate
                int xx = j + threadIdx.z;
                int yy = i + threadIdx.y;
                int k_idx = threadIdx.z + threadIdx.y*kernelSize;
                
                int inputBuffOffset =  (j    % kernelSize);
                int inputLoadOffset =  (j-1) % kernelSize;
                int inputLoadOffset_idx = ((inputLoadOffset + k_idx)%kernelSize)*inputAmount+ threadIdx.x;
                int inputBuffOffset_Idx = ((inputBuffOffset + k_idx)%kernelSize)*inputAmount+ threadIdx.x;

                int buff_idx  = k_idx*inputAmount + threadIdx.x;
                int input_idx = (xx + yy*inputDim)*inputAmount + threadIdx.x;
                
                if(j==0){
                    inputShared[buff_idx] = __float2half(curInput[input_idx]);
                }
                else{
                    //always right most column threads need to load
                    if( threadIdx.z == (kernelSize-1) ){
                        inputShared[inputLoadOffset_idx] = __float2half(curInput[input_idx]);
                    }
                }
                /**Buffer Loading End**/
                /**Conv Start**/
                half tmp_val = 0;
                for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
                    int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
                    if(t_ch_idx < inputAmount){
                        
                        filterShared[tid] = __hmul(inputShared[inputBuffOffset_Idx], __float2half(t_w_array[ch_chunk_idx]));
                        __syncthreads();
                        //reduction
                        int activeBuffSize = filterBufferSize;
                        for (int stride = ceil(filterBufferSize / 2.0); stride > 1; stride = ceil(stride/2.0)) {
                            if (tid < stride && (tid + stride) < activeBuffSize ) {
                                filterShared[tid] = __hadd(filterShared[tid], filterShared[tid + stride]);
                            }
                            activeBuffSize = stride;
                            __syncthreads();
                        }
                        if (tid == 0){// write result back to tmp_val
                            filterShared[0] = __hadd(filterShared[0], filterShared[1]);
                            tmp_val = __hadd(tmp_val, filterShared[0]);
                        }
                    }
                }
                /**Conv End**/
                
                if (tid == 0){// write result back to outputbuff
                    outputShared[j*outputAmount + ok] = __hadd(tmp_val, __float2half(b));
                }
            }
        }
        //save output buffer to global Mem
        for(int loadChunk=0; loadChunk< outputDim * outputAmount; loadChunk+=filterBufferSize){
            if(loadChunk + tid < outputDim * outputAmount){
                curOutput[outputAmount*(i*outputDim) + loadChunk + tid] = __half2float(outputShared[loadChunk + tid]);
            }
        }
    }
}
__global__ void g_ConvCFM_feedforward_fool(
        float*  inputs,
        float* ws,
        float* bs,
        float*  outputs,
        int inputDim,
        int kernelSize,
        int padding,
        int outputDim,
        int inputAmount,
        int outputAmount)
{
    int sp = blockIdx.x;
    int outputSize2 = outputDim * outputDim;
    int inputSize2  = inputDim* inputDim;
    int kernelSize2 = kernelSize * kernelSize;
    int cfm = inputAmount;
    int inputArea  = inputSize2 * inputAmount;
    int outputArea = outputSize2* outputAmount;
    /*convolution*/
    //int tid = threadIdx.x + threadIdx.y*kernelSize + threadIdx.z*kernelSize2;
    
    
    for(int ok=0; ok<outputAmount; ok++){
        float* curOutput = outputs + sp * outputArea;
        for(int tidx = 0; tidx < outputSize2; tidx += blockDim.x)
        {
            int idx = tidx + threadIdx.x;
            if(idx < outputSize2)
            {
                int x = idx / outputDim;
                int y = idx % outputDim;

                half val = 0.0;

                for(int c = 0; c < cfm; c++){
                    float* curInput = inputs + sp * inputArea;
                    float* w = ws + ok * kernelSize2 * inputAmount;
                    
                    /*put curInput and w into shared memory*/
                    for(int i = 0; i < kernelSize; i++){
                        int xx = x + i ;
                        for(int j = 0; j < kernelSize; j++){
                            int yy = y + j;
                            if(xx >= 0 && xx < inputDim && yy >= 0 && yy < inputDim)
                                val += __hmul(__float2half(curInput[cfm*(xx*inputDim + yy)+c]), __float2half(w[cfm*(i * kernelSize + j) + c]));
                        }
                    }
                }
                //HWC
                curOutput[outputAmount*(x*outputDim + y) + ok] = (__half2float(val) + bs[ok]);
            }
        }
    }
}
__global__ void g_ConvCFM_feedforward_IR(
        float*  inputs,
        float* ws,
        float* bs,
        float*  outputs,
        int inputDim,
        int kernelSize,
        int padding,
        int outputDim,
        int inputAmount,
        int outputAmount)
{
    int sp = blockIdx.x;
    int outputSize2 = outputDim * outputDim;
    int inputSize2  = inputDim* inputDim;
    int kernelSize2 = kernelSize * kernelSize;
    int inputArea  = inputSize2 * inputAmount;
    int outputArea = outputSize2* outputAmount;
    int tid = threadIdx.z + threadIdx.y*kernelSize + threadIdx.x*kernelSize2;
    int k_idx = threadIdx.z + threadIdx.y*kernelSize;
    //declare shared mem
    extern __shared__ half bufferShared[];
    //point to diff addr
    int filterBufferSize= blockDim.x  * blockDim.y  * blockDim.z;
    half *filterShared = bufferShared;

    float t_w_array[5];
    float* curInput = inputs + sp * inputArea;
    float* curOutput = outputs + sp * outputArea;

    for(int i=0; i<inputDim-kernelSize+1; i++){//i: input column
        //j:row
        for(int j=0; j<inputDim-kernelSize+1; j++){
            for(int ok=0; ok<outputAmount; ok++){//ok: filter
                float b = bs[ok];
                float* w = ws + ok * kernelSize2 * inputAmount;
                /**Conv Start**/
                half tmp_val = 0;
                for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
                    int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
                    if(t_ch_idx < inputAmount){
                        int xx = j+threadIdx.z;
                        int yy = i+threadIdx.y;
                        
                        filterShared[tid] = __hmul(curInput[(xx + yy*inputDim)*inputAmount + threadIdx.x], __float2half(w[k_idx*inputAmount + t_ch_idx]));
                        __syncthreads();
                        //reduction
                        int activeBuffSize = filterBufferSize;
                        for (int stride = ceil(filterBufferSize / 2.0); stride > 1; stride = ceil(stride/2.0)) {
                            if (tid < stride && (tid + stride) < activeBuffSize ) {
                                filterShared[tid] = __hadd(filterShared[tid], filterShared[tid + stride]);
                            }
                            activeBuffSize = stride;
                            __syncthreads();
                        }
                        if (tid == 0){// write result back to tmp_val
                            filterShared[0] = __hadd(filterShared[0], filterShared[1]);
                            tmp_val = __hadd(tmp_val, filterShared[0]);
                        }
                    }
                }
                /**Conv End**/
                //save output buffer to global Mem
                if (tid == 0){
                    curOutput[outputAmount*(i*inputDim + j) + ok] = __hadd(tmp_val, __float2half(b));
                }
            }
        }
    }
}
__global__ void g_ConvCFM_feedforward_FR(
        float*  inputs,
        float* ws,
        float* bs,
        float*  outputs,
        int inputDim,
        int kernelSize,
        int padding,
        int outputDim,
        int inputAmount,
        int outputAmount)
{
    int sp = blockIdx.x;
    int outputSize2 = outputDim * outputDim;
    int inputSize2  = inputDim* inputDim;
    int kernelSize2 = kernelSize * kernelSize;
    int inputArea  = inputSize2 * inputAmount;
    int outputArea = outputSize2* outputAmount;
    int tid = threadIdx.z + threadIdx.y*kernelSize + threadIdx.x*kernelSize2;
    int k_idx = threadIdx.z + threadIdx.y*kernelSize;
    //declare shared mem
    extern __shared__ half bufferShared[];
    //point to diff addr
    int filterBufferSize= blockDim.x  * blockDim.y  * blockDim.z;
    half *filterShared = bufferShared;

    float t_w_array[5];
    float* curInput = inputs + sp * inputArea;
    float* curOutput = outputs + sp * outputArea;
    for(int ok=0; ok<outputAmount; ok++){//ok: filter
        float b = bs[ok];
        float* w = ws + ok * kernelSize2 * inputAmount;
        for(int i=0; i<inputDim-kernelSize+1; i++){//i: input column
            //j:row
            for(int j=0; j<inputDim-kernelSize+1; j++){
            
                /**Conv Start**/
                half tmp_val = 0;
                for(int ch_chunk_idx = 0; ch_chunk_idx < 5; ch_chunk_idx++){
                    int t_ch_idx = ch_chunk_idx * blockDim.x + threadIdx.x;
                    if(t_ch_idx < inputAmount){
                        int xx = j+threadIdx.z;
                        int yy = i+threadIdx.y;
                        
                        filterShared[tid] = __hmul(curInput[(xx + yy*inputDim)*inputAmount + threadIdx.x], __float2half(w[k_idx*inputAmount + t_ch_idx]));
                        __syncthreads();
                        //reduction
                        int activeBuffSize = filterBufferSize;
                        for (int stride = ceil(filterBufferSize / 2.0); stride > 1; stride = ceil(stride/2.0)) {
                            if (tid < stride && (tid + stride) < activeBuffSize ) {
                                filterShared[tid] = __hadd(filterShared[tid], filterShared[tid + stride]);
                            }
                            activeBuffSize = stride;
                            __syncthreads();
                        }
                        if (tid == 0){// write result back to tmp_val
                            filterShared[0] = __hadd(filterShared[0], filterShared[1]);
                            tmp_val = __hadd(tmp_val, filterShared[0]);
                        }
                    }
                }
                /**Conv End**/
                //save output buffer to global Mem
                if (tid == 0){
                    curOutput[outputAmount*(i*inputDim + j) + ok] = __hadd(tmp_val, __float2half(b));
                }
            }
        }
    }
}