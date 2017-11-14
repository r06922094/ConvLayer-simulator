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

void Launch_naive_kernel(ConvLayer *conv);
void Launch_mini_IR_kernel(ConvLayer *conv);
void Launch_row_FR_kernel(ConvLayer *conv);

static void feedforward(LayerBase *thiz)
{
    ConvLayer *conv = (ConvLayer *) thiz;
    /* prepare output on GPU memory */
    tensor_create(&conv->lb->output, conv->lb->batchSize, conv->lb->outputDim, conv->lb->outputDim, conv->lb->outputChannel);
    conv->lb->output->mallocDev(conv->lb->output);

    /* TODO: feedforward implementation */
    //Launch_row_FR_kernel(conv);
    Launch_mini_IR_kernel(conv);
    //Launch_naive_kernel(conv);
    //printf("Doing feedforward\n");
}

void Launch_row_FR_kernel(ConvLayer *conv){
    /**check shared memory usage**/
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int sharedMemSize = (unsigned int)deviceProp.sharedMemPerBlock;
    int filterSize = sizeof(float) * conv->kernelDim * conv->kernelDim * conv->lb->inputChannel;
    int suInput    = sizeof(float) * conv->kernelDim * conv->lb->inputDim * conv->lb->inputChannel;
    int suOutput   = sizeof(float) * conv->lb->outputDim * conv->lb->outputDim * conv->kernelAmount;
    int suFilter   = filterSize;
    int sharedMemRequire = suInput + suOutput + suFilter;
    if(suInput + suOutput + suFilter > sharedMemSize){
        printf("row FR init error\n");
		exit(0);
    }
    printf("%d = %d + %d + %d / %d bytes needed\n",sharedMemRequire,suInput,suOutput,suFilter, sharedMemSize);
    dim3 block = dim3(conv->lb->batchSize);
    dim3 thread= dim3(conv->kernelDim , conv->kernelDim , conv->lb->inputChannel);
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

}


void Launch_mini_IR_kernel(ConvLayer *conv){
    /**check shared memory usage**/
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int sharedMemSize = (unsigned int)deviceProp.sharedMemPerBlock;
    int filterSize = sizeof(float) * conv->kernelDim * conv->kernelDim * conv->lb->inputChannel;
    int suInput    = filterSize;
    int suOutput   = sizeof(float) * conv->kernelAmount;
    int suFilter   = filterSize;
    int sharedMemRequire = suInput + suOutput + suFilter;
    if(sharedMemRequire > sharedMemSize){
        printf("mini IR init error\n");
		exit(0);
    }
    dim3 block = dim3(conv->lb->batchSize);
    dim3 thread= dim3(conv->kernelDim , conv->kernelDim , conv->lb->inputChannel);
    
    g_ConvCFM_feedforward_mini_IR<<<block, thread, sharedMemRequire, 0>>>(
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


void ConvLayer_init(ConvLayer **thiz, int batchSize, \
                    int inputDim, int inputChannel, \
                    int kernelDim, int kernelAmount, \
                    LayerBase *preLayer, LayerBase *nextLayer)
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
    base->outputDim = inputDim - kernelDim + 1;
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
                    tzr->set(tzr, i, j, k, w, Gx_array[j][k]);
                    /*
                    if(w==i){
                        tzr->set(tzr, i, j, k, w, 1);
                    }
                    else{
                        tzr->set(tzr, i, j, k, w, 0);
                    }
                    */
                    
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
    int filterBufferSize = blockDim.x  * blockDim.y  * blockDim.z;
    int suInput = kernelSize * inputDim * inputAmount;
    int suOutput = outputDim * outputDim * outputAmount;

    int tid = threadIdx.x + threadIdx.y*kernelSize + threadIdx.z*kernelSize2;
    //declare shared mem
    extern __shared__ float bufferShared[];
    //point to diff addr
    //output: save back to global sharedMemPerBlock
    //filter: save the partial sum to do reduction
    float *inputShared  = bufferShared;
    float *outputShared = bufferShared + suInput;
    float *filterShared = bufferShared + suInput + suOutput;

    float* curInput = inputs + sp * inputArea;
    float* curOutput = outputs + sp * outputArea;
    int k_idx = threadIdx.x + threadIdx.y*kernelSize;
    //loop filter
    for(int ok=0; ok<outputAmount; ok++){
        //load weight to thread register
        float* w = ws + ok * kernelSize2 * inputAmount;
        float b = bs[ok];
        float t_w = w[k_idx*inputAmount + threadIdx.z];
        //float t_w = ws[ok][wIdx];
        for(int i=0; i<inputDim-kernelSize+1; i++){
            /**Buffer Loading**/
            int inputBuffOffset =   i    % kernelSize;
            int inputLoadOffset =  (i-1) % kernelSize;

            //int inputLoadOffset_idx = ((inputLoadOffset + k_idx)%kernelSize)*inputAmount+ threadIdx.z;
            if(i==0){
                //naive version
                for(int load_idx=0; load_idx < inputDim; load_idx+=kernelSize){
                    int buff_x = load_idx + threadIdx.x;
                    int buff_y = threadIdx.y;
                    int buff_idx  = (buff_x+buff_y*inputDim)*inputAmount + threadIdx.z;
                    int input_idx = (buff_x + (i+buff_y)*inputDim)*inputAmount + threadIdx.z;
                    if(buff_x < inputDim){
                        inputShared[buff_idx] = curInput[input_idx];
                    }
                }
            }
            else{//just need to load last row, use as much threads as possible
                for(int loadChunk=0; loadChunk < inputDim; loadChunk += kernelSize2){
                    int input_x = loadChunk + k_idx;
                    int input_y = i+kernelSize-1;
                    int input_idx = (input_x + input_y*inputDim)*inputAmount + threadIdx.z;

                    int inputLoadOffset_idx = (input_x + ((input_y+inputLoadOffset)% kernelSize)*inputDim)*inputAmount + threadIdx.z;
                    if(input_x < inputDim){
                        inputShared[inputLoadOffset_idx] = curInput[input_idx];
                    }
                }
            }
            

            for(int j=0; j<inputDim-kernelSize+1; j++){
                int xx = j+threadIdx.x;
                int yy = i+threadIdx.y;
                int inputBufferIdx = (xx + ((yy+inputBuffOffset) % kernelSize)*inputDim)*inputAmount + threadIdx.z;

                /**start Conv**/
                filterShared[tid] = t_w * inputShared[inputBufferIdx];
                //wait for reduction
                __syncthreads();
                //reduction
                int activeBuffSize = filterBufferSize;
                for (int stride = ceil(filterBufferSize / 2.0); stride > 1; stride = ceil(stride/2.0)) {
                    if (tid < stride && (tid + stride) < activeBuffSize ) {
                        //filterShared[tid] = __hadd(filterShared[tid], filterShared[tid + stride]);
                        filterShared[tid] += filterShared[tid + stride];
                    }
                    activeBuffSize = stride;
                    __syncthreads();
                }
                // write result back to outputbuff
                if (tid == 0){
                    filterShared[0] += filterShared[1];
                    outputShared[(j+i*outputDim)*outputAmount + ok] = filterShared[tid] + b;
                }
            }
        }
    }
    //write back global mem (待補)
    
    for(int loadChunk=0; loadChunk < outputArea; loadChunk += filterBufferSize){
        if(loadChunk + tid < outputArea){
            curOutput[loadChunk + tid] = outputShared[loadChunk + tid];
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
    //declare shared mem
    extern __shared__ float bufferShared[];
    //point to diff addr
    //output: save back to global sharedMemPerBlock
    //filter: save the partial sum to do reduction
    int suInput = kernelSize * kernelSize * inputAmount;
    int suOutput = outputAmount;
    int filterBufferSize= blockDim.x  * blockDim.y  * blockDim.z;
    float *inputShared  = bufferShared;
    float *outputShared = bufferShared + suInput;
    float *filterShared = bufferShared + suInput + suOutput;
    
    float* curInput = inputs + sp * inputArea;
    float* curOutput = outputs + sp * outputArea;
    int tid = threadIdx.x + threadIdx.y*kernelSize + threadIdx.z*kernelSize2;
    // input loop
    //i: column
    for(int i=0; i<inputDim-kernelSize+1; i++){
        //j:row
        for(int j=0; j<inputDim-kernelSize+1; j++){
            /**Buffer Loading**/
            //(xx, yy): threadId map to input coordinate
            int xx = j + threadIdx.x;
            int yy = i + threadIdx.y;
            int k_idx = threadIdx.x + threadIdx.y*kernelSize;
            //move to addr of just one channel
            
            //start of the buffer:
            int inputBuffOffset =  (j    % kernelSize);
            int inputLoadOffset =  (j-1) % kernelSize;
            int inputLoadOffset_idx = ((inputLoadOffset + k_idx)%kernelSize)*inputAmount+ threadIdx.z;
            int inputBuffOffset_Idx = ((inputBuffOffset + k_idx)%kernelSize)*inputAmount+ threadIdx.z;

            int buff_idx  = k_idx*inputAmount + threadIdx.z;
            int input_idx = (xx + yy*inputDim)*inputAmount + threadIdx.z;
            

            if(j==0){
                inputShared[buff_idx] = curInput[input_idx];
            }
            else{
                //always right most column threads need to load
                if( threadIdx.x == (kernelSize-1) ){
                    inputShared[inputLoadOffset_idx] = curInput[input_idx];
                }
            }
            
            
            // filter loop
            for(int ok=0; ok<outputAmount; ok++){
                /**start Conv**/
                float b = bs[ok];
                float* w = ws + ok * kernelSize2 * inputAmount;
                float t_w = w[k_idx*inputAmount + threadIdx.z];
                //float tmp_r = t_w * __half2float(inputShared[inputBuffOffset_Idx]);
                filterShared[tid] = inputShared[inputBuffOffset_Idx] * t_w;
                //filterShared[tid] = __float2half(tmp_r);
                __syncthreads();
                //reduction
                int activeBuffSize = filterBufferSize;
                for (int stride = ceil(filterBufferSize / 2.0); stride > 1; stride = ceil(stride/2.0)) {
                    if (tid < stride && (tid + stride) < activeBuffSize ) {
                        //filterShared[tid] = __hadd(filterShared[tid], filterShared[tid + stride]);
                        filterShared[tid] += filterShared[tid + stride];
                    }
                    activeBuffSize = stride;
                    __syncthreads();
                }
                // write result back to outputbuff
                if (tid == 0){
                    //filterShared[0] = __hadd(filterShared[0], filterShared[1]);
                    filterShared[0] += filterShared[1];
                    //outputShared[ok] = __hadd(filterShared[tid], b);
                    outputShared[ok] = filterShared[tid] + b;
                }
            }
            //save output buffer to global Mem
            
            for(int loadChunk=0; loadChunk<outputAmount; loadChunk+=filterBufferSize){
                if(loadChunk + tid < outputAmount){
                    curOutput[outputAmount*(i*outputDim + j) + loadChunk + tid] = outputShared[loadChunk + tid];
                }
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
    int tid = threadIdx.x + threadIdx.y*kernelSize + threadIdx.z*kernelSize2;
    
    
    for(int ok=0; ok<outputAmount; ok++){
        float* curOutput = outputs + sp * outputArea;
        for(int tidx = 0; tidx < outputSize2; tidx += blockDim.x)
        {
            int idx = tidx + threadIdx.x;
            if(idx < outputSize2)
            {
                int x = idx / outputDim;
                int y = idx % outputDim;

                float val = 0.0;

                for(int c = 0; c < cfm; c++){
                    float* curInput = inputs + sp * inputArea;
                    float* w = ws + ok * kernelSize2 * inputAmount;
                    
                    /*put curInput and w into shared memory*/
                    for(int i = 0; i < kernelSize; i++){
                        int xx = x + i ;
                        for(int j = 0; j < kernelSize; j++){
                            int yy = y + j;
                            if(xx >= 0 && xx < inputDim && yy >= 0 && yy < inputDim)
                                val += curInput[cfm*(xx*inputDim + yy)+c] * w[cfm*(i * kernelSize + j) + c];
                        }
                    }
                }
                //HWC
                curOutput[outputAmount*(x*outputDim + y) + ok] = (val + bs[ok]);
            }
        }
    }
}