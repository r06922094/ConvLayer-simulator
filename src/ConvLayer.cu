#include "ConvLayer.h"
#include "tensor.h"

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

__global__ void g_ConvCFM_print();

static void feedforward(LayerBase *thiz)
{
    ConvLayer *conv = (ConvLayer *) thiz;
    conv->lb->output->toGpu(conv->lb->output);

    /**check shared memory usage**/
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int sharedMemSize = (unsigned int)deviceProp.sharedMemPerBlock;
    int suInput=0;
    int suOutput=0;
    int suFilter=0;
    int filterSize = sizeof(float)*conv->kernelDim * conv->kernelDim * conv->lb->inputChannel;
    /**mini buff, IR**/
    suInput  = filterSize;
    suOutput = sizeof(float) * conv->kernelAmount;
    suFilter = ((int)((sharedMemSize - suInput - suOutput)/filterSize))*filterSize;
    if(suInput+suOutput+suFilter > sharedMemSize){
        printf("mini IR init error\n");
		exit(0);
    }
    dim3 block = dim3(conv->lb->batchSize);
    dim3 thread= dim3(conv->kernelDim , conv->kernelDim , conv->lb->inputChannel);
    int sharedMemRequire = suInput + suOutput + suFilter;
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


    /*naive Version*/
    /*
    dim3 block = dim3(conv->lb->batchSize);
    dim3 thread= dim3(conv->kernelDim * conv->kernelDim * conv->lb->inputChannel);
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
    */
    //printf("Doing feedforward\n");
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
    base->input = preLayer->output;
    /* Allocate memory for output */
    tensor_create(&base->output, base->batchSize, base->outputDim, base->outputDim, base->outputChannel);
    
    /******************************/
    base->preLayer = preLayer;
    base->nextLayer = NULL;
    base->feedforward = feedforward;
    /* ConvLayer */
    (*thiz)->kernelDim = kernelDim;
    (*thiz)->kernelAmount = kernelAmount;
    /* TODO: Initialize Weights and bias */
    ConvLayer_weight_init(*thiz);
    ConvLayer_bias_init(*thiz);
    /*
    (*thiz)->weight = NULL;
    (*thiz)->bias = NULL;
    */
}

void ConvLayer_weight_init(ConvLayer *thiz)
{
    tensor_create(&thiz->weight, thiz->kernelAmount, thiz->kernelDim, thiz->kernelDim, thiz->lb->inputChannel);
    tensor *tzr = thiz->weight;
    for (int i = 0; i < tzr->D0; i++) {
        for (int j = 0; j < tzr->D1; j++) {
            for (int k = 0; k < tzr->D2; k++) {
                for (int w = 0; w < tzr->D3; w++) {
                    tzr->set(tzr, i, j, k, w, 0.5*(i+1));
                }
            }
        }
    }
    tzr->toGpu(tzr);
}

void ConvLayer_bias_init(ConvLayer *thiz)
{
    tensor_create(&thiz->bias, thiz->kernelAmount, 1, 1, 1);
    tensor *tzr = thiz->bias;
    for (int i = 0; i < tzr->D0; i++) {
        tzr->set(tzr, i, 0, 0, 0, 100);
    }
    thiz->bias->toGpu(thiz->bias);
}

__global__ void g_ConvCFM_print(){
    printf("hello~, I'm %d\n", threadIdx.x);
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
    //int outputArea = outputSize2* outputAmount;
    //declare shared mem
    extern __shared__ float bufferShared[];
    //point to diff addr
    //output: save back to global sharedMemPerBlock
    //filter: save the partial sum to do reduction
    int suInput = kernelSize * kernelSize * inputAmount;
    int suOutput = outputAmount;
    int filterBufferSize = blockDim.x  * blockDim.y  * blockDim.z;
    float *inputShared  = bufferShared;
    float *outputShared = bufferShared + suInput;
    float *filterShared = bufferShared + suInput + suOutput;
    
    int tid = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    // input loop
    //i: column
    for(int i=0; i<inputDim-kernelSize+1; i++){
        //j:row
        for(int j=0; j<inputDim-kernelSize+1; j++){
            //load input to buffer
            //(xx, yy): threadId map to input coordinate
            int xx = j + threadIdx.x;
            int yy = i + threadIdx.y;

            int k_idx = threadIdx.x + threadIdx.y*kernelSize;
            
            //move to addr of just one channel
            float* curInput = inputs + sp * inputArea;
            

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
            
            __syncthreads();
            for(int ok=0; ok<outputAmount; ok++){
                float b = bs[ok];
                float* w = ws + ok * kernelSize2 * inputAmount;
                float t_w = w[k_idx*inputAmount + threadIdx.z];
                
                filterShared[tid] = inputShared[inputBuffOffset_Idx] * t_w;
                __syncthreads();
                //reduction
                int activeBuffSize = filterBufferSize;
                for (int stride = (filterBufferSize / 2.0); stride > 0; stride = stride>>1) {
                    if (tid < stride && (tid + stride) < activeBuffSize ) {
                        filterShared[tid] += filterShared[tid + stride];
                    }
                    activeBuffSize = stride;
                    __syncthreads();
                }
                //save result to output buffer
                if(tid==0){
                    outputShared[ok] = filterShared[tid] + b;
                }
            }
            //save output buffer to global Mem
            if(tid < outputAmount){
                //CHW addr, need to modify to HWC
                float* curOutput = outputs + sp * outputSize2;
                curOutput[outputAmount*(i*outputDim + j) + tid] = outputShared[tid];
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
                curOutput[outputAmount*(x*outputDim + y) + ok] = val + bs[ok];
            }
        }
    }
}