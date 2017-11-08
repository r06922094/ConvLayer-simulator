#include "ConvLayer.h"

static void feedforward(LayerBase *thiz)
{
    ConvLayer *conv = (ConvLayer *) thiz;
    /* TODO: feedforward implementation */
    printf("Doing feedforward\n");
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
    //tensor *output = base->output;
    tensor_create(&base->output, base->batchSize, base->outputDim, base->outputDim, base->outputChannel);
    /******************************/
    base->preLayer = preLayer;
    base->nextLayer = NULL;
    base->feedforward = feedforward;
    /* ConvLayer */
    (*thiz)->kernelDim = kernelDim;
    (*thiz)->kernelAmount = kernelAmount;
    /* TODO: Initialize Weights and bias */
    (*thiz)->weight = NULL;
    (*thiz)->bias = NULL;
}
