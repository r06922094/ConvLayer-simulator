#ifndef CONV_LAYER_H_
#define CONV_LAYER_H_

#include <stdio.h>
#include <stdlib.h>
#include "LayerBase.h"

typedef struct __ConvLayer ConvLayer;

struct __ConvLayer {
    LayerBase *lb;
    int kernelDim;
    int kernelAmount;
    tensor *weight;
    tensor *bias;
    int policy;
};

/* TODO: feedforward implementation */
void ConvLayer_init(ConvLayer **thiz, int batchSize, \
                    int inputDim, \
                    int inputChannel, \
                    int kernelDim, int kernelAmount, \
                    LayerBase *preLayer, LayerBase *nextLayer, int policy);

/* TODO: Random or Read from file? */
void ConvLayer_weight_init(ConvLayer *thiz);

void ConvLayer_bias_init(ConvLayer *thiz);

#endif
