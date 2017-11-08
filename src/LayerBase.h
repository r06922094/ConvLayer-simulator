#ifndef LAYER_BASE_H_
#define LAYER_BASE_H_

#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"

typedef struct __LayerBase LayerBase;

struct __LayerBase {
    /* Virtual Functions */
    void (*feedforward)(LayerBase *thiz);

    /* Member Data */
    int inputDim, outputDim;
    int inputChannel, outputChannel;
    int batchSize;
    tensor *input;
    tensor *output;
    LayerBase *preLayer;
    LayerBase *nextLayer;
};

#endif
