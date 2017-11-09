#ifndef DATA_LAYER_H_
#define DATA_LAYER_H_

#include <stdio.h>
#include <stdlib.h>
#include "LayerBase.h"

typedef struct __DataLayer DataLayer;

struct __DataLayer {
    LayerBase *lb;
};

/* TODO: feedforward implementation */
void DataLayer_init(DataLayer **thiz, int batchSize, \
                    int inputDim, \
                    int inputChannel, \
                    LayerBase *nextLayer);

#endif
