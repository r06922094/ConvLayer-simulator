#ifndef NET_H_
#define NET_H_

#include "DataLayer.h"
#include "ConvLayer.h"

#define _BATCH_SIZE 1
#define _INPUT_DIM 29
#define _IMAGE_CHANNEL 3
#define _CONV_FILTER_NUM 32
#define _CONV_FILTER_DIM 3

LayerBase *buildNetwork();

tensor *trainNetwork(LayerBase *head, tensor *x);

#endif
