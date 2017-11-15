#ifndef NET_H_
#define NET_H_

#include "DataLayer.h"
#include "ConvLayer.h"

#define _BATCH_SIZE 1
#define _CONV_FILTER_NUM 1
#define _CONV_FILTER_DIM 3

LayerBase *buildNetwork(int _INPUT_DIM, int _IMAGE_CHANNEL, int policy);

tensor *trainNetwork(LayerBase *head, tensor *x);

#endif
