#include "net.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

LayerBase *buildNetwork(int _INPUT_DIM, int _IMAGE_CHANNEL)
{
    DataLayer *dat;
    ConvLayer *conv;

    DataLayer_init(&dat, _BATCH_SIZE, _INPUT_DIM, \
                   _IMAGE_CHANNEL, NULL);

    ConvLayer_init(&conv, _BATCH_SIZE, _INPUT_DIM, \
                   _IMAGE_CHANNEL, _CONV_FILTER_DIM, \
                   _CONV_FILTER_NUM, dat->lb, NULL);
    
    /* Cheat */
    dat->lb->nextLayer = (LayerBase *) conv;

    return (LayerBase *) dat;
}

/* Split data into batches */
/* N means "batch index" */
tensor *next_batch(tensor *x, int N)
{
   int area = x->D1 * x->D2 * x->D3;
   tensor *ret = NULL;
   tensor_create(&ret, _BATCH_SIZE, x->D1, x->D2, x->D3);

   memcpy(ret->hostData, x->hostData + N * area * _BATCH_SIZE * sizeof(conv_unit_t), area * _BATCH_SIZE * sizeof(conv_unit_t));

   return ret;
}

tensor *trainNetwork(LayerBase *head, tensor *x)
{
    DataLayer *dat = (DataLayer *) head;
    ConvLayer *conv = (ConvLayer *) dat->lb->nextLayer; 

    /* Start Training */
    for (int i = 0; i < x->D0; i += _BATCH_SIZE) {
        tensor *x_bat = next_batch(x, i);
        dat->lb->input = x_bat;

        /* Start feedforward */
        dat->lb->feedforward((LayerBase *) dat);
        conv->lb->input = dat->lb->output;
        conv->lb->feedforward((LayerBase *) conv);
    }

    conv->lb->output->toCpu(conv->lb->output);
    return conv->lb->output;
}