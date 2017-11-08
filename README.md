# ConvLayer-simulator
A Neural Network which only have Data Layer and Convolution Layer with CUDA implementation.

## Build
* ```$ mkdir build```
* ```$ make main```

## How
* You con modify definition of input in ```src/net.h```
* If you want to redefine network, you can modify **buildNetwork()** in ```src/net.cu```. And remember to set the relation of each layer.
* Also you need to add some code in **trainNetwork()** if structure of network modified.

## TODO
* Convolution Layer implementation
* Initialize weights and bias
  * Read from File
  * Random
* Construct a Network by a Config File
