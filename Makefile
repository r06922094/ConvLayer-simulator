CC := /usr/local/cuda-9.0/bin/nvcc
CFLAGS = -g -I/usr/local/cuda/samples/common/inc/
OUT ?= ./build
SRC ?= ./src

$(OUT)/tensor.o: $(SRC)/tensor.cu $(SRC)/tensor.h $(OUT)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OUT)/MemoryAllocator.o: $(SRC)/MemoryAllocator.cu $(SRC)/MemoryAllocator.h $(OUT)
	$(CC) $(CLFAGS) -c -o $@ $<

$(OUT)/DataLayer.o: $(SRC)/DataLayer.cu $(SRC)/DataLayer.h $(OUT)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OUT)/ConvLayer.o: $(SRC)/ConvLayer.cu $(SRC)/ConvLayer.h $(OUT)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OUT)/net.o: $(SRC)/net.cu $(SRC)/net.h $(OUT)
	$(CC) $(CFLAGS) -c -o $@ $<

main: main.cu $(OUT)/tensor.o $(OUT)/MemoryAllocator.o \
	  $(OUT)/DataLayer.o $(OUT)/ConvLayer.o \
	  $(OUT)/net.o
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm $(OUT)/*.o main
