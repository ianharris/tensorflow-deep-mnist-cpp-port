#include <iomanip>
#include <string>
#include <fstream>
#include <iostream>

#include "tensorflow/core/framework/tensor.h"

#include "image.h"

int Image::getNumImages(void)
{
    return mn;
}

int Image::getNumRows(void)
{
    return rows;
}

int Image::getNumColumns(void)
{
    return columns;
}

Image::~Image()
{
    if(tensor) delete tensor;
    if(batchTensor) delete batchTensor;
}

Image::Image()
{
    tensor = NULL;
    batchTensor = NULL;
}

Image::Image(std::string imageuri)
{
    uri = imageuri;
    tensor = NULL;
    batchTensor = NULL;
}

void Image::read(void)
{
    int i;
    std::ifstream in;
   
    // open the input file for reading
    in.open(uri, 
            std::ios::in | std::ios::binary);
    
    mn = readBigEndianInt(in);
    numImages = readBigEndianInt(in);
    rows = readBigEndianInt(in);
    columns = readBigEndianInt(in);

    // check the magic number
    if(mn != IMAGE_MAGIC_NUMBER) {
        std::cout << "Magic Number in image file is " << mn << " when it should be " << IMAGE_MAGIC_NUMBER << std::endl;
    }

    // read the remainder of the data
    std::cout << "Allocating " << numImages * rows * columns << " bytes!" << std::endl;
    // allocate memory
    data = (uint8_t *)std::malloc(numImages * rows * columns);
    // read the data
    in.read((char *)data, numImages * rows * columns);

    // create the tensor
    tensor = new tensorflow::Tensor(tensorflow::DT_DOUBLE, 
                                    tensorflow::TensorShape({ numImages, rows, columns, 1 })); 
    
    // copy the read data into the tensor
    auto dptr = tensor->flat<double>().data();

    for(i=0; i<numImages * rows * columns; ++i) {
        dptr[i] = double(data[i]);
    }

    in.close();
}

void Image::loadBatch(int batchStartIndex, int batchSize)
{
    // allocate the batch tensor if NULL
    if(!batchTensor) {
        batchTensor = new tensorflow::Tensor(tensorflow::DT_DOUBLE,
                                             tensorflow::TensorShape({ batchSize, rows, columns, 1 })); 
    }

    // load the next batch
    std::memcpy(batchTensor->flat<double>().data(), 
                tensor->flat<double>().data() + batchStartIndex * rows * columns * sizeof(double),
                batchSize * rows * columns * sizeof(double));
}

