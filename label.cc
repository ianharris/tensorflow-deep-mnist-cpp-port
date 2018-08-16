#include <fstream>

#include "tensorflow/core/framework/tensor.h"
#include "image.h"
#include "label.h"

#include "utils.h"

Label::Label()
{
    tensor = NULL;
    batchTensor = NULL;
}

Label::Label(std::string labelUri)
{
    uri = labelUri;
    tensor = NULL;
    batchTensor = NULL;
}

Label::~Label()
{
    if(tensor) delete tensor;
    if(batchTensor) delete batchTensor;
}

void Label::read(void)
{
    std::ifstream in;

    // open the input file for reading
    in.open(uri,
            std::ios::in | std::ios::binary);

    mn = readBigEndianInt(in);
    numLabels = readBigEndianInt(in);

    // check the magic number
    if(mn != LABEL_MAGIC_NUMBER) {
        std::cout << "Magic Number in image file is " << mn << " when it should be " << IMAGE_MAGIC_NUMBER << std::endl;
    }

    // read the remainder of the data
    std::cout << "Allocating " << numLabels << " bytes!" << std::endl;
    // allocate memory
    data = (char *)std::malloc(numLabels);
    // read the data
    in.read(data, numLabels);

    // create the tensor
    tensor = new tensorflow::Tensor(tensorflow::DT_UINT8, 
                                    tensorflow::TensorShape({ numLabels, 1 })); 
    
    // copy the read data into the tensor
    std::memcpy(tensor->flat<uint8_t>().data(), 
                data, 
                numLabels);

    in.close();

}

void Label::loadBatch(int batchStartIndex, int batchSize)
{
    // allocate the batch tensor if NULL
    if(!batchTensor) {
        batchTensor = new tensorflow::Tensor(tensorflow::DT_UINT8,
                                            tensorflow::TensorShape({ batchSize, 1 })); 
    }

    // load the next batch
    std::memcpy(batchTensor->flat<uint8_t>().data(), 
                tensor->flat<uint8_t>().data() + batchStartIndex,
                batchSize);
}

