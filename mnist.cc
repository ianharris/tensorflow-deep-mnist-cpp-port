#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <list>

// include tensorflow headers
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

#include "image.h"
#include "label.h"

void loadBatch(Image &trainImage, 
               Label &trainLabel)
{
    static int batchStartIndex = 0;
    int BATCH_SIZE = 50;

    trainImage.loadBatch(batchStartIndex, BATCH_SIZE);
    trainLabel.loadBatch(batchStartIndex, BATCH_SIZE);

    batchStartIndex += BATCH_SIZE;
    batchStartIndex %= trainImage.getNumImages();
}

tensorflow::ops::Variable weight_variable(tensorflow::Scope &root,
                                          tensorflow::ClientSession &session,
                                          std::initializer_list<tensorflow::int64> shape)
{
    std::cout << "Creating the weight variable" << std::endl << std::flush;
    // create the variable
    tensorflow::ops::Variable w(root, 
                                tensorflow::TensorShape(shape),
                                tensorflow::DT_FLOAT);
    std::cout << "Weight created" << std::endl << std::flush;
    // create the variables value
    tensorflow::ops::TruncatedNormal wval(root, 
                                          tensorflow::Input(shape),
                                          tensorflow::DT_FLOAT);
    std::cout << "Weight val created" << std::endl << std::flush;
    // assign the variable to the value
    tensorflow::Output assign_w = tensorflow::ops::Assign(root,
                                                          w,
                                                          wval);
    std::cout << "Weight assign prepared" << std::endl << std::flush;
    TF_CHECK_OK(session.Run({assign_w},
                            nullptr));
    std::cout << "Weight assign run" << std::endl << std::flush;
    return w;
}

tensorflow::ops::Variable bias_variable(tensorflow::Scope &root,
                                        tensorflow::ClientSession &session,
                                        std::initializer_list<tensorflow::int64> shape)
{
    // create the bias variable
    std::cout << "Creating the bias variable" << std::endl << std::flush;
    tensorflow::ops::Variable b(root,
                                tensorflow::TensorShape(shape),
                                tensorflow::DT_FLOAT);
    std::cout << "Bias Variable created" << std::endl << std::flush;
    // create the variables value
    // tensorflow::ops::Fill bval(root,
    //                            tensorflow::Input(shape),
    //                            tensorflow::Input(0.1));
    tensorflow::ops::TruncatedNormal bval(root, 
                                          tensorflow::Input(shape),
                                          tensorflow::DT_FLOAT);
    std::cout << "Bias variable value created" << std::endl << std::flush;

    // assign the value to the variable
    tensorflow::ops::Assign assign_b(root,
                                     b,
                                     bval);
    assign_b.ValidateShape(true);
    std::cout << "Bias value assignment created" << std::endl << std::flush;

    TF_CHECK_OK(session.Run({assign_b},
                            nullptr));
    std::cout << "bias assignment run" << std::endl << std::flush;
    return b;
}

tensorflow::Output deepnn(tensorflow::Scope &root,
                          tensorflow::ClientSession &session,
                          tensorflow::ops::Placeholder &x)
{

    std::cout << "Creating the first layer" << std::endl << std::flush;
    // create the first convolutional layer
    // - create the weight variable
    tensorflow::ops::Variable w1 = weight_variable(root,
                                                   session,
                                                   {5, 5, 1, 32});
    // - create the bias variable
    tensorflow::ops::Variable b1 = bias_variable(root,
                                                 session,
                                                 {32});
    
    std::cout << "First layer weights / biases created" << std::endl << std::flush;
    // - perform the 2D convolution
    tensorflow::ops::Conv2D conv1(root,
                                  x,
                                  w1,
                                  {1, 1, 1, 1},
                                  "SAME");
    
    // - add the convolution and bias and then perform ReLU activation
    tensorflow::ops::Relu relu1(root,
                                tensorflow::ops::Add(root,
                                                     conv1,
                                                      b1));
    
    // - apply the max_pool layer
    tensorflow::ops::MaxPool h1(root,
                                relu1,
                                {1, 2, 2, 1},
                                {1, 2, 2, 1},
                                "SAME");
    
    std::cout << "Creating the second layer" << std::endl << std::flush;
    // create the second convolutional layer
    // - create the weight variable
    tensorflow::ops::Variable w2 = weight_variable(root,
                                                   session,
                                                   {5, 5, 32, 64});
    // - create the bias variable
    tensorflow::ops::Variable b2 = bias_variable(root,
                                                 session,
                                                 {64});
    // - perform the 2D convolution
    tensorflow::ops::Conv2D conv2(root,
                                  h1,
                                  w2,
                                  {1, 1, 1, 1},
                                  "SAME");
    // - add the convolution and bias and then perform ReLU activation
    tensorflow::ops::Relu relu2(root,
                                tensorflow::ops::Add(root,
                                                     conv2,
                                                     b2));
    // - apply the max_pool layer
    tensorflow::ops::MaxPool h2(root,
                                relu2,
                                {1, 2, 2, 1},
                                {1, 2, 2, 1},
                                "SAME");

    std::cout << "Creating the first fc layer" << std::endl << std::flush;
    // create the fully connected layer
    // - create the weight variable
    tensorflow::ops::Variable w_fc1 = weight_variable(root,
                                                      session,
                                                      {7 * 7 * 64, 1024});
    // - create the bias variable
    tensorflow::ops::Variable b_fc1 = bias_variable(root,
                                                    session,
                                                    {1024});

    // - create a flattened h2
    tensorflow::ops::Reshape h2_flat(root,
                                     h2,
                                     {-1, 7 * 7 * 64});

    // - do the matrix multiplication
    tensorflow::ops::Relu h_fc1(root,
                                tensorflow::ops::Add(root,
                                                     tensorflow::ops::MatMul(root,
                                                                             h2_flat,
                                                                             w_fc1),
                                                     b_fc1));

    // FIXME add dropout here
    std::cout << "Creating the output layer" << std::endl << std::flush;

    // create the output latyer
    // - create the weight variable
    tensorflow::ops::Variable w_fc2 = weight_variable(root,
                                                      session,
                                                      {1024, 10});

    // - create the bias variable
    tensorflow::ops::Variable b_fc2 = bias_variable(root,
                                                    session,
                                                    {10});

    // - do the matrix multiplication
    tensorflow::ops::Add y_conv(root,
                                tensorflow::ops::MatMul(root,
                                                        h_fc1,
                                                        w_fc2),
                                b_fc2);
    
    // return the y_conv
    return y_conv;
}

int main(int argc,
         char **argv)
{
    // define the Images
    Image trainImages("mnist/data/train-images-idx3-ubyte");
    Image testImages("mnist/data/t10k-images-idx3-ubyte");
    // read the images
    trainImages.read();
    testImages.read();

    // define the labels
    Label trainLabels("mnist/data/train-labels-idx1-ubyte");
    Label testLabels("mnist/data/t10k-labels-idx1-ubyte");
    // read the labels
    trainLabels.read();
    testLabels.read();

    // load a batch
    loadBatch(trainImages, trainLabels);

    tensorflow::Scope root = tensorflow::Scope::NewRootScope();
    tensorflow::ClientSession session(root);
 
    // create a placeholder for X
    tensorflow::ops::Placeholder x(root.WithOpName("X"), 
                                   tensorflow::DT_FLOAT, 
                                   tensorflow::ops::Placeholder::Attrs().Shape(tensorflow::PartialTensorShape({-1, 28, 28, 1})));

    // create a placeholder for y
    tensorflow::ops::Placeholder y(root.WithOpName("y"),
                                   tensorflow::DT_UINT8,
                                   tensorflow::ops::Placeholder::Attrs().Shape(tensorflow::PartialTensorShape({-1, 10})));

    tensorflow::Output y_conv = deepnn(root,
                                       session,
                                       x);

    std::cout << "Returned from deepnn" << std::endl << std::flush;

    tensorflow::Tensor tensor(tensorflow::DT_FLOAT,
                              tensorflow::TensorShape({1, 28, 28, 1}));
    float tval[784];
    int i;
    for(i=0;i<784;++i) tval[i] = 1;

    std::memcpy(tensor.flat<float>().data(), tval, 784*sizeof(float));

    tensorflow::ClientSession::FeedType feed;
    // feed.emplace(x, tensorflow::Input::Initializer(*(trainImages.batchTensor)));
    feed.emplace(x, tensorflow::Input::Initializer(tensor));
    TF_CHECK_OK(session.Run(feed, {y_conv}, nullptr));

    // session.Run({{x, trainImages.batchTensor}}, {y_conv}, nullptr);

    // TODO remove this is only for checking
    // output a layer of the tensor to check it was read properly
    auto flat = trainImages.batchTensor->flat<float>();
    std::ofstream out;
    out.open("output.txt", std::ios::out);
    int j;

    int numRows = trainImages.getNumRows();
    int numColumns = trainImages.getNumColumns();

    for (i=0;i<numRows;++i) {
        for(j=0;j<numColumns;++j) {
            out << std::setfill('0') << std::setw(3) << flat(j + i * numColumns + (numRows * numColumns * 12)) << " ";
            // out << flat(j + i * numColumns + (numRows * numColumns * 0)) << " ";
        }
        out << std::endl << std::flush;
    }
    out.close();
    // end of TODO

    return 0;
}
