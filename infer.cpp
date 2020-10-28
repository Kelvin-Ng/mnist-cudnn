#include "src/mnist.h"
#include "src/network.h"
#include "src/layer.h"

#include <iomanip>
#include <nvtx3/nvToolsExt.h>

#include <chrono>

using namespace cudl;

int main(int argc, char* argv[])
{
    bool load_pretrain = false;

    //int batch_size_test = 10;
    //int num_steps_test = 1000;
    int batch_size_test = 1;
    int num_steps_test = 10000;


    /* Welcome Message */
    std::cout << "== MNIST training with CUDNN ==" << std::endl;

    Network model;
    model.add_layer(new Conv2D("conv1", 20, 5));
    model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));
    model.add_layer(new Conv2D("conv2", 50, 5));
    model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));
    model.add_layer(new Dense("dense1", 500));
    model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
    model.add_layer(new Dense("dense2", 10));
    model.add_layer(new Softmax("softmax"));
    model.cuda();

    if (load_pretrain)
        model.load_pretrain();
    model.train();

    // phase 2. inferencing
    // step 1. load test set
    std::cout << "[INFERENCE]" << std::endl;
    MNIST test_data_loader = MNIST("./dataset");
    test_data_loader.test(batch_size_test);

    // step 2. model initialization
    model.test();
    
    // step 3. iterates the testing loop
    Blob<float> *test_data = test_data_loader.get_data();
    Blob<float> *test_target = test_data_loader.get_target();
    test_data_loader.get_batch();
    int step = 0;
    int tp_count = 0;
    while (step < num_steps_test)
    {
        // nvtx profiling start
        std::string nvtx_message = std::string("step" + std::to_string(step));
        nvtxRangePushA(nvtx_message.c_str());

        // update shared buffer contents
		test_data->to(cuda);
		test_target->to(cuda);

        // forward
        auto start_time = std::chrono::steady_clock::now();
        model.forward(test_data);
        tp_count += model.get_accuracy(test_target);
        auto end_time = std::chrono::steady_clock::now();
        auto time_taken = end_time - start_time;
        std::cout << std::chrono::duration<double, std::micro>(time_taken).count() << std::endl;

        // fetch next data
        step = test_data_loader.next();

        // nvtx profiling stop
        nvtxRangePop();
    }

    // step 4. calculate loss and accuracy
    float loss = model.loss(test_target);
    float accuracy = 100.f * tp_count / num_steps_test / batch_size_test;

    std::cout << "loss: " << std::setw(4) << loss << ", accuracy: " << accuracy << "%" << std::endl;

    std::cout << "Done." << std::endl;

    return 0;
}
