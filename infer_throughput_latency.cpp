#include "src/mnist.h"
#include "src/network.h"
#include "src/layer.h"

#include <iomanip>

#include <chrono>
#include <thread>

using namespace cudl;

std::vector<std::vector<double>> latencies;

void func(int thr_id) {
    std::vector<double>& my_latencies = latencies[thr_id];

    constexpr int batch_size_test = 1;
    constexpr int num_steps_test = 10000;

    Network model;
    //model.add_layer(new Conv2D("conv1", 20, 5));
    //model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));
    //model.add_layer(new Conv2D("conv2", 50, 5));
    //model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));
    //model.add_layer(new Dense("dense1", 500));
    //model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
    //model.add_layer(new Dense("dense2", 10));
    //model.add_layer(new Softmax("softmax"));
    model.add_layer(new Conv2D("conv1", 8, 5));
    model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));
    model.add_layer(new Conv2D("conv2", 16, 5));
    model.add_layer(new Pooling("pool", 3, 0, 3, CUDNN_POOLING_MAX));
    model.add_layer(new Dense("dense2", 10));
    model.cuda();

    model.train();

    MNIST test_data_loader = MNIST("./dataset");
    test_data_loader.test(batch_size_test);

    model.test();
    
    Blob<float> *test_data = test_data_loader.get_data();
    Blob<float> *test_target = test_data_loader.get_target();
    auto very_start_time = std::chrono::steady_clock::now();
    while (true) {
        test_data_loader.get_batch();
        int step = 0;
        while (step < num_steps_test) {
            // update shared buffer contents
            test_data->to(cuda);
            test_target->to(cuda);

            // forward
            auto start_time = std::chrono::steady_clock::now();
            model.forward(test_data);
            model.get_output();
            auto end_time = std::chrono::steady_clock::now();
            auto latency = std::chrono::duration<double, std::micro>(end_time - start_time).count();
            auto time_elasped = std::chrono::duration<double, std::micro>(end_time - very_start_time).count();
            if (time_elasped > 10000000) { // 10s
                my_latencies.push_back(latency);
            }

            if (time_elasped > 30000000) { // 30s
                return;
            }

            // fetch next data
            step = test_data_loader.next();
        }
    }
}

int main(int argc, char* argv[]) {
    int num_thrs = atoi(argv[1]);
    const char* output_path = argv[2];

    latencies.resize(num_thrs);

    std::vector<std::thread> thrs;
    thrs.reserve(num_thrs);

    for (int i = 0; i < num_thrs; ++i) {
        thrs.emplace_back(func, i);
    }

    for (std::thread& thr : thrs) {
        thr.join();
    }

    unsigned long long total_infer = 0;
    std::vector<double> all_latencies;
    for (std::vector<double>& my_latencies : latencies) {
        total_infer += my_latencies.size();
        all_latencies.insert(all_latencies.end(), my_latencies.begin(), my_latencies.end());
    }

    double throughput = total_infer / 20.0;

    std::sort(all_latencies.begin(), all_latencies.end());
    
    double p50 = all_latencies[all_latencies.size() / 2];
    double p90 = all_latencies[all_latencies.size() * 0.90];
    double p95 = all_latencies[all_latencies.size() * 0.95];
    double p99 = all_latencies[all_latencies.size() * 0.99];

    FILE* fp = fopen(output_path, "a");
    fprintf(fp, "%f,%f,%f,%f,%f\n", throughput, p50, p90, p95, p99);
    fclose(fp);
}
