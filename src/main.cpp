#include <iostream>
#include <vector>
#include <chrono>

#include "Data.hpp"
#include "NeuralNetwork.hpp"

using namespace NeuralNetwork;
using namespace std::chrono;
using namespace std;

int main(void) {
    srand(time(NULL));

    auto start = high_resolution_clock::now();

    MultilayerPerceptron model(new CostFn::SparseCategoricalCrossEntropy());
    model.AddLayer(Layer(28 * 28));
    model.AddLayer(Layer(128, new ActivationFn::ReLU()));
    model.AddLayer(Layer(64, new ActivationFn::ReLU()));
    model.AddLayer(Layer(32, new ActivationFn::ReLU()));
    model.AddLayer(Layer(10, new ActivationFn::LogisticSigmoid()));

    // vector<Data> dataset = {};

    // for (int i = 0; i < 1000; i++) {
    //     srand(std::chrono::system_clock::now().time_since_epoch().count());

    //     double a = rand() / static_cast<double>(RAND_MAX) * 20 - 10;
    //     double b = rand() / static_cast<double>(RAND_MAX) * 20 - 10;
        
    //     double value = a * a * a + 4 * b * b + 10 > 0;

    //     dataset.push_back(Data({a, b}, value));
    // }

    // Data::TrainTestPartition data = Data::PartitionData(dataset);

    Data::TrainTestPartition data = Data::LoadMNIST();

    // cout << data.first[1].label << endl;
    // for (int i = 0; i < 28; i++) {
    //     for (int j = 0; j < 28; j++) {
    //         cout << (data.first[1].parameters[i * 28 + j][0] == 0 ? " " : "0");
    //     }
    //     cout << endl;
    // }

    model.Train(data.first, data.second, 5000, 0.1, 128);
    
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);
 
    cout << duration.count() << endl;
}