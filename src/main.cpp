#include <iostream>
#include <vector>
#include <chrono>

#include "Data.hpp"
#include "NeuralNetwork.hpp"

using namespace NeuralNetwork;
using namespace std::chrono;
using namespace std;

int main(void) {

    auto start = high_resolution_clock::now();

    MultilayerPerceptron model(new CostFn::L2());
    model.AddLayer(Layer(2));
    model.AddLayer(Layer(10, new ActivationFn::LeakyReLU()));
    model.AddLayer(Layer(1, new ActivationFn::LogisticSigmoid()));

    vector<Data> dataset = {};

    for (int i = 0; i < 1000; i++) {
        srand(std::chrono::system_clock::now().time_since_epoch().count());

        double a = rand() / static_cast<double>(RAND_MAX) * 40 - 20;
        double b = rand() / static_cast<double>(RAND_MAX) * 40 - 20;
        
        double value = a + b < 10 ? 1 : -1;

        dataset.push_back(Data({a, b}, value));
    }

    model.Train(dataset, 1000, 0.01f);
    
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);
}