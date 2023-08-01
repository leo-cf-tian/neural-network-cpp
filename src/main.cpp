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
    model.AddLayer(Layer(3));
    model.AddLayer(Layer(1, new ActivationFn::Linear()));

    vector<Data> dataset = {};

    for (int i = 0; i < 1000; i++) {
        srand(std::chrono::system_clock::now().time_since_epoch().count());

        float a = rand() / static_cast<float>(RAND_MAX) * 40 - 20;
        float b = rand() / static_cast<float>(RAND_MAX) * 40 - 20;
        float c = rand() / static_cast<float>(RAND_MAX) * 40 - 20;
        
        float value = -2 * a + b + 3 * c - 5;

        dataset.push_back(Data({a, b, c}, value));
    }

    model.Train(dataset, 1000);
    
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);
}