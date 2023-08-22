#include <iostream>
#include <vector>
#include <chrono>

#include "Data.hpp"
#include "NeuralNetwork.hpp"

using namespace NeuralNetwork;
using namespace std::chrono;
using namespace std;

int main(void) {

    MultilayerPerceptron model(new CostFn::L2());
    model.AddLayer(Layer(2));
    model.AddLayer(Layer(10, new ActivationFn::LeakyReLU()));
    model.AddLayer(Layer(1, new ActivationFn::LogisticSigmoid()));

    vector<Data> dataset = {};

    for (int i = 0; i < 1000; i++) {
        srand(std::chrono::system_clock::now().time_since_epoch().count());

        float a = rand() / static_cast<float>(RAND_MAX) * 40 - 20;
        float b = rand() / static_cast<float>(RAND_MAX) * 40 - 20;
        
        float value = a * a + b * b < 100 ? 1 : -1;

        dataset.push_back(Data({a, b}, value));
    }

    model.Train(dataset, 10000, 0.001);
}