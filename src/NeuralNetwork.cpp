#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

#include "Data.hpp"
#include "Layer.hpp"
#include "Matrix.hpp"
#include "NeuralNetwork.hpp"
#include "Neuron.hpp"

namespace NeuralNetwork
{
    MultilayerPerceptron::MultilayerPerceptron()
        :costFn(nullptr)
    {
        layers = std::vector<Layer>(0);
    };

    MultilayerPerceptron::MultilayerPerceptron(CostFn::CostFn* p_costFn)
        :costFn(p_costFn)
    {
        layers = std::vector<Layer>(0);
    };

    void MultilayerPerceptron::AddLayer(Layer layer)
    {
        if (layers.size() == 0) {
            layer.activationFn = nullptr;
            layers.push_back(layer);
            return;
        }
        
        layer.InitializeConnections(layers.back().neurons.size());
        layers.push_back(layer);
    }

    void MultilayerPerceptron::SetCostFunction(CostFn::CostFn* p_costFn)
    {
        costFn = p_costFn;
    }

    void MultilayerPerceptron::LoadDataInstance(Data input)
    {
        if (layers.size() < 1)
            throw std::invalid_argument("neural network layers are not defined");

        if (input.parameters.size() != layers[0].neurons.size()) 
            throw std::invalid_argument("input dimensions do not match specified dimensions");

        for (unsigned int i = 0; i < input.parameters.size(); i++) {
            layers[0].neurons[i].value = input.parameters[i];
        }
    }

    void MultilayerPerceptron::RunModel()
    {
        std::vector<float> output = layers[0].OutputVector();

        for (unsigned int i = 1; i < layers.size(); i++) {
            output = layers[i].CalculateValues(output);
        }
    }

    LinearParams MultilayerPerceptron::GradientDescent(std::vector<Data> batch, float learningRate = 0.01f)
    {
        std::vector<Math::Matrix> weightDerivatives = std::vector<Math::Matrix>(layers.back().neurons.size(), Math::Matrix(layers.back().connectionCount, 1));
        std::vector<float> biasDerivatives = std::vector<float>(layers.back().neurons.size(), 0);
        
        float avg = 0;
        float bavg = 0;

        for (auto data : batch) {
            LoadDataInstance(data);
            RunModel();

            for (unsigned int i = 0; i < layers.back().neurons.size(); i++) {
                NeuralNetwork::Neuron neuron = layers.back().neurons[i];
                float adjustment = layers.back().activationFn->dx(neuron.value)
                                        * costFn->dx(data.label, layers.back().activationFn->fn(neuron.value))
                                        / batch.size();

                weightDerivatives[i] += adjustment * Math::Matrix::ColumnMatrix(layers[layers.size() - 2].OutputVector());
                biasDerivatives[i] += adjustment;

                avg += adjustment * layers[layers.size() - 2].OutputVector()[0];
                bavg += adjustment * 1;
            }
        }

        // std::cout << avg << std::endl;
        // std::cout << bavg << std::endl;

        std::vector<std::vector<float>> weightAdjustments = {};
        std::vector<float> biasAdjustments = {};

        for (unsigned int i = 0; i < layers.back().neurons.size(); i++) {
            weightAdjustments.push_back(-weightDerivatives[i]);
            biasAdjustments.push_back(-biasDerivatives[i]);
        }
         
        layers.back().AdjustNeurons(weightAdjustments, biasAdjustments, learningRate);

        return LinearParams(weightAdjustments, biasAdjustments);
    }

    // LinearParams MultilayerPerceptron::Backpropagate(LinearParams changes)
    // {

    // }
    
    TrainTestPartition MultilayerPerceptron::PartitionData(std::vector<Data> data, float trainingDataRatio = 0.9f)
    {
        if (trainingDataRatio < 0 || trainingDataRatio > 1)
            throw std::invalid_argument("size of partition must be between 0 and 1");

        if ((int)(data.size() * trainingDataRatio) < 1 || (int)(data.size() * (1 - trainingDataRatio)) < 1)
            throw std::invalid_argument("partition must not return lists of size 0");

        std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());
        std::shuffle(std::begin(data), std::end(data), rng);

        unsigned int bound = data.size() * trainingDataRatio;
        std::vector<Data> trainingSet = {};
        std::vector<Data> testingSet = {};

        for (unsigned int i = 0; i < data.size(); i++) {
            if (i < bound) {
                trainingSet.push_back(data[i]);
            }
            else {
                testingSet.push_back(data[i]);
            }
        }

        return TrainTestPartition(trainingSet, testingSet);
    }
    
    std::tuple<float> MultilayerPerceptron::TestData(std::vector<Data> testingSet)
    {
        int correct = 0;
        int incorrect = 0;
        for (auto data : testingSet) {
            LoadDataInstance(data);
            RunModel();
            if ((layers.back().neurons[0].value > 0 && data.label == 1) || (layers.back().neurons[0].value < 0 && data.label == -1))
                correct++;
            else
                incorrect++;
        }

        float accuracy = (float)correct / (float)(correct + incorrect);

        return std::tuple<float>(accuracy);
    }


    void MultilayerPerceptron::Train(std::vector<Data> data, int epochs)
    {
        for (int epoch = 0; epoch < epochs; epoch++) {
            std::cout << "Epoch " << epoch << std::endl;

            TrainTestPartition partition = PartitionData(data);

            std::vector<Data> trainingSet = partition.first;
            std::vector<Data> testingSet = partition.second;

            GradientDescent(trainingSet, 0.001);

            std::tuple<float> results = TestData(testingSet);
            float accuracy = std::get<0>(results);

            std::cout << layers.back().neurons[0].weights[0] << "x + " << layers.back().neurons[0].weights[1] << "y + " << layers.back().neurons[0].bias << std::endl;

            std::cout << "Accuracy: " << accuracy << std::endl;
        }
    }

    // class MultilayerPerceptron
    // {
    //     Layer inputLayer;
    //     std::vector<Layer> hiddenLayer;
    //     Layer outputLayer;

    //     MultilayerPerceptron();

    //     void AddLayer(Layer layer);

    //     void LoadDataInstance(Data input);
    //     void RunModel();
    //     std::vector<float> GradientDescent(std::vector<Data> batch);
    //     std::vector<float> Backpropagate();
    //     void Train(std::vector<Data> data, int epochs);
    // };
}