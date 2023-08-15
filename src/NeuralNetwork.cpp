#include <iostream>
#include <cmath>
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
        std::vector<double> output = layers[0].OutputVector();

        for (unsigned int i = 1; i < layers.size(); i++) {
            output = layers[i].CalculateValues(output);
        }
    }

    LinearParams MultilayerPerceptron::GradientDescent(std::vector<Data> batch, double learningRate)
    {
        Layer &layer = layers.back();

        std::vector<Math::Matrix> weightDerivatives = std::vector<Math::Matrix>(layer.neurons.size(), Math::Matrix(layer.connectionCount, 1));
        std::vector<double> biasDerivatives = std::vector<double>(layer.neurons.size(), 0);
        std::vector<double> prevValueDerivatives = std::vector<double>(layer.connectionCount, 0);

        for (auto data : batch) {
            LoadDataInstance(data);
            RunModel();

            for (unsigned int i = 0; i < layer.neurons.size(); i++) {
                NeuralNetwork::Neuron neuron = layer.neurons[i];
                double adjustment = layer.activationFn->dx(neuron.value)
                                        * costFn->dx(data.label, layer.activationFn->fn(neuron.value))
                                        / batch.size();

                weightDerivatives[i] += adjustment * Math::Matrix::ColumnMatrix(layers[layers.size() - 2].OutputVector());
                biasDerivatives[i] += adjustment;
                for (unsigned int j = 0; j < layer.connectionCount; j++)
                {
                    prevValueDerivatives[j] += adjustment * layer.neurons[i].weights[j];
                }
            }
        }

        std::vector<std::vector<double>> weightAdjustments = {};
        std::vector<double> biasAdjustments = {};

        for (unsigned int i = 0; i < layers.back().neurons.size(); i++) {
            weightAdjustments.push_back(-weightDerivatives[i]);
            biasAdjustments.push_back(-biasDerivatives[i]);
        }
         
        layer.AdjustNeurons(weightAdjustments, biasAdjustments, learningRate);

        return LinearParams(weightDerivatives, biasDerivatives, prevValueDerivatives);
    }

    LinearParams MultilayerPerceptron::Backpropagate(LinearParams changes, std::size_t layerIndex)
    {
        std::vector<Math::Matrix> nextWeightAdjustments = std::get<0>(changes);
        std::vector<double> nextBiasAdjustments = std::get<1>(changes);
        std::vector<double> valueAdjustments = std::get<2>(changes);

        Layer &layer = layers[layerIndex];

        std::vector<Math::Matrix> weightDerivatives = std::vector<Math::Matrix>(layer.neurons.size(), Math::Matrix(layer.connectionCount, 1));
        std::vector<double> biasDerivatives = std::vector<double>(layer.neurons.size(), 0);
        std::vector<double> prevValueDerivatives = std::vector<double>(layer.connectionCount, 0);

        for (unsigned int i = 0; i < layer.neurons.size(); i++) {
            NeuralNetwork::Neuron neuron = layer.neurons[i];
            double adjustment = layer.activationFn->dx(neuron.value)
                                    * valueAdjustments[i];

            weightDerivatives[i] += adjustment * Math::Matrix::ColumnMatrix(layers[layerIndex - 1].OutputVector());
            biasDerivatives[i] += adjustment;

            for (unsigned int j = 0; j < layer.connectionCount; j++)
            {
                prevValueDerivatives[j] += adjustment * layer.neurons[i].weights[j];
            }
        }
        
        std::vector<std::vector<double>> weightAdjustments = {};
        std::vector<double> biasAdjustments = {};

        for (unsigned int i = 0; i < layer.neurons.size(); i++) {
            weightAdjustments.push_back(-weightDerivatives[i]);
            biasAdjustments.push_back(-biasDerivatives[i]);
        }
        
        layer.AdjustNeurons(weightAdjustments, biasAdjustments);

        return LinearParams(weightDerivatives, biasDerivatives, prevValueDerivatives);
    }
    
    TrainTestPartition MultilayerPerceptron::PartitionData(std::vector<Data> data, double trainingDataRatio, int batchSize)
    {
        if (trainingDataRatio < 0 || trainingDataRatio > 1)
            throw std::invalid_argument("size of partition must be between 0 and 1");

        if ((int)(data.size() * trainingDataRatio) < 1 || (int)(data.size() * (1 - trainingDataRatio)) < 1)
            throw std::invalid_argument("partition must not return lists of size 0");

        unsigned int bound = data.size() * trainingDataRatio;

        if (batchSize == 0)
            batchSize = bound;
            
        unsigned int batches = ceil((data.size() * trainingDataRatio) / (double) batchSize);

        std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());
        std::shuffle(std::begin(data), std::end(data), rng);

        std::vector<std::vector<Data>> trainingSets = std::vector<std::vector<Data>>(batches, std::vector<Data>());
        std::vector<Data> testingSet = {};

        for (unsigned int i = 0; i < data.size(); i++) {
            if (i < bound) {
                trainingSets[i / batchSize].push_back(data[i]);
            }
            else {
                testingSet.push_back(data[i]);
            }
        }

        return TrainTestPartition(trainingSets, testingSet);
    }
    
    std::tuple<double, double> MultilayerPerceptron::TestData(std::vector<Data> testingSet)
    {
        int correct = 0;
        int incorrect = 0;
        double cost = 0;
        for (auto data : testingSet) {
            LoadDataInstance(data);
            RunModel();
            if ((layers.back().neurons[0].value > 0 && data.label == 1) || (layers.back().neurons[0].value < 0 && data.label == -1))
                correct++;
            else
                incorrect++;

            cost += costFn->fn(data.label, layers.back().activationFn->fn(layers.back().neurons[0].value)) / testingSet.size();
        }

        double accuracy = (double)correct / (double)(correct + incorrect);

        return std::tuple<double, double>(accuracy, cost);
    }

    void MultilayerPerceptron::Train(std::vector<Data> data, int epochs, double learningRate, int batchSize)
    {
        for (int epoch = 0; epoch < epochs; epoch++) {
            std::cout << "Epoch " << epoch << std::endl;

            TrainTestPartition partition = PartitionData(data);

            std::vector<std::vector<Data>> trainingSets = partition.first;
            std::vector<Data> testingSet = partition.second;

            for (auto trainingSet : trainingSets) {
                LinearParams changes = GradientDescent(trainingSet, learningRate);

                for (unsigned int i = layers.size() - 2; i > 0; i--) {
                    changes = Backpropagate(changes, i);
                }
            }

            std::tuple<double, double> results = TestData(testingSet);
            double accuracy = std::get<0>(results);
            double cost = std::get<1>(results);

            std::cout << "Accuracy: " << accuracy << std::endl;
            std::cout << "Cost: " << cost << std::endl;
            std::cout << std::endl;
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
    //     std::vector<double> GradientDescent(std::vector<Data> batch);
    //     std::vector<double> Backpropagate();
    //     void Train(std::vector<Data> data, int epochs);
    // };
}