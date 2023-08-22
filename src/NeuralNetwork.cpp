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
        
        layer.InitializeConnections(layers.back().neuronCount);
        layers.push_back(layer);
    }

    void MultilayerPerceptron::SetCostFunction(CostFn::CostFn* p_costFn)
    {
        costFn = p_costFn;
    }

    void MultilayerPerceptron::LoadDataInstance(Data &input)
    {
        if (layers.size() < 1)
            throw std::invalid_argument("neural network layers are not defined");

        if (input.parameterSize != layers[0].neuronCount) 
            throw std::invalid_argument("input dimensions do not match specified dimensions");

        layers[0].valueMatrix = input.parameters;
    }

    void MultilayerPerceptron::RunModel()
    {
        Math::Matrix output = layers[0].Output();

        for (unsigned int i = 1; i < layers.size(); i++) {
            output = layers[i].CalculateValues(output);
        }
    }

    Math::Matrix MultilayerPerceptron::GradientDescent(Data &batch, double learningRate)
    {
        Layer &layer = layers.back();
        
        LoadDataInstance(batch);
        RunModel();

        // dZ[n]
        // batch count divided here to prevent overflow
        Math::Matrix adjustmentMatrix = layer.valueMatrix.Apply(layer.activationFn->dx())
                                            & layer.Output().ApplyForEach(costFn->dx(), batch.label)
                                            / double(batch.dataInstanceCount);
        
        // dW[n]
        Math::Matrix weightDerivatives = adjustmentMatrix * layers[layers.size() - 2].Output().Transpose();
        
        // db[n]
        // vector multiplication sums each row
        Math::Vector biasDerivatives = adjustmentMatrix * Math::Vector(adjustmentMatrix.cols, 1, true);
        
        // dA[n-1]
        Math::Matrix prevValueDerivatives = layer.weightMatrix.Transpose() * adjustmentMatrix;
        
        layer.AdjustNeurons(-weightDerivatives, -biasDerivatives, learningRate);

        return prevValueDerivatives;
    }

    Math::Matrix MultilayerPerceptron::Backpropagate(Math::Matrix &changes, std::size_t layerIndex, double learningRate)
    {
        Layer &layer = layers[layerIndex];

        Math::Matrix adjustmentMatrix = layer.valueMatrix.Apply(layer.activationFn->dx())
                                            & changes;

        Math::Matrix weightDerivatives = adjustmentMatrix * layers[layerIndex - 1].Output().Transpose();
        Math::Vector biasDerivatives = adjustmentMatrix * Math::Vector(adjustmentMatrix.cols, 1, true);
        Math::Matrix prevValueDerivatives = layer.weightMatrix.Transpose() * adjustmentMatrix;
        
        layer.AdjustNeurons(-weightDerivatives, -biasDerivatives, learningRate);

        return prevValueDerivatives;
    }
    
    TrainTestPartition MultilayerPerceptron::PartitionData(std::vector<Data> &data, double trainingDataRatio, int batchSize)
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

        std::vector<Data> trainingBatches = {};

        for (auto batch : trainingSets) {
            trainingBatches.push_back(Data(batch));
        }

        return TrainTestPartition(trainingBatches, testingSet);
    }
    
    std::tuple<double, double> MultilayerPerceptron::TestData(std::vector<Data> &testingSet)
    {
        int correct = 0;
        int incorrect = 0;
        double cost = 0;
        for (auto data : testingSet) {
            LoadDataInstance(data);
            RunModel();
            if ((layers.back().valueMatrix[0][0] > 0 && data.label[0] == 1) || (layers.back().valueMatrix[0][0] < 0 && data.label[0] == -1))
                correct++;
            else
                incorrect++;

            cost += costFn->fn(data.label[0], layers.back().activationFn->fn(layers.back().valueMatrix[0][0])) / testingSet.size();
        }

        double accuracy = (double)correct / (double)(correct + incorrect);

        return std::tuple<double, double>(accuracy, cost);
    }

    void MultilayerPerceptron::Train(std::vector<Data> &data, int epochs, double learningRate, int batchSize)
    {
        for (int epoch = 0; epoch < epochs; epoch++) {
            std::cout << "Epoch " << epoch << std::endl;

            TrainTestPartition partition = PartitionData(data);

            std::vector<Data> trainingSets = partition.first;
            std::vector<Data> testingSet = partition.second;

            for (auto trainingSet : trainingSets) {
                Math::Matrix changes = GradientDescent(trainingSet, learningRate);

                for (unsigned int i = layers.size() - 2; i > 0; i--) {
                    changes = Backpropagate(changes, i, learningRate);
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