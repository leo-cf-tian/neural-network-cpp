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

    MultilayerPerceptron::~MultilayerPerceptron()
    {
        delete costFn;
    }

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
        
        Data transformedBatch = costFn->transformLabels(batch, layers.back());

        // dZ[n]
        // batch count divided here to prevent overflow
        Math::Matrix adjustmentMatrix = layer.valueMatrix.Apply(layer.activationFn->dx())
                                            & layer.Output().ApplyForEach(costFn->dx(), transformedBatch.label)
                                            / double(transformedBatch.dataInstanceCount);
        
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
    
    std::vector<Data> MultilayerPerceptron::BatchData(std::vector<Data> &data, int batchSize)
    {
        std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());
        std::shuffle(std::begin(data), std::end(data), rng);

        if (batchSize == 0)
            batchSize = data.size();
            
        unsigned int batches = ceil(data.size() / (double) batchSize);
            
        std::vector<std::vector<Data>> trainingSets = std::vector<std::vector<Data>>(batches, std::vector<Data>());

        for (unsigned int i = 0; i < data.size(); i++) {
            trainingSets[i / batchSize].push_back(data[i]);
        }

        std::vector<Data> trainingBatches = {};

        for (auto batch : trainingSets) {
            trainingBatches.push_back(Data(batch));
        }

        return trainingBatches;
    }
    
    std::tuple<double, double> MultilayerPerceptron::TestData(Data &data)
    {
        LoadDataInstance(data);
        RunModel();

        Data transformedData = costFn->transformLabels(data, layers.back());

        double accuracy = costFn->evaluate(layers.back().Output(), data.label);
        double cost = (double) (Math::Vector(layers.back().neuronCount, 1, false) * layers.back().Output().ApplyForEach(costFn->fn(), transformedData.label) * Math::Vector(transformedData.dataInstanceCount, 1, true)) / transformedData.dataInstanceCount;

        return std::tuple<double, double>(accuracy, cost);
    }

    void MultilayerPerceptron::Train(std::vector<Data> &trainingSet, std::vector<Data> &testingSet, int epochs, double learningRate, int batchSize)
    {
        Data trainingSetCache = Data(trainingSet);
        Data testingSetCache = Data(testingSet);

        for (int epoch = 0; epoch < epochs; epoch++) {
            std::cout << "Epoch " << epoch << std::endl;

            std::vector<Data> batches = BatchData(trainingSet, batchSize);

            for (auto batch : batches) {
                Math::Matrix changes = GradientDescent(batch, learningRate);

                for (unsigned int i = layers.size() - 2; i > 0; i--) {
                    changes = Backpropagate(changes, i, learningRate);
                }
            }

            std::tuple<double, double> results = TestData(trainingSetCache);
            double accuracy = std::get<0>(results);
            double cost = std::get<1>(results);
            std::cout << "Accuracy: " << accuracy << "\t\t";
            std::cout << "Cost: " << cost << std::endl;

            if (testingSet.size()) {
                std::tuple<double, double> valResults = TestData(testingSetCache);
                double valAccuracy = std::get<0>(valResults);
                double valCost = std::get<1>(valResults);
                std::cout << "Validation Accuracy: " << valAccuracy << "\t";
                std::cout << "Validation Cost: " << valCost << std::endl;
            }

            std::cout << std::endl;
        }
    }

    void MultilayerPerceptron::Train(std::vector<Data> &trainingSet, int epochs, double learningRate, int batchSize)
    {
        std::vector<Data> testingSet = {};
        Train(trainingSet, testingSet, epochs, learningRate, batchSize);
    }
}