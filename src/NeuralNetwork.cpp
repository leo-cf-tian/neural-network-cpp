#include <iostream>
#include <tuple>
#include <vector>

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

    std::tuple<std::vector<Math::Matrix>, std::vector<float>> MultilayerPerceptron::GradientDescent(std::vector<Data> batch)
    {
        std::vector<Math::Matrix> weightDerivatives = std::vector<Math::Matrix>(layers.back().neurons.size(), Math::Matrix(layers.back().connectionCount, 1));
        std::vector<float> biasDerivatives = std::vector<float>(layers.back().neurons.size(), 0);
        
        for (auto data : batch) {
            for (unsigned int i = 0; i < layers.back().neurons.size(); i++) {
                NeuralNetwork::Neuron neuron = layers.back().neurons[i];
                weightDerivatives[i] += layers.back().activationFn->dx(neuron.value)
                                        * costFn->dx(data.label, layers.back().activationFn->fn(neuron.value))
                                        * Math::Matrix::ColumnMatrix(layers[layers.size() - 2].OutputVector())
                                        / batch.size();
                biasDerivatives[i] += layers.back().activationFn->dx(neuron.value)
                                        * costFn->dx(data.label, layers.back().activationFn->fn(neuron.value))
                                        / batch.size();
            }
        }

        for (unsigned int i = 0; i < layers.back().neurons.size(); i++) {
            layers.back().neurons[i].Adjust(weightDerivatives[i], biasDerivatives[i]);
        }

        return std::tuple(weightDerivatives, biasDerivatives);
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