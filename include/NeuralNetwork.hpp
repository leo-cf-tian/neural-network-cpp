#pragma once
#include <tuple>
#include <utility>
#include <vector>

#include "CostFn.hpp"
#include "Data.hpp"
#include "Layer.hpp"
#include "Matrix.hpp"

namespace NeuralNetwork
{
    using TrainTestPartition = std::pair<std::vector<Data>, std::vector<Data>>;

    class MultilayerPerceptron
    {
    public:
        MultilayerPerceptron();
        MultilayerPerceptron(CostFn::CostFn* costFn);

        void AddLayer(Layer layer);
        void SetCostFunction(CostFn::CostFn* costFn);
        
        /**
         * @brief Trains the model on a set of data
         * @param data a vector of singular instances of data
         * @param epochs number of epochs for gradient descent
         * @param learningRate learning rate for gradient descent and backpropagation in this training session
         * @param batchSize the size of each training batch
         * @returns a matrix describing the derivative each neuron value in the previous layer relative to the cost
         */
        void Train(std::vector<Data> &data, int epochs = 20, double learningRate = 0.01, int batchSize = 50);
        
    private:
        std::vector<Layer> layers;
        CostFn::CostFn* costFn;

        /**
         * @brief Loads an instance of data into the first layer of the matrix
         * @param input an instance of data, can be multiple columns of different instances
         */
        void LoadDataInstance(Data &input);
        
        /**
         * @brief Partitions data into training batches and a testing set
         * @param data a vector of singular instances of data
         * @param trainingDataRatio the percentage of data meant for training examples
         * @param batchSize the size of each training batch
         * @returns a vector of batched training examples, and a vector of testing instances
         */
        TrainTestPartition PartitionData(std::vector<Data> &data, double trainingDataRatio = 0.9, int batchSize = 0);

        /**
         * @brief Takes a vector of data values and uses it to evaluate the model
         * @param data a vector of singular instances of data
         * @return a tuple describing accuracy and cost
         */
        std::tuple<double, double> TestData(std::vector<Data> &data);

        /**
         * @brief Calculates the layer values with data loaded into the first layer
        */
        void RunModel();

        /**
         * @brief Does gradient descent on single batch of data, updates the parameters of the output layer
         * @param batch an instance of data, can be multiple columns of different instances
         * @param learningRate learning rate for this particular instance of gradient descent
         * @returns a matrix describing the derivative each neuron value in the previous layer relative to the cost
         */
        Math::Matrix GradientDescent(Data &batch, double learningRate);

        /**
         * @brief Does gradient descent on single batch of data, updates the parameters of the output layer
         * @param changes a matrix describing the derivative each neuron value in the current layer relative to the cost
         * @param layerIndex a the index of the layer to perform backpropagation on
         * @param learningRate learning rate for this particular instance of backpropagation
         * @returns a matrix describing the derivative each neuron value in the previous layer relative to the cost
         */
        Math::Matrix Backpropagate(Math::Matrix &changes, std::size_t layerIndex, double learningRate);
    };
}