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
    class MultilayerPerceptron
    {
    public:
        MultilayerPerceptron();
        MultilayerPerceptron(CostFn::CostFn* costFn);
        ~MultilayerPerceptron();

        void AddLayer(Layer layer);
        void SetCostFunction(CostFn::CostFn* costFn);
        
        /**
         * Trains the model on a set of data
         * @param trainingSet a vector of singular instances of data used to train the model
         * @param testingSet a vector of singular instances of data used to test the model
         * @param epochs number of epochs for gradient descent
         * @param learningRate learning rate for gradient descent and backpropagation in this training session
         * @param batchSize the size of each training batch
         * @returns a matrix describing the derivative each neuron value in the previous layer relative to the cost
         */
        void Train(std::vector<Data> &trainingSet, std::vector<Data> &testingSet, int epochs = 20, double learningRate = 0.01, int batchSize = 0);
        void Train(std::vector<Data> &trainingSet, int epochs = 20, double learningRate = 0.01, int batchSize = 0);

    private:
        std::vector<Layer> layers;
        CostFn::CostFn* costFn;

        /**
         * Loads an instance of data into the first layer of the matrix
         * @param input an instance of data, can be multiple columns of different instances
         */
        void LoadDataInstance(Data &input);

        /**
         * Partitions training data into batches
         * @param data a vector of singular instances of data
         * @param batchSize the size of each training batch (0 for no batching)
         * @returns a vector of batched training examples
         */
        std::vector<Data> BatchData(std::vector<Data> &data, int batchSize = 0);

        /**
         * Takes a vector of data values and uses it to evaluate the model
         * @param data a vector of singular instances of data
         * @return a tuple describing accuracy and cost
         */
        std::tuple<double, double> TestData(Data &data);

        /**
         * Calculates the layer values with data loaded into the first layer
         * 
         * Run `LoadDataInstance` first
        */
        void RunModel();

        /**
         * Does gradient descent on single batch of data, updates the parameters of the output layer
         * @param batch an instance of data, can be multiple columns of different instances
         * @param learningRate learning rate for this particular instance of gradient descent
         * @returns a matrix describing the derivative each neuron value in the previous layer relative to the cost
         */
        Math::Matrix GradientDescent(Data &batch, double learningRate);

        /**
         * Does gradient descent on single batch of data, updates the parameters of the output layer
         * @param changes a matrix describing the derivative each neuron value in the current layer relative to the cost
         * @param layerIndex a the index of the layer to perform backpropagation on
         * @param learningRate learning rate for this particular instance of backpropagation
         * @returns a matrix describing the derivative each neuron value in the previous layer relative to the cost
         */
        Math::Matrix Backpropagate(Math::Matrix &changes, std::size_t layerIndex, double learningRate);
    };
}