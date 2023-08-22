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
        
        void Train(std::vector<Data> &data, int epochs = 20, double learningRate = 0.01, int batchSize = 50);
        
    private:
        std::vector<Layer> layers;
        CostFn::CostFn* costFn;

        /**
         * @brief Loads an instance of data into the first layer of the matrix
         * @param fn fn(x, y): where x is the matrix member, and y is the argument matrix member
         * @param argMatrix Matrix of arguments corresponding to each member
         */
        void LoadDataInstance(Data &input);
        TrainTestPartition PartitionData(std::vector<Data> &data, double trainingDataRatio = 0.9, int batchSize = 0);
        std::tuple<double, double> TestData(std::vector<Data> &data);

        void RunModel();
        Math::Matrix GradientDescent(Data &batch, double learningRate);
        Math::Matrix Backpropagate(Math::Matrix &changes, std::size_t layerIndex, double learningRate);
    };
}