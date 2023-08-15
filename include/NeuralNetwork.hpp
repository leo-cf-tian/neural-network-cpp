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
    using LinearParams = std::tuple<std::vector<Math::Matrix>, std::vector<double>, std::vector<double>>;
    using TrainTestPartition = std::pair<std::vector<std::vector<Data>>, std::vector<Data>>;

    class MultilayerPerceptron
    {
    public:
        MultilayerPerceptron();
        MultilayerPerceptron(CostFn::CostFn* costFn);

        void AddLayer(Layer layer);
        void SetCostFunction(CostFn::CostFn* costFn);
        
        void Train(std::vector<Data> data, int epochs = 20, double learningRate = 0.01, int batchSize = 50);
        
    private:
        std::vector<Layer> layers;
        CostFn::CostFn* costFn;

        void LoadDataInstance(Data input);
        TrainTestPartition PartitionData(std::vector<Data> data, double trainingDataRatio = 0.9, int batchSize = 0);
        std::tuple<double, double> TestData(std::vector<Data> data);

        void RunModel();
        LinearParams GradientDescent(std::vector<Data> batch, double learningRate);
        LinearParams Backpropagate(LinearParams changes, std::size_t layerIndex);
    };
}