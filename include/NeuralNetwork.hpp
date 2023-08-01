#pragma once
#include <tuple>
#include <utility>
#include <vector>

#include "CostFn.hpp"
#include "Data.hpp"
#include "Layer.hpp"

namespace NeuralNetwork
{
    using LinearParams = std::pair<std::vector<std::vector<float>>, std::vector<float>>;
    using TrainTestPartition = std::pair<std::vector<Data>, std::vector<Data>>;

    class MultilayerPerceptron
    {
    public:
        MultilayerPerceptron();
        MultilayerPerceptron(CostFn::CostFn* costFn);

        void AddLayer(Layer layer);
        void SetCostFunction(CostFn::CostFn* costFn);
        
        void Train(std::vector<Data> data, int epochs);
        
    private:
        std::vector<Layer> layers;
        CostFn::CostFn* costFn;

        void LoadDataInstance(Data input);
        TrainTestPartition PartitionData(std::vector<Data> data, float trainingDataRatio);
        std::tuple<float> TestData(std::vector<Data> data);

        void RunModel();
        LinearParams GradientDescent(std::vector<Data> batch, float learningRate);
        LinearParams Backpropagate(LinearParams changes);
    };
}