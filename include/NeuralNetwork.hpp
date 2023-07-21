#pragma once
#include <tuple>
#include <vector>

#include "CostFn.hpp"
#include "Data.hpp"
#include "Layer.hpp"

namespace NeuralNetwork
{
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
        void RunModel();
        std::tuple<std::vector<Math::Matrix>, std::vector<float>> GradientDescent(std::vector<Data> batch);
        std::tuple<std::vector<Math::Matrix>, std::vector<float>> Backpropagate();
    };
}