#pragma once
#include <functional>

#include "Matrix.hpp"
#include "Layer.hpp"
#include "Data.hpp"

namespace CostFn
{
    // Uses L2 as placeholder virtual functions to stop compiler from screaming
    class CostFn
    {
    public:
        virtual ~CostFn();

        virtual double fn(double value, double target);
        std::function<double(double)> fn(double target);
        std::function<double(double, double)> fn();
        virtual double dx(double value, double target);
        std::function<double(double)> dx(double target);
        std::function<double(double, double)> dx();

        /**
         * evaluate whether output is correctly predicted
         * @param pred predicted label as an output vector
         * @param label real label
         * @return whether prediction is correct
        */
        virtual double evaluate(const Math::Matrix &pred, const Math::Matrix &label);
        
        /**
         * transforms labels of a dataset to fit the output space, with reference to the output layer
         * @param data original data to transform
         * @param outputLayer reference to the output layer
         * @return data with transformed labels
        */
        virtual Data transformLabels(const Data &data, const NeuralNetwork::Layer &outputLayer);
    };

    class L2 : public CostFn
    {
    public:
        double fn(double value, double target) override;
        double dx(double value, double target) override;
    };

    // A.K.A. Log Loss
    class CrossEntropy : public CostFn
    {
    public:
        double fn(double value, double target) override;
        double dx(double value, double target) override;
        double evaluate(const Math::Matrix &pred, const Math::Matrix &label) override;
    };

    // A.K.A. Log Loss for Multiclass Classification
    class SparseCategoricalCrossEntropy : public CrossEntropy
    {
    public:
        double evaluate(const Math::Matrix &pred, const Math::Matrix &label) override;
        virtual Data transformLabels(const Data &data, const NeuralNetwork::Layer &outputLayer) override;
    };
}
