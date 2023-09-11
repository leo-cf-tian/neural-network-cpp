#include <cmath>
#include <iostream>

#include "CostFn.hpp"
#include "Data.hpp"

namespace CostFn
{
    CostFn::~CostFn() {};

    std::function<double(double)> CostFn::fn(double target)
    {
        return [this, target](double prediction) { return fn(prediction, target); };
    };

    std::function<double(double, double)> CostFn::fn()
    {
        return [this](double prediction, double target) { return fn(prediction, target); };
    };

    std::function<double(double)> CostFn::dx(double target)
    {
        return [this, target](double prediction) { return dx(prediction, target); };
    };

    std::function<double(double, double)> CostFn::dx()
    {
        return [this](double prediction, double target) { return dx(prediction, target); };
    };

    double CostFn::fn(double value, double target)
    {
        return (value - target) * (value - target);
    };

    double CostFn::dx(double value, double target)
    {
        return 2 * (value - target);
    };

    double CostFn::evaluate(const Math::Matrix &pred, const Math::Matrix &label)
    {
        int correct = 0;
        int incorrect = 0;

        for (unsigned int j = 0; j < pred.cols; j++) {
            bool truthy = true;

            for (unsigned int i = 0; i < pred.rows; i++) {
                truthy = truthy && abs(pred[i][j] - label[i][j]) < 0.01;
            }

            if (truthy)
                correct++;
            else
                incorrect++;
        }

        return (double) correct / (correct + incorrect);
    };

    Data CostFn::transformLabels(const Data &data, const NeuralNetwork::Layer &outputLayer)
    {
        return data;
    };

    double CrossEntropy::fn(double value, double target)
    {
        if (-std::log(1 - value) > 100)
            return - target * std::log(value) + (1 - target) * 100;
        else if (-std::log(value) > 100)
            return target * 10 - (1 - target) * std::log(1 - value);
        else
            return - target * std::log(value) - (1 - target) * std::log(1 - value);
    };

    double CrossEntropy::dx(double value, double target)
    {
        if (value < 0.000001)
            return (- target / 0.000001 + (1 - target) / (1 - value)) / 2.303;
        else if (1 - value < 0.000001)
            return (- target / value + (1 - target) / 0.000001) / 2.303;
        else
            return (- target / value + (1 - target) / (1 - value)) / 2.303;
    };

    double CrossEntropy::evaluate(const Math::Matrix &pred, const Math::Matrix &label)
    {
        int correct = 0;
        int incorrect = 0;

        for (unsigned int j = 0; j < pred.cols; j++) {
            bool truthy = true;

            for (unsigned int i = 0; i < pred.rows; i++) {
                truthy = truthy && round(pred[i][j]) == label[i][j];
            }

            if (truthy)
                correct++;
            else
                incorrect++;
        }

        return (double) correct / (correct + incorrect);
    };

    Data SparseCategoricalCrossEntropy::transformLabels(const Data &data, const NeuralNetwork::Layer &outputLayer)
    {
        if (data.label.rows != 1)
            throw std::invalid_argument("data must have scalar label");

        Math::Matrix label(outputLayer.neuronCount, data.dataInstanceCount);

        for (unsigned int i = 0; i < data.dataInstanceCount; i++) {
            if (data.label[0][i] != floor(data.label[0][i])) 
                throw std::invalid_argument("label for class must be an integer");

            if (data.label[0][i] < 0 || data.label[0][i] >= outputLayer.neuronCount) 
                throw std::invalid_argument("label for class is out of bounds");

            label.at(data.label[0][i], i) = 1;
        }

        return Data(data.parameters, label);
    };

    double SparseCategoricalCrossEntropy::evaluate(const Math::Matrix &pred, const Math::Matrix &label)
    {
        int correct = 0;
        int incorrect = 0;

        for (unsigned int j = 0; j < pred.cols; j++) {
            int maxIndex = 0;

            for (unsigned int i = 1; i < pred.rows; i++) {
                if (pred[i][j] > pred[maxIndex][j])
                {
                    maxIndex = i;
                }
            }

            if (maxIndex == label[0][j])
                correct++;
            else
                incorrect++;
        }

        return (double) correct / (correct + incorrect);
    };
}

