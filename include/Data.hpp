#pragma once
#include <vector>
#include <utility>
#include <string>

#include "Matrix.hpp"
#include "Vector.hpp"

struct Data
{
    using TrainTestPartition = std::pair<std::vector<Data>, std::vector<Data>>;

    Math::Matrix parameters;
    Math::Matrix label;

    /**
     * Size of parameter list
     */
    std::size_t parameterSize;

    /**
     * Size of label list
     */
    std::size_t labelSize;
    
    /**
     * Instances of data the struct stores
     */
    std::size_t dataInstanceCount;

    Data(std::vector<double> p_parameters, double p_labels);
    Data(std::vector<double> p_parameters, std::vector<double> p_label);
    Data(Math::Matrix p_parameters, Math::Matrix p_labels);
    Data(std::vector<Data> data);
        
    /**
     * Partitions data into a training and a testing set
     * @param data a vector of singular instances of data
     * @param trainingDataRatio the percentage of data meant for training examples
     * @returns a vector of training instances, and a vector of testing instances
     */
    static TrainTestPartition PartitionData(std::vector<Data> &data, double trainingDataRatio = 0.9);

    /**
     * Loads MNIST data
     * @returns a pair containing a vector of training data and a vector of testing data
    */
    static TrainTestPartition LoadMNIST(std::size_t maxDataSize = 10000);

private:
    static std::vector<Data> ReadMNISTFile(std::string pathname, std::size_t maxDataSize);
};