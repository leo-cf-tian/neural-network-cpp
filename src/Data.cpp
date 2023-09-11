#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <algorithm>

#include "Data.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"

Data::Data(std::vector<double> p_parameters, double p_label)
    : parameters(Math::Vector(p_parameters)), label(Math::Vector(1, p_label, false)), parameterSize(p_parameters.size()), labelSize(1), dataInstanceCount(1) {}

Data::Data(std::vector<double> p_parameters, std::vector<double> p_label)
    : parameters(Math::Vector(p_parameters)), label(Math::Vector(p_label)), parameterSize(p_parameters.size()), labelSize(p_label.size()), dataInstanceCount(1) {}

Data::Data(Math::Matrix p_parameters, Math::Matrix p_labels)
    : parameters(p_parameters), label(p_labels), parameterSize(p_parameters.cols), labelSize(p_labels.rows), dataInstanceCount(p_parameters.cols)
{
    if (parameters.cols != p_labels.cols)
        throw std::invalid_argument("number of labels does match number of rows");   
};

Data::Data(std::vector<Data> data)
    : parameters(Math::Matrix(1, 1)), label(Math::Vector(1)), parameterSize(data[0].parameters.rows), labelSize(data[0].label.rows), dataInstanceCount(data.size())
{
    if (data.size() == 0)
        throw std::invalid_argument("data vector cannot be empty");

    for (auto entry : data) {
        if (entry.parameters.rows != parameterSize)
            throw std::invalid_argument("data sizes do not match");

        if (entry.label.rows != labelSize)
            throw std::invalid_argument("label sizes do not match");

        if (entry.parameters.cols != 1 || entry.label.cols != 1)
            throw std::invalid_argument("only singular instances of data can be used to construct matrix");
    }

    parameters = Math::Matrix(parameterSize, data.size());
    label = Math::Matrix(labelSize, data.size());

    for (unsigned int i = 0; i < data.size(); i++) {
        for (unsigned int j = 0; j < data[i].parameters.rows; j++) {
            parameters.at(j, i) = data[i].parameters.at(j, 0);
        }

        for (unsigned int j = 0; j < data[i].label.rows; j++) {
            label.at(j, i) = data[i].label.at(j, 0);
        }
    }
};

Data::TrainTestPartition Data::PartitionData(std::vector<Data> &data, double trainingDataRatio)
{
    if (trainingDataRatio < 0 || trainingDataRatio > 1)
        throw std::invalid_argument("size of partition must be between 0 and 1");

    if ((int)(data.size() * trainingDataRatio) < 1 || (int)(data.size() * (1 - trainingDataRatio)) < 1)
        throw std::invalid_argument("partition must not return lists of size 0");

    unsigned int bound = data.size() * trainingDataRatio;

    std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());
    std::shuffle(std::begin(data), std::end(data), rng);

    std::vector<Data> trainingSet = {};
    std::vector<Data> testingSet = {};

    for (unsigned int i = 0; i < data.size(); i++) {
        if (i < bound) {
            trainingSet.push_back(data[i]);
        }
        else {
            testingSet.push_back(data[i]);
        }
    }

    return TrainTestPartition(trainingSet, testingSet);
}

Data::TrainTestPartition Data::LoadMNIST(std::size_t maxDataSize)
{
    std::cout << "Loading MNIST..." << std::endl;

    // reads from root folder
    std::vector<Data> trainingSet = ReadMNISTFile("data/MNIST/mnist_train.csv", maxDataSize);
    std::vector<Data> testingSet = ReadMNISTFile("data/MNIST/mnist_test.csv", maxDataSize);

    std::cout << "Loaded MNIST!" << std::endl;

    return TrainTestPartition(trainingSet, testingSet);
};

std::vector<Data> Data::ReadMNISTFile(std::string pathname, std::size_t maxDataSize)
{
    std::ifstream file(pathname);

    if (!file.is_open())
        throw std::runtime_error("error opening MNIST csv at \"" + pathname + "\"");

    std::string line;
    std::vector<Data> data;

    // throw out first line since they are just csv titles
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::vector<double> params;
        size_t pos = 0;
        std::string token;

        while ((pos = line.find(',')) != std::string::npos) {
            token = line.substr(0, pos);
            params.push_back(std::stoi(token));
            line.erase(0, pos + 1);
        }

        params.push_back(std::stoi(line));

        int label = params[0];
        params.erase(params.begin());

        for (double &num : params) {
            num /= 255;
        }

        data.push_back(Data(params, label));

        if (data.size() >= maxDataSize)
            break;
    }

    file.close();

    return data;
}