#include <vector>
#include <iostream>

#include "Data.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"

Data::Data(std::vector<double> p_parameters, double p_label)
    : parameters(Math::Vector(p_parameters)), label(Math::Vector(1, p_label, false)), parameterSize(p_parameters.size()), dataInstanceCount(1) {}

Data::Data(Math::Matrix p_parameters, Math::Vector p_labels)
    : parameters(p_parameters), label(p_labels)
{
    if (parameters.cols != p_labels.cols)
        throw std::invalid_argument("number of labels does match number of rows");

    if (p_labels.rows != 1)
        throw std::invalid_argument("number of rows for labels should be 1");    
};

Data::Data(std::vector<Data> data)
    : parameters(Math::Matrix(1, 1)), label(Math::Vector(1)), parameterSize(data[0].parameters.rows), dataInstanceCount(data.size())
{
    if (data.size() == 0)
        throw std::invalid_argument("data vector cannot be empty");

    for (auto entry : data) {
        if (entry.parameters.rows != parameterSize)
            throw std::invalid_argument("data sizes do not match");

        if (entry.parameters.cols != 1 || entry.label.cols != 1)
            throw std::invalid_argument("only singular instances of data can be used to construct matrix");
    }

    parameters = Math::Matrix(data[0].parameters.rows, data.size());
    label = Math::Vector(data.size(), false);


    for (unsigned int i = 0; i < data.size(); i++) {
        for (unsigned int j = 0; j < data[i].parameters.rows; j++) {
            parameters.at(j, i) = data[i].parameters.at(j, 0);
        }

        label[i] = data[i].label;
    }
};