#pragma once
#include <vector>

#include "Matrix.hpp"
#include "Vector.hpp"

struct Data
{
    Math::Matrix parameters;
    Math::Vector label;

    /**
     * @brief Size of parameter list
     */
    std::size_t parameterSize;
    
    /**
     * @brief Instances of data the struct stores
     */
    std::size_t dataInstanceCount;

    Data(std::vector<double> p_parameters, double p_label);
    Data(Math::Matrix p_parameters, Math::Vector p_labels);
    Data(std::vector<Data> data);
};