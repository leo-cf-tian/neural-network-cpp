#pragma once
#include <vector>

struct Data
{
    Data(std::vector<double> parameters, int label);

    std::vector<double> parameters;
    int label;
};