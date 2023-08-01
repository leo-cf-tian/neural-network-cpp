#pragma once
#include <vector>

struct Data
{
    Data(std::vector<float> parameters, int label);

    std::vector<float> parameters;
    int label;
};