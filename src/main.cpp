#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <chrono>

#include "Matrix.hpp"

using namespace std::chrono;

using namespace std;

Math::Matrix RandomMatrix(unsigned int rows, unsigned int cols)
{
    vector<vector<float>> values = vector<vector<float>>(rows, std::vector<float>(cols, 0));

    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < cols; j++) {
            values[i][j] = rand() / static_cast<float>(RAND_MAX) * 200 - 100;
        }
    }

    Math::Matrix result(values);

    return result;
}

int main(void) {
    srand(time(NULL));

    auto start = high_resolution_clock::now();

    Math::Matrix m1 = RandomMatrix(512, 512);
    Math::Matrix m2 = RandomMatrix(512, 512);

    Math::Matrix m3 = m1 * m2;
    
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);
 
    cout << duration.count() << endl;

    // for (unsigned int i = 0; i < m1.rows; i++) {
    //     for (unsigned int j = 0; j < m1.cols; j++) {
    //         cout << m1[i][j] << ",";
    //     }
    //     cout << endl;
    // }
    
    // cout << endl;

    // for (unsigned int i = 0; i < m2.rows; i++) {
    //     for (unsigned int j = 0; j < m2.cols; j++) {
    //         cout << m2[i][j] << ",";
    //     }
    //     cout << endl;
    // }
    
    // cout << endl;

    // for (unsigned int i = 0; i < m3.rows; i++) {
    //     for (unsigned int j = 0; j < m3.cols; j++) {
    //         cout << m3[i][j] << ",";
    //     }
    //     cout << endl;
    // }

}