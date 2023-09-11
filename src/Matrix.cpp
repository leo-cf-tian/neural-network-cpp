#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

#include "Matrix.hpp"
#include "Vector.hpp"
#include "ThreadPool.hpp"

namespace Math
{
    ThreadPool Matrix::threadPool;
    
    void Matrix::UseThreadPool(std::function<void(unsigned int start, unsigned int end)> fn, int total)
    {
        std::condition_variable event;
        static std::mutex eventMutex;
        std::atomic<int> completedTasksCount(0);

        const int MAX_THREADS = Matrix::threadPool.poolSize();
        const int THREAD_NUM = std::min({total, MAX_THREADS});

        int block = total / THREAD_NUM;

        int start = 0;
        int end = block;

        for (int i = 0; i < THREAD_NUM; i++)
        {

            Matrix::threadPool.QueueTask(
                [start, end, &fn, &completedTasksCount, &event] {
                    fn(start, end);
                    {
                        std::unique_lock<std::mutex> lock{eventMutex};
                        completedTasksCount.fetch_add(1);
                        event.notify_one();
                    }
                }
            );
            start = end;
            end = std::min(end + block, total);
        }

        {
            std::unique_lock<std::mutex> lock{eventMutex};
            event.wait(lock, [&completedTasksCount, THREAD_NUM]
                        { return completedTasksCount == THREAD_NUM; });
            event.notify_all();
        }
    };

    Matrix::Matrix(std::size_t p_rows, std::size_t p_cols)
        : rows(p_rows), cols(p_cols)
    {
        if (rows < 1 || cols < 1) 
            throw std::invalid_argument("matrix dimensions must be positive");

        values = std::vector<double>(rows * cols, 0);
    };

    Matrix::Matrix(std::size_t p_rows, std::size_t p_cols, double value)
        : rows(p_rows), cols(p_cols)
    {
        if (rows < 1 || cols < 1) 
            throw std::invalid_argument("matrix dimensions must be positive");
        
        values = std::vector<double>(rows * cols, value);
    };


    Matrix::Matrix(std::size_t p_rows, std::size_t p_cols, matrix p_values)
        : rows(p_rows), cols(p_cols)
    {
        if (rows < 1 || cols < 1) 
            throw std::invalid_argument("matrix dimensions must be positive");

        if (p_values.size() != rows)
            throw std::invalid_argument("matrix row count does not match specified size");

        for (auto row : p_values) {
            if (row.size() != cols) {
                throw std::invalid_argument("matrix column count does not match specified size");
            }
        }
        
        values = std::vector<double>(rows * cols, 0);

        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                values[i * cols + j] = p_values[i][j];
            }
        }
    };

    Matrix::Matrix(matrix p_values)
    {
        rows = p_values.size();
        cols = p_values[0].size();

        if (rows < 1 || cols < 1) 
            throw std::invalid_argument("matrix dimensions must be positive");

        for (auto row : p_values) {
            if (row.size() != cols) {
                throw std::invalid_argument("matrix column count does not match specified size");
            }
        }
            
        values = std::vector<double>(rows * cols, 0);

        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                values[i * cols + j] = p_values[i][j];
            }
        }
    };

    Matrix::Matrix(std::size_t p_rows, std::size_t p_cols, std::vector<double> p_values)
        : rows(p_rows), cols(p_cols)
    {
        if (rows < 1 || cols < 1) 
            throw std::invalid_argument("matrix dimensions must be positive");

        if (p_values.size() != rows * cols) 
            throw std::invalid_argument("length of values array does not match matrix size");
            
        values = p_values;
    };

    Matrix Matrix::RandomMatrix(std::size_t rows, std::size_t cols, double min, double max)
    {
        srand(std::chrono::system_clock::now().time_since_epoch().count());

        std::vector<std::vector<double>> values = std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0));

        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                values[i][j] = rand() / static_cast<double>(RAND_MAX) * (max - min) + min;
            }
        }

        Math::Matrix result(values);

        return result;
    }

    Matrix Matrix::operator+(Matrix const &matrix) const
    {
        if (rows != matrix.rows || cols != matrix.cols)
            throw std::invalid_argument("matrices are not of the same size");

        Matrix result(rows, cols);

        for (unsigned int i = 0; i < rows * cols; i++) {
                result.values[i] = values[i] + matrix.values[i];
        }

        return result;
    };

    Matrix Matrix::operator+(Vector const &vector) const
    {
        if (rows != vector.size())
            throw std::invalid_argument("vector cannot be expanded to matrix of the same size");

        Matrix result(rows, cols);

        for (unsigned int i = 0; i < rows * cols; i++) {
                result.values[i] = values[i] + vector.values[i % vector.rows];
        }

        return result;
    };

    Matrix &Matrix::operator+=(Matrix const &matrix)
    {
        if (rows != matrix.rows || cols != matrix.cols)
            throw std::invalid_argument("matrices are not of the same size");

        for (unsigned int i = 0; i < rows * cols; i++) {
                values[i] += matrix.values[i];
        }
        
        return *this;
    };

    Matrix Matrix::operator-(Matrix const &matrix) const
    {
        if (rows != matrix.rows || cols != matrix.cols)
            throw std::invalid_argument("matrices are not of the same size");

        Matrix result(rows, cols);

        for (unsigned int i = 0; i < rows * cols; i++) {
                result.values[i] = values[i] - matrix.values[i];
        }

        return result;
    };

    Matrix Matrix::operator-(Vector const &vector) const
    {
        if (rows != vector.size())
            throw std::invalid_argument("vector cannot be expanded to matrix of the same size");

        Matrix result(rows, cols);

        for (unsigned int i = 0; i < rows * cols; i++) {
                result.values[i] = values[i] - vector.values[i % vector.rows];
        }

        return result;
    };

    Matrix &Matrix::operator-=(Matrix const &matrix)
    {
        if (rows != matrix.rows || cols != matrix.cols)
            throw std::invalid_argument("matrices are not of the same size");

        for (unsigned int i = 0; i < rows * cols; i++) {
            values[i] -= matrix.values[i];
        }
        
        return *this;
    };

    Matrix Matrix::operator-()
    {
        return Matrix(rows, cols, values) * -1.0;
    };

    Matrix operator*(const double &num, Matrix const &matrix)
    {
        Matrix result(matrix.rows, matrix.cols);

        for (unsigned int i = 0; i < matrix.rows; i++) {
            for (unsigned int j = 0; j < matrix.cols; j++) {
                result.values[i * matrix.cols + j] = matrix.values[i * matrix.cols + j] * num;
            }
        }

        return result;
    };

    Matrix Matrix::operator*(const double &num) const
    {
        Matrix result(rows, cols);

        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                result.values[i * cols + j] = values[i * cols + j] * num;
            }
        }

        return result;
    };

    Matrix &Matrix::operator*=(const double &num)
    {
        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                values[i * cols + j] *= num;
            }
        }

        return *this;
    };

    Matrix Matrix::operator/(const double &num) const
    {
        Matrix result(rows, cols);

        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                result.values[i * cols + j] = values[i * cols + j] / num;
            }
        }

        return result;
    };

    Matrix &Matrix::operator/=(const double &num)
    {
        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                values[i * cols + j] /= num;
            }
        }

        return *this;
    };

    Matrix Matrix::operator*(Matrix const &matrix) const
    {
        if (cols != matrix.rows)
            throw std::invalid_argument("left matrix column count and right matrix row count does not match");
    
        Matrix result(rows, matrix.cols);

        // Optimization of whether to multithread based on general benchmarking
        if (rows > 64 || matrix.cols > 64) {
            UseThreadPool(
                [&](unsigned int start, unsigned int end) {
                    for (unsigned int i = start; i < end; i++) {
                        for (unsigned int j = 0; j < matrix.cols; j++) {
                            for (unsigned int k = 0; k < matrix.rows; k++) {
                                result.values[i * matrix.cols + j] += values[i * cols + k] * matrix.values[k * matrix.cols + j];
                            }
                        }
                    }
                }
            , rows);
        }
        else {
            for (unsigned int i = 0; i < rows; i++) {
                        for (unsigned int j = 0; j < matrix.cols; j++) {
                            for (unsigned int k = 0; k < matrix.rows; k++) {
                                result.values[i * matrix.cols + j] += values[i * cols + k] * matrix.values[k * matrix.cols + j];
                            }
                        }
                    }
        }

        return result;
    };

    Matrix Matrix::operator&(Matrix const &matrix) const
    {
        if (rows != matrix.rows || cols != matrix.cols)
            throw std::invalid_argument("matrices are not of the same size");

        Matrix result(rows, cols);

        for (unsigned int i = 0; i < rows * cols; i++) {
                result.values[i] = values[i] * matrix.values[i];
        }

        return result;
    };

    std::vector<double> Matrix::operator[](std::size_t i) const
    {
        if (i >= rows || i < 0)
            throw std::invalid_argument("index out of range");

        return std::vector<double>(values.begin() + i * cols, values.begin() + (i + 1) * cols);
    };

    double &Matrix::at(std::size_t row, std::size_t col)
    {
        return values[row * cols + col];
    }

    double Matrix::at(std::size_t row, std::size_t col) const
    {
        return values[row * cols + col];
    }

    Matrix::operator std::vector<double>() const
    {
        if (rows != 1 && cols != 1)
            throw std::invalid_argument("only column or row matrices can be cast to vectors");

        return values;
    }

    Matrix::operator double() const
    {
        if (rows != 1 || cols != 1)
            throw std::invalid_argument("only matrices with a single value can be cast to doubles");

        return values[0];
    }

    Matrix Matrix::Transpose()
    {
        Matrix result(cols, rows);

        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                result.values[j * rows + i] = values[i * cols + j];
            }
        }

        return result;
    };

    Matrix Matrix::Apply(std::function<double(double)> fn)
    {
        Matrix result(rows, cols);

        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                result.values[i * cols + j] = fn(values[i * cols + j]);
            }
        }

        return result;
    }
    
    Matrix Matrix::ApplyForEach(std::function<double(double, double)> fn, Matrix argMatrix)
    {
        if (rows != argMatrix.rows || cols != argMatrix.cols)
            throw std::invalid_argument("argument matrix size not match matrix size");
        
        Matrix result(rows, cols);

        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                result.values[i * cols + j] = fn(values[i * cols + j], argMatrix.values[i * cols + j]);
            }
        }

        return result;
    };

    void Matrix::print() {
        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                std::cout << values[i * cols + j] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}
