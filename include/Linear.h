#ifndef CNN_LINEAR_H
#define CNN_LINEAR_H

#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <fstream>
#include <vector>

namespace CNN
{
    using namespace Eigen;

    class Linear
    {
        public:
            Linear(int input_size, int output_size);
            ~Linear();
            bool loadModel(const std::string& file_path, bool verbose=false);
            bool forward(const std::vector<VectorXd>& input, std::vector<VectorXd>& output);
            bool forward(std::vector<VectorXd>& input);

        private:
            unsigned int input_size_, output_size_;
            MatrixXd weights_; // shape: [output, input]
            VectorXd bias_;    // shape: [output]

            bool is_loaded_;
    };
}

#endif
