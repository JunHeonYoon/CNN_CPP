#ifndef CNN_CONV1D_H
#define CNN_CONV1D_H

#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <fstream>
#include <vector>

namespace CNN
{
    using namespace Eigen;

    class Conv1d
    {
        public:
            /// @brief Convolution 1D layer
            /// @param in_channels (int) size of input channel
            /// @param out_channels (int) size of output channel
            /// @param kernel_size (int) size of kernel
            /// @param padding (int) size of padding
            /// @param stride (int) size of stride
            Conv1d(int in_channels, int out_channels, int kernel_size, int padding, int stride);

            ~Conv1d();

            /// @brief Load pre-trained Conv1d layer
            /// @param file_path (std::string) path to parameter file folder - weight.txt, bias.txt
            /// @param verbose (bool) whether to see loaded model parameter
            /// @return (bool) status
            bool loadModel(const std::string& file_path, bool verbose=false);

            /// @brief forward of Conv1d
            /// @param input (std::vector<Eigen::VectorXd>) input 
            /// @param output (std::vector<Eigen::VectorXd>) output
            /// @return (bool) status
            bool forward(const std::vector<VectorXd>& input, std::vector<VectorXd>& output);

            /// @brief forward of Conv1d
            /// @param input (std::vector<Eigen::VectorXd>) input 
            /// @return (bool) status
            bool forward(std::vector<VectorXd>& input);

        private:
            VectorXd ConvSingle(const Eigen::VectorXd& input, const Eigen::VectorXd& kernel);
            unsigned int in_channels_, out_channels_, kernel_size_, padding_, stride_;
            std::vector<std::vector<VectorXd>> weights_; // shape: [out_channel, in_channel, kernel_size]
            std::vector<double> bias_;                   // shape: [out_channel]

            bool is_loaded_;

    };
}

#endif