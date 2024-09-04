#ifndef CNN_LAYER_H
#define CNN_LAYER_H

#include <Eigen/Dense>
#include <vector>
#include "Conv1d.h"
#include "BatchNorm1d.h"
#include "Linear.h"
#include "Activation.h"

namespace CNN 
{

    class Layer 
    {
        public:
            virtual ~Layer() {}
            virtual bool forward(std::vector<Eigen::VectorXd>& input) = 0;
    };

    class Conv1dLayer : public Layer 
    {
        public:
            Conv1dLayer(int in_channels, int out_channels, int kernel_size, int padding, int stride)
                : conv1d_(in_channels, out_channels, kernel_size, padding, stride) {}

            bool loadModel(const std::string& path) 
            {
                bool status = conv1d_.loadModel(path);
                return status;
            }

            bool forward(std::vector<Eigen::VectorXd>& input) override 
            {
                bool status = conv1d_.forward(input);
                return status;
            }

        private:
            Conv1d conv1d_;
    };

    class BatchNorm1dLayer : public Layer 
    {
        public:
            BatchNorm1dLayer(int num_features) : bn1d_(num_features) {}

            bool loadModel(const std::string& path) 
            {
                bool status = bn1d_.loadModel(path);
                return status;
            }

            bool forward(std::vector<Eigen::VectorXd>& input) override 
            {
                bool status = bn1d_.forward(input);
                return status;
            }

        private:
            BatchNorm1d bn1d_;
    };

    class LinearLayer : public Layer 
    {
        public:
            LinearLayer(int input_size, int output_size) : linear_(input_size, output_size) {}

            bool loadModel(const std::string& path) 
            {
                bool status = linear_.loadModel(path);
                return status;
            }

            bool forward(std::vector<Eigen::VectorXd>& input) override 
            {
                bool status = linear_.forward(input);
                return status;
            }

        private:
            Linear linear_;
    };

    class ReLULayer : public Layer 
    {
        public:
            bool forward(std::vector<Eigen::VectorXd>& input) override 
            {
                bool status = ReLU1d(input);
                return status;
            }
    };
}

#endif
