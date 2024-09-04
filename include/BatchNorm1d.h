#ifndef CNN_BATCHNORM1D_H
#define CNN_BATCHNORM1D_H

#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <fstream>
#include <vector>

namespace CNN
{
    using namespace Eigen;

    class BatchNorm1d
    {
        public:
            BatchNorm1d(int channels);
            ~BatchNorm1d();
            bool loadModel(const std::string& file_path, bool verbose=false);
            bool forward(const std::vector<VectorXd>& input, std::vector<VectorXd>& output);
            bool forward(std::vector<VectorXd>& input);

        private:
            unsigned int channels_;
            std::vector<double> weights_; // shape: [channel]
            std::vector<double> bias_;    // shape: [channel]
            VectorXd weights_vec_;        // shape: [channel]
            VectorXd bias_vec_;           // shape: [channel]

            bool is_loaded_;

    };
}

#endif