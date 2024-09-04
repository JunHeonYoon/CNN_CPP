#ifndef CNN_ACTIVATION_H
#define CNN_ACTIVATION_H

#include <iostream>
#include <Eigen/Dense>
#include <vector>

namespace CNN
{
    using namespace Eigen;

    bool ReLU1d(std::vector<VectorXd>& input)
    {
        for(size_t i=0; i<input.size(); i++)
        {
            for(size_t j=0; j<input[i].size(); j++)
            {
                if(input[i](j) < 0)
                {
                    input[i](j) = 0;
                }
            }
        }
        return true;
    }
}

#endif