#include <iostream>
#include <fstream>
#include <Eigen/Dense>

#include "config.h"
#include "utils.h"
#include "SequentialLayer.h"

using namespace CNN;

int main()
{
    Timer timer;
    // std::cout << model_path_ << std::endl;
    std::string input_path = model_path_+"input.txt";
    std::ifstream input_file;
    input_file.open(input_path, std::ios::in);
    if(!input_file.is_open())
    {
        std::cerr << "Can not find the file: " << input_path << std::endl;
        return 0;
    }

    int dof = 7;

    std::vector<Eigen::VectorXd> input;
    input.resize(1);
    for(size_t i=0; i<1; i++)
    {
        input[i].resize(dof);
        for(size_t j=0; j<dof; j++)
        {
            input_file >> input[i](j);
        }
    }
    input_file.close();
    
    std::cout << "Input: " << std::endl;
    for(size_t i=0; i<input.size(); i++)
    {
        std::cout << "\tChannel[" << i << "]:" << std::endl;
        std::cout << "\t\t" << input[i].transpose() << std::endl;
    }

    SequentialLayer conv1,conv2,conv3,conv4;
    conv1.addLayer(std::make_shared<Conv1dLayer>(1, 32, 3, 2, 1));
    conv1.addLayer(std::make_shared<BatchNorm1dLayer>(32));
    conv1.addLayer(std::make_shared<ReLULayer>());
    conv1.addLayer(std::make_shared<Conv1dLayer>(32, 32, 3, 1, 1));
    conv1.addLayer(std::make_shared<BatchNorm1dLayer>(32));
    conv1.addLayer(std::make_shared<ReLULayer>());

    conv2.addLayer(std::make_shared<Conv1dLayer>(32, 64, 3, 1, 1));
    conv2.addLayer(std::make_shared<BatchNorm1dLayer>(64));
    conv2.addLayer(std::make_shared<ReLULayer>());
    conv2.addLayer(std::make_shared<Conv1dLayer>(64, 64, 3, 1, 1));
    conv2.addLayer(std::make_shared<BatchNorm1dLayer>(64));

    conv3.addLayer(std::make_shared<Conv1dLayer>(64, 128, 3, 1, 1));
    conv3.addLayer(std::make_shared<BatchNorm1dLayer>(128));
    conv3.addLayer(std::make_shared<ReLULayer>());
    conv3.addLayer(std::make_shared<Conv1dLayer>(128, 128, 3, 1, 1));
    conv3.addLayer(std::make_shared<BatchNorm1dLayer>(128));

    conv4.addLayer(std::make_shared<LinearLayer>((dof+2)*128, 1024));
    conv4.addLayer(std::make_shared<BatchNorm1dLayer>(1024));
    conv4.addLayer(std::make_shared<ReLULayer>());
    conv4.addLayer(std::make_shared<LinearLayer>(1024, 1024));

    if(!conv1.loadModel(model_path_ + "conv1/")) return 0;
    if(!conv2.loadModel(model_path_ + "conv2/")) return 0;
    if(!conv3.loadModel(model_path_ + "conv3/")) return 0;
    if(!conv4.loadModel(model_path_ + "conv4/")) return 0;

    timer.reset();

    if(!conv1.forward(input)) return 0;
    std::cout << "Duration[sec]: " << timer.elapsedAndReset() << std::endl;
    if(!conv2.forward(input)) return 0;
    std::cout << "Duration[sec]: " << timer.elapsedAndReset() << std::endl;
    if(!conv3.forward(input)) return 0;
    std::cout << "Duration[sec]: " << timer.elapsedAndReset() << std::endl;
    flatten1d(input);
    std::cout << "Duration[sec]: " << timer.elapsedAndReset() << std::endl;
    if(!conv4.forward(input)) return 0;

    std::cout << "Duration[sec]: " << timer.elapsed() << std::endl;

    std::cout << "Output: " << std::endl;
    for(size_t i=0; i<input.size(); i++)
    {
        std::cout << "\tChannel[" << i << "]:" << std::endl;
        std::cout << "\t\t" << input[i].transpose() << std::endl;
    }

    return 0;
}