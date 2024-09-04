#include "Linear.h"

namespace CNN
{
Linear::Linear(int input_size, int output_size)
: input_size_(input_size), output_size_(output_size)
{
    weights_.setZero(output_size_, input_size_);
    bias_.setZero(output_size_);
    is_loaded_ = false;
}

Linear::~Linear()
{
}

bool Linear::loadModel(const std::string &file_path, bool verbose)
{
    std::string weight_path = file_path + "/weight.txt";
    std::string bias_path = file_path + "/bias.txt";

    std::ifstream weight_file, bias_file;
    weight_file.open(weight_path, std::ios::in);
    bias_file.open(bias_path, std::ios::in);

    if(!weight_file.is_open())
    {
        std::cerr << "Can not find the file: " << weight_path << std::endl;
        return false;
    }
    if(!bias_file.is_open())
    {
        std::cerr << "Can not find the file: " << bias_path << std::endl;
        return false;
    }

    for (size_t i=0; i<weights_.rows(); i++) // out_channel
    {
        for(size_t j=0; j<weights_.cols(); j++)
        {
            weight_file >> weights_(i,j);
        }
    }
    weight_file.close();

    for (size_t i=0; i<bias_.size(); i++)
    {
        bias_file >> bias_(i);
    }
    bias_file.close();

    is_loaded_ = true;

    if(verbose)
    {
        std::cout << "Linear " << std::endl;
        std::cout << "\tWeight: " << std::endl;
        for(size_t i=0; i<weights_.cols(); i++)
        {
            std::cout << "\t\t" << weights_.row(i) << std::endl;
        }
        std::cout << "\tBias: " << std::endl;
        std::cout << bias_.transpose() << std::endl;
    }

    return true;
}

bool Linear::forward(const std::vector<VectorXd>& input, std::vector<VectorXd>& output)
{
    if(!is_loaded_)
    {
        std::cerr << "Model has not been loaded!" << std::endl;
        return false;
    }
    if(input.size() != 1)
    {
        std::cerr << "Input data is not flattend" << std::endl;
        return false;
    }
    if(input[0].size() != input_size_)
    {
        std::cerr << "Input size is not matched with " << input_size_ << " but " << input.size() << " is given!" << std::endl;
        return false;
    }
    output.resize(1);
    output[0] = weights_*input[0] + bias_;

    return true;
}

bool Linear::forward(std::vector<VectorXd>& input)
{
    std::vector<VectorXd> output;
    bool status = Linear::forward(input, output);
    if(status)
    {
        input = output;
    }
    return status;
}
}