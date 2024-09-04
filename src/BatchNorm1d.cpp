#include "BatchNorm1d.h"

namespace CNN{

BatchNorm1d::BatchNorm1d(int channels)
: channels_(channels)
{
    is_loaded_ = false;
    weights_.resize(channels_);
    bias_.resize(channels_);
    weights_vec_.setZero(channels);
    bias_vec_.setZero(channels);
}

BatchNorm1d::~BatchNorm1d()
{
}

bool BatchNorm1d::loadModel(const std::string &file_path, bool verbose)
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

    for (size_t i=0; i<weights_.size(); i++)
    {
        weight_file >> weights_[i];
        weights_vec_(i) = weights_[i];
    }
    weight_file.close();

    for (size_t i=0; i<bias_.size(); i++)
    {
        bias_file >> bias_[i];
        bias_vec_(i) = bias_[i];
    }
    bias_file.close();

    is_loaded_ = true;

    if(verbose)
    {
        std::cout << "BatchNorm1d " << std::endl;
        std::cout << "\tWeight: " << std::endl;
        for(size_t i=0; i<weights_.size(); i++)
        {
            std::cout << "\t\tOut channel[" << i << "]:" << std::endl;
            std::cout << "\t\t\t" << weights_[i] << std::endl;
        }
        std::cout << "\tBias: " << std::endl;
        for(size_t i=0; i<bias_.size(); i++)
        {
            std::cout << "\t\tOut channel[" << i << "]:" << std::endl;
            std::cout << "\t\t\t" << bias_[i] << std::endl;
        }
    }

    return true;
}

bool BatchNorm1d::forward(const std::vector<VectorXd>& input, std::vector<VectorXd>& output)
{
    if(!is_loaded_)
    {
        std::cerr << "Model has not been loaded!" << std::endl;
        return false;
    }
    if(input.size() == 1) // flattened data
    {
        if(input[0].size() != channels_)
        {
            std::cerr << "Flattend input size is not matched with " << channels_ << " but " << input[0].size() << " is given!" << std::endl;
            return false;
        }
        output.resize(1);
        output[0] = weights_vec_.array()*input[0].array() + bias_vec_.array();
    }
    else
    {
        if(input.size() != channels_)
        {
            std::cerr << "Input channel is not matched with " << channels_ << " but " << input.size() << " is given!" << std::endl;
            return false;
        }

        output.resize(channels_);

        for (size_t i=0; i<channels_; ++i)
        {   
            output[i] = input[i].array() * weights_[i] + bias_[i];
        }
    }

    return true;
}

bool BatchNorm1d::forward(std::vector<VectorXd>& input)
{
    std::vector<VectorXd> output;
    bool status = BatchNorm1d::forward(input, output);
    if(status)
    {
        input = output;
    }
    return status;
}
}
