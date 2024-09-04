#include "Conv1d.h"

namespace CNN{

Conv1d::Conv1d(int in_channels, int out_channels, int kernel_size, int padding, int stride)
: in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), padding_(padding), stride_(stride)
{
    is_loaded_ = false;
    weights_.resize(out_channels_);
    for(size_t i=0; i<weights_.size(); i++)
    {
        weights_[i].resize(in_channels_);
        for(size_t j=0; j<weights_[i].size(); j++)
        {
            weights_[i][j].setZero(kernel_size_);
        }
    }
    bias_.resize(out_channels_);
}

Conv1d::~Conv1d()
{
}

bool Conv1d::loadModel(const std::string &file_path, bool verbose)
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

    for (size_t i=0; i<weights_.size(); i++) // out_channel
    {
        for (size_t j=0; j<weights_[i].size(); j++) // in_channel
        {
            for(size_t k=0; k<weights_[i][j].size(); k++) // kernel
            {
                weight_file >> weights_[i][j](k);
            }
        }
    }
    weight_file.close();

    for (size_t i=0; i<bias_.size(); i++)
    {
        bias_file >> bias_[i];
    }
    bias_file.close();

    is_loaded_ = true;

    if(verbose)
    {
        std::cout << "Conv1d " << std::endl;
        std::cout << "\tWeight: " << std::endl;
        for(size_t i=0; i<weights_.size(); i++)
        {
            for(size_t j=0; j<weights_[i].size(); j++)
            {
                std::cout << "\t\tOut channel[" << i << "], In channel[" << j <<"], kernel:" << std::endl;
                std::cout << "\t\t\t" << weights_[i][j].transpose() << std::endl;
            }
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

VectorXd Conv1d::ConvSingle(const VectorXd& input, const VectorXd& kernel)
{
    int input_size = input.size();
    int kernel_size = kernel.size();

    // Add padding to the input
    VectorXd padded_input = VectorXd::Zero(input_size + 2*padding_);
    padded_input.segment(padding_, input_size) = input;

    int output_size = int(floor((input_size - kernel_size_ + 2*padding_) / stride_)) + 1;

    VectorXd output(output_size);

    for (size_t i=0; i<output_size; ++i) 
    {
        output(i) = padded_input.segment(i * stride_, kernel_size).dot(kernel);
    }                                                                                                                           

    return output;
}

bool Conv1d::forward(const std::vector<VectorXd>& input, std::vector<VectorXd>& output)
{
    if(!is_loaded_)
    {
        std::cerr << "Model has not been loaded!" << std::endl;
        return false;
    }
    if(input.size() != in_channels_)
    {
        std::cerr << "Input channel is not matched with " << in_channels_ << " but " << input.size() << " is given!" << std::endl;
        return false;
    }

    // Output initialize
    unsigned int input_size = input[0].size();
    unsigned int output_size = int(floor((input_size - kernel_size_ + 2*padding_) / stride_)) + 1;

    output.resize(out_channels_);

    for (size_t i=0; i<out_channels_; ++i)
    {   
        VectorXd output_single;
        output_single.setZero(output_size);
        for (int j = 0; j < in_channels_; ++j)
        {   
            output_single += Conv1d::ConvSingle(input[j], weights_[i][j]);
        }

        output[i] = output_single.array() + bias_[i];
    }
    return true;
}

bool Conv1d::forward(std::vector<VectorXd>& input)
{
    std::vector<VectorXd> output;
    bool status = Conv1d::forward(input, output);
    if(status)
    {
        input = output;
    }
    return status;
}
}
