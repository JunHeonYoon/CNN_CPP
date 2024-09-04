#ifndef CNN_SEQUENTIAL_LAYER_H
#define CNN_SEQUENTIAL_LAYER_H

#include "Layer.h"
#include <vector>
#include <memory>

namespace CNN 
{

class SequentialLayer : public Layer 
{
public:
    SequentialLayer() = default;

    void addLayer(std::shared_ptr<Layer> layer) 
    {
        layers_.push_back(layer);
    }

    bool forward(std::vector<Eigen::VectorXd>& input) override 
    {
        for (auto& layer : layers_) 
        {
            bool status = layer->forward(input);
            if(!status) return false;
        }
        return true;
    }

    bool loadModel(const std::string& base_path) 
    {
        for (size_t i = 0; i < layers_.size(); ++i) 
        {
            auto conv1dLayer = std::dynamic_pointer_cast<Conv1dLayer>(layers_[i]);
            if (conv1dLayer) 
            {
                bool status = conv1dLayer->loadModel(base_path + std::to_string(i + 1) + "_conv1d");
                if(!status) return false;
            }

            auto bn1dLayer = std::dynamic_pointer_cast<BatchNorm1dLayer>(layers_[i]);
            if (bn1dLayer) 
            {
                bool status = bn1dLayer->loadModel(base_path + std::to_string(i + 1) + "_batchnorm1d");
                if(!status) return false;
            }

            auto linearLayer = std::dynamic_pointer_cast<LinearLayer>(layers_[i]);
            if (linearLayer) 
            {
                bool status = linearLayer->loadModel(base_path + std::to_string(i + 1) + "_linear");
                if(!status) return false;
            }
        }
        return true;
    }

private:
    std::vector<std::shared_ptr<Layer>> layers_;
};

}


#endif