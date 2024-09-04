#ifndef CNN_UTILS_H
#define CNN_UTILS_H

#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <chrono>

class Timer
{
    public:
    Timer() : beg_(hd_clock::now()) {}
    void reset() { beg_ = hd_clock::now(); }
    double elapsed() const 
    { 
        return std::chrono::duration_cast<second>(hd_clock::now() - beg_).count(); 
    }
    double elapsedAndReset() 
    { 
        double e = elapsed(); 
        reset();
        return e;
    }

    private:
    typedef std::chrono::high_resolution_clock hd_clock;
    typedef std::chrono::duration<double, std::ratio<1> > second;
    std::chrono::time_point<hd_clock> beg_;
};

namespace CNN
{
    bool flatten1d(std::vector<Eigen::VectorXd>& input)
    {
        std::vector<Eigen::VectorXd> flatten_input;
        flatten_input.resize(1);
        int flatten_input_size = input.size() * input[0].size();
        flatten_input[0].setZero(flatten_input_size);
        for(size_t i=0; i<input.size(); i++)
        {
            flatten_input[0].segment(i*input[i].size(), input[i].size()) = input[i];
        }
        input = flatten_input;

        return true;
    }
}

#endif