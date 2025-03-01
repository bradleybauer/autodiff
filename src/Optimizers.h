#pragma once
#include <vector>
#include <cmath>
#include "Tensor.h"  // Make sure this header defines the Tensor class and its operators

namespace xtorch {
using std::vector;

class SGD {
    vector<Tensor> parameters;
    double lr;

  public:
    SGD(const vector<Tensor>& parameters, const double lr = .01) : parameters(parameters), lr(lr) {}
    void zeroGrad() {
        for (const auto& param : parameters) {
            param.node->grad.fill(0.);
        }
    }
    void step() {
        for (const auto& param : parameters) {
            param.node->value += (-lr) * param.node->grad;
        }
    }
};

using std::vector;

class Adam {
    vector<Tensor> parameters;
    vector<Tensor> m;  // First moment estimates
    vector<Tensor> v;  // Second moment estimates
    double lr;
    double beta1;
    double beta2;
    double eps;
    int t; // time step (iteration count)

  public:
    // Constructor: default learning rate and beta parameters typical for GANs.
    Adam(const vector<Tensor>& parameters, double lr = 0.0002, double beta1 = 0.5, double beta2 = 0.999, double eps = 1e-8)
        : parameters(parameters), lr(lr), beta1(beta1), beta2(beta2), eps(eps), t(0) {
        // Initialize m and v to zeros, matching the shape of each parameter.
        for (const auto& param : parameters) {
            m.push_back(Tensor(xt::zeros<double>(param.node->value.shape())));
            v.push_back(Tensor(xt::zeros<double>(param.node->value.shape())));
        }
    }

    // Reset gradients for all parameters.
    void zeroGrad() {
        for (const auto& param : parameters) {
            param.node->grad.fill(0.);
        }
    }

    // Update parameters using the Adam update rule.
    void step() {
        t++;  // Increment timestep
        for (size_t i = 0; i < parameters.size(); i++) {
            // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
            m[i].node->value = beta1 * m[i].node->value + (1 - beta1) * parameters[i].node->grad;
            // Update biased second moment estimate: v = beta2 * v + (1 - beta2) * grad^2
            v[i].node->value = beta2 * v[i].node->value + (1 - beta2) * xt::square(parameters[i].node->grad);
            
            // Compute bias-corrected estimates
            auto m_hat = m[i].node->value / (1 - std::pow(beta1, t));
            auto v_hat = v[i].node->value / (1 - std::pow(beta2, t));
            
            // Update parameters: param = param - lr * m_hat / (sqrt(v_hat) + eps)
            parameters[i].node->value += -lr * (m_hat / (xt::sqrt(v_hat) + eps));
        }
    }
};

} // namespace xtorch

