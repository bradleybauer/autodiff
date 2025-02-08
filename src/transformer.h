// transformer.h
#pragma once

#include "Tensor.h"
#include "Modules.h"
#include <cmath>
#include <vector>

namespace xtorch {

// ------------------------------
// 1. Softmax Node and Module
// ------------------------------
struct SoftmaxNode : Node {
    int axis;
    // store output in value for backward
    SoftmaxNode(const shared_ptr<Node>& child, int axis)
        : Node(child), axis(axis) {}
    virtual void forward() {
        // Compute softmax along the given axis:
        //   output = exp(input - max(input)) / sum(exp(input - max(input)))
        auto input = c[0]->value;
        // Use xtensor’s amax to compute maximum along 'axis'
        auto max_val = xt::amax(input, {static_cast<size_t>(axis)});
        // Broadcasting subtraction:
        auto shifted = input - max_val;
        auto exp_val = xt::exp(shifted);
        auto sum_val = xt::sum(exp_val, {static_cast<size_t>(axis)});
        xt::noalias(value) = exp_val / sum_val;
    }
    virtual void backward() {
        // The derivative for softmax:
        // grad_input = output * (grad - sum(grad * output, axis, keepdims=true))
        auto dot = xt::sum(grad * value, {static_cast<size_t>(axis)});
        c[0]->grad += value * (grad - dot);
    }
};

Tensor softmax(const Tensor& x, int axis) {
    return {make_shared<SoftmaxNode>(x.node, axis)};
}

class Softmax : public Module {
    int axis;
public:
    Softmax(int axis) : axis(axis) {}
    virtual Tensor forward(const Tensor& x) {
        return softmax(x, axis);
    }
};

// ------------------------------
// 2. Helper: Batch Dot Product
// ------------------------------
// (This is a simple loop‐based implementation. For performance and full autodiff,
//  you might implement a dedicated BatchDotNode.)
Tensor batch_dot(const Tensor& a, const Tensor& b) {
    // Assume a has shape (batch, M, K) and b has shape (batch, K, N)
    // We will loop over the batch dimension and perform dot for each sample.
    auto a_val = a.getValue();
    auto b_val = b.getValue();
    int batch = a.shape(0);
    std::vector<xt::xarray<double>> outputs;
    for (int i = 0; i < batch; i++) {
        auto a_i = xt::view(a_val, i);
        auto b_i = xt::view(b_val, i);
        auto out_i = xt::linalg::dot(a_i, b_i);
        outputs.push_back(out_i);
    }
    auto stacked = xt::stack(outputs, 0);
    return Tensor(stacked);
}

// ------------------------------
// 3. Multi-Head Attention Module
// ------------------------------
class MultiHeadAttention : public Module {
    int d_model;
    int num_heads;
    int d_k, d_v;
    Linear q_linear, k_linear, v_linear, out_linear;
public:
    MultiHeadAttention(int d_model, int num_heads)
      : d_model(d_model), num_heads(num_heads),
        d_k(d_model / num_heads), d_v(d_model / num_heads),
        q_linear(d_model, d_model), k_linear(d_model, d_model),
        v_linear(d_model, d_model), out_linear(d_model, d_model) {}

    virtual Tensor forward(const Tensor& x) {
        // x is assumed to have shape: (batch, seq_length, d_model)
        Tensor Q = q_linear.forward(x);
        Tensor K = k_linear.forward(x);
        Tensor V = v_linear.forward(x);
        
        // Reshape Q, K, V to (batch, seq_length, num_heads, d_k) and then transpose to (batch, num_heads, seq_length, d_k)
        int batch = x.shape(0);
        int seq_length = x.shape(1);
        Q = Q.reshape({batch, seq_length, num_heads, d_k}).transpose({0, 2, 1, 3});
        K = K.reshape({batch, seq_length, num_heads, d_k}).transpose({0, 2, 1, 3});
        V = V.reshape({batch, seq_length, num_heads, d_v}).transpose({0, 2, 1, 3});
        
        // Flatten batch and head dimensions to perform batch dot
        Q = Q.reshape({batch * num_heads, seq_length, d_k});
        K = K.reshape({batch * num_heads, seq_length, d_k});
        V = V.reshape({batch * num_heads, seq_length, d_v});
        
        // Compute attention scores: scores = Q dot K^T / sqrt(d_k)
        Tensor KT = K.transpose({0, 2, 1});
        Tensor scores = batch_dot(Q, KT);
        double scale = 1.0 / std::sqrt((double)d_k);
        scores = scores * scale;
        
        // Apply softmax along the last dimension (over keys)
        Tensor attn = softmax(scores, -1);
        
        // Compute attention output: output = attn dot V
        Tensor output = batch_dot(attn, V);
        
        // Restore original shape: first reshape to (batch, num_heads, seq_length, d_v) then transpose to (batch, seq_length, num_heads, d_v)
        output = output.reshape({batch, num_heads, seq_length, d_v}).transpose({0, 2, 1, 3});
        // Finally, reshape to (batch, seq_length, d_model)
        output = output.reshape({batch, seq_length, d_model});
        // Apply final linear projection
        return out_linear.forward(output);
    }
};

// ------------------------------
// 4. Layer Normalization Module
// ------------------------------
class LayerNorm : public Module {
    int normalized_shape;
    Tensor gamma, beta;
    double eps;
public:
    // normalized_shape: the number of features (e.g. d_model)
    LayerNorm(int normalized_shape, double eps = 1e-5)
      : normalized_shape(normalized_shape), eps(eps),
        gamma(xt::ones<double>({normalized_shape})),
        beta(xt::zeros<double>({normalized_shape})) {}

    virtual Tensor forward(const Tensor& x) {
        // Assume x has shape: (batch, seq_length, normalized_shape)
        // Compute mean along last dimension:
        Tensor mean = x.sum({2}) * (1.0 / normalized_shape);  // shape: (batch, seq_length)
        mean = mean.reshape({x.shape(0), x.shape(1), 1});
        Tensor diff = x - mean;
        Tensor variance = (diff.square()).sum({2}) * (1.0 / normalized_shape);
        variance = variance.reshape({x.shape(0), x.shape(1), 1});
        // Compute standard deviation using pow for square root:
        Tensor std = (variance + eps).pow(0.5);
        Tensor x_norm = diff / std;
        // gamma and beta will broadcast to (batch, seq_length, normalized_shape)
        return (x_norm * gamma) + beta;
    }
};

// ------------------------------
// 5. Transformer Encoder Layer
// ------------------------------
class TransformerLayer : public Module {
    MultiHeadAttention mha;
    LayerNorm norm1;
    Linear ff_linear1, ff_linear2;
    LayerNorm norm2;
    ReLU relu; // using ReLU as the nonlinearity in the feed-forward network
public:
    TransformerLayer(int d_model, int num_heads, int ff_hidden_dim)
      : mha(d_model, num_heads),
        norm1(d_model),
        ff_linear1(d_model, ff_hidden_dim),
        ff_linear2(ff_hidden_dim, d_model),
        norm2(d_model),
        relu(0.0) {}  // ReLU with zero negative slope

    virtual Tensor forward(const Tensor& x) {
        // Multi-head self-attention with a residual connection + layer norm:
        Tensor attn_out = mha.forward(x);
        Tensor res1 = norm1.forward(x + attn_out);
        // Feed-forward network: two linear layers with ReLU, again with residual + norm:
        Tensor ff_out = ff_linear2.forward(relu.forward(ff_linear1.forward(res1)));
        Tensor res2 = norm2.forward(res1 + ff_out);
        return res2;
    }
};

// ------------------------------
// 6. Simple Transformer Model
// ------------------------------
class TransformerModel : public Module {
    // In this simple model we assume the input has already been embedded to (batch, seq_length, d_model)
    std::vector<Module*> layers;  // transformer layers
    Linear final_linear;          // a final projection (for classification, regression, etc.)
public:
    TransformerModel(int num_layers, int d_model, int num_heads, int ff_hidden_dim, int num_classes)
      : final_linear(d_model, num_classes) {
        // Create transformer layers and store pointers:
        for (int i = 0; i < num_layers; i++) {
            layers.push_back(new TransformerLayer(d_model, num_heads, ff_hidden_dim));
        }
    }

    virtual Tensor forward(const Tensor& x) {
        // x: (batch, seq_length, d_model)
        Tensor out = x;
        for (auto layer : layers) {
            out = layer->forward(out);
        }
        // For example, for classification we might pool over the sequence dimension.
        // Here we simply average over the sequence:
        Tensor pooled = out.sum({1});
        pooled = pooled / (double) out.shape(1);
        return final_linear.forward(pooled);
    }

    virtual ~TransformerModel() {
        for (auto layer : layers) {
            delete layer;
        }
    }
};

} // namespace xtorch
