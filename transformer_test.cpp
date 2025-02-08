#include "transformer.h"
#include "Tensor.h"
#include <xtensor/xrandom.hpp>

int main() {
    using namespace xtorch;
    // Define model parameters
    int batch = 2;
    int seq_length = 10;
    int d_model = 64;
    int num_heads = 8;
    int ff_hidden_dim = 256;
    int num_layers = 2;
    int num_classes = 5;

    // Create a dummy input (e.g. from an embedding lookup)
    xt::xarray<double> input_array = xt::random::randn<double>({batch, seq_length, d_model});
    Tensor input(input_array);

    // Build the transformer model
    TransformerModel model(num_layers, d_model, num_heads, ff_hidden_dim, num_classes);

    // Forward pass
    Tensor output = model.forward(input);
    std::cout << "Output: " << output.getValue() << std::endl;

    // (Optionally, compute loss, call backward(), and update parameters)
    return 0;
}
