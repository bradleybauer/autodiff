#pragma once

#include "xtorch.h"
#include <functional>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

namespace xtorch {

// Performs numerical gradient checking for both weights and (optionally) the input.
void gradCheckModule(Module& module,
                     const std::function<Tensor()>& lossFn,
                     Tensor& input) {
    const double eps = .000001;

    // Run a forward pass to build the computation graph and compute analytic gradients.
    auto loss0 = lossFn();
    loss0.backward();

    // Helper lambda that numerically checks the gradient of a given tensor.
    auto checkGradient = [&](Tensor& t, const std::string& tensorLabel) {
        auto data = xt::flatten(t.node->value);
        auto analyticGrad = xt::flatten(t.node->grad);
        size_t discrepancies = 0;
        bool printedHeader = false;

        for (size_t i = 0; i < data.size(); ++i) {
            const double orig = data[i];
            // Compute loss at perturbed values.
            data[i] = orig + eps;
            auto loss1 = lossFn();
            data[i] = orig - eps;
            auto loss2 = lossFn();
            data[i] = orig;

            // Central difference quotient.
            double numericalGrad = ((loss1 - loss2) / (2 * eps)).getValue()[0];

            // If the difference is above threshold (or if we want verbose output), print details.
            if (std::abs(numericalGrad - analyticGrad[i]) > 0.001) {
                if (!printedHeader) {
                    std::cout << "::: " << tensorLabel << " check for module '"
                              << module.name() << "' :::\n";
                    printedHeader = true;
                }
                std::cout << "Index " << i << ": numerical grad = " << numericalGrad
                          << ", analytic grad = " << analyticGrad[i]
                          << ", ratio = " << (analyticGrad[i] / numericalGrad)
                          << ", shape = " << xt::adapt(t.shape()) << "\n";
                discrepancies++;
            }
        }
        if (discrepancies > 0) {
            std::cout << "Total discrepancies in " << tensorLabel << ": "
                      << discrepancies << "\n";
        }
    };

    // Check gradients for each weight in the module.
    for (auto& w : module.parameters()) {
        checkGradient(w, "Weights");
    }
    // Optionally check gradients with respect to the input.
    checkGradient(input, "Input");
}

// Runs a series of gradient check tests. Each test is wrapped in a header so you know which block fails.
void gradCheck() {
    auto runTest = [&](const std::string& testName, auto testFn) {
        std::cout << "========================\n";
        std::cout << "Running Test Block: " << testName << "\n";
        testFn();
        std::cout << "Finished Test Block: " << testName << "\n";
        std::cout << "========================\n\n";
    };

    runTest("Linear MSELoss with 1D input", [&]() {
        Tensor input{xt::xarray<double>{{0.5, 0.3, 0.1}}};
        Tensor target{xt::xarray<double>{{1, 0, -1, 0}}};
        Linear m{3, 4};
        auto crit = MSELoss();
        auto lossFn = [&]() { return crit(m(input), target); };
        gradCheckModule(m, lossFn, input);
    });

    runTest("Linear MSELoss with 2D input", [&]() {
        Tensor input{xt::xarray<double>{{0.5, 0.3, 0.1},
                                         {0.99, 2.3, -0.4}}};
        Tensor target{xt::xarray<double>{{1, 0, -1, 0},
                                          {-0.5, 0.3, 0.4, 1}}};
        Linear m{3, 4};
        auto crit = MSELoss();
        auto lossFn = [&]() { return crit(m(input), target); };
        gradCheckModule(m, lossFn, input);
    });

    runTest("Linear MSELoss with 2D input test 2", [&]() {
        Tensor input{xt::xarray<double>{{0.5, 0.3, 0.1},
                                         {0.99, 2.3, -0.4}}};
        Tensor target{xt::xarray<double>{{1, 0, -1, 0},
                                          {-0.5, 0.3, 0.4, 1}}};
        Linear m{3, 4};
        auto crit = MSELoss();
        auto lossFn = [&]() { return crit(m(input), target); };
        gradCheckModule(m, lossFn, input);
    });

    runTest("Linear sum loss with 1D input", [&]() {
        Tensor input{xt::xarray<double>{{0.5, 0.3, 0.1}}};
        Linear m{3, 4};
        auto lossFn = [&]() { return m(input).sum(); };
        gradCheckModule(m, lossFn, input);
    });

    runTest("Linear sum loss with 2D input", [&]() {
        Tensor input{xt::xarray<double>{{0.5, 0.3, 0.1},
                                         {3, 2, -6}}};
        Linear m{3, 4};
        auto lossFn = [&]() { return m(input).sum(); };
        gradCheckModule(m, lossFn, input);
    });

    runTest("Nonlinear composite loss with 2D input", [&]() {
        Tensor input{xt::xarray<double>{{0.5, 0.3, 0.1},
                                         {3, 2, -6}}};
        Linear m{3, 4};
        auto lossFn = [&]() {
            auto y = m(input).sum();
            return (y * y - y * 0.5).sin();
        };
        gradCheckModule(m, lossFn, input);
    });

    runTest("Linear with Nonlinear Composite Loss", [&]() {
        Tensor input{xt::xarray<double>{{0.5, 0.3, 0.1},
                                         {3, 2, -6}}};
        Linear m{3, 4};
        auto lossFn = [&]() {
            auto y = m(input).sum();
            return y / ((y * y).relu() + y.relu() * 3.141592).sin();
        };
        gradCheckModule(m, lossFn, input);
    });

    runTest("Linear Sum over Multiple Dimensions", [&]() {
        Tensor input{xt::xarray<double>{{0.5, 0.3, 0.1},
                                         {3, 2, -6}}};
        Linear m{3, 4};
        auto lossFn = [&]() { return m(input).sum({0}).sum({0}); };
        gradCheckModule(m, lossFn, input);
    });

    runTest("Two-Layer Network with ReLU", [&]() {
        Tensor input{xt::xarray<double>{{0.5, 0.3, 0.1},
                                         {3, 2, -6}}};
        Linear m{3, 4};
        Linear m1{4, 3};
        auto lossFn = [&]() { return m1(m(input).relu(0.1)).relu(0.1).sum(); };
        gradCheckModule(m, lossFn, input);
    });

    runTest("Conv2d with ReLU", [&]() {
        Tensor input{xt::random::randn<double>({1, 2, 4, 4})};
        Conv2d conv(2, 2, 4, 2, 1, true);
        auto lossFn = [&]() { return conv(input).relu(0.1).sum(); };
        gradCheckModule(conv, lossFn, input);
    });

    runTest("Sequential: Conv2d -> Flatten -> Linear", [&]() {
        const int inDepth = 2, outDepth = 3, filterSize = 5, pad = 2, stride = 2, inSz = 7, batchSize = 1;
        const int outSz = (inSz - filterSize + 2 * pad) / stride + 1;
        const int flatSize = outDepth * (outSz * outSz);
        Tensor input{xt::random::randn<double>({batchSize, inDepth, inSz, inSz})};
        auto m = Sequential(Conv2d{inDepth, outDepth, filterSize, stride, pad}, FlattenBatch{}, Linear{flatSize, 10});
        auto lossFn = [&]() { return m(input).sum(); };
        gradCheckModule(m, lossFn, input);
    });

    runTest("BatchNorm2d with ReLU", [&]() {
        const int numFeatures = 3, outSz = 4, batchSize = 32;
        Tensor x{xt::random::randn<double>({batchSize, numFeatures, outSz, outSz}) + 2.};
        BatchNorm2d b(numFeatures, 1e-5, 0.1, true, false);
        auto lossFn = [&]() {
            return b(x).relu().sum();
        };
        gradCheckModule(b, lossFn, x);
    });

    runTest("Simplified Sequential netD with Conv2d and BatchNorm2d", [&]() {
        const int numFeatures = 3, outSz = 16, batchSize = 1;
        Tensor x{xt::random::randn<double>({batchSize, numFeatures, outSz, outSz}) + 2.};
        Sequential netD{
            // First layer: from 3 channels to 16 channels.
            Conv2d{3, 16, 4, 2, 1, false},
            BatchNorm2d{16, 1e-5, 0.1, true, false},
            ReLU{0.2},
            // Second layer: from 16 channels to 1 channel.
            Conv2d{16, 1, 4, 2, 1, false}
        };
        auto lossFn = [&]() { return netD(x).sum().sigmoid(); };
        gradCheckModule(netD, lossFn, x);
    });

    runTest("Simplified Sequential netG with ConvTranspose2d and BatchNorm2d", [&]() {
        const int batchSize = 1;
        // Reduced input channels from 100 to 10 for speed.
        Tensor x{xt::random::randn<double>({batchSize, 10, 1, 1}) + 2.};
        Sequential netG{
            ConvTranspose2d{10, 16, 4, 1, 0, false},
            BatchNorm2d{16, 1e-5, 0.1, true, false},
            ReLU{0.2},
            ConvTranspose2d{16, 1, 4, 1, 0, false}
        };
        auto lossFn = [&]() { return netG(x).sum().sigmoid(); };
        gradCheckModule(netG, lossFn, x);
    });

    runTest("BCEWithLogitsLoss with Linear", [&]() {
        Tensor x{xt::xarray<double>{{0.5, 0.3, 0.1, -2},
                                    {0.2342, 0.1, -0.06, 0}}};
        Tensor y{xt::xarray<double>{1, 0}};
        Linear l{4, 1};
        auto crit = BCEWithLogitsLoss();
        auto lossFn = [&]() { return crit(l(x).reshape({2}), y); };
        gradCheckModule(l, lossFn, x);
    });

    runTest("Sequential: ConvTranspose2d and ReLU", [&]() {
        const int inDepth = 2, outDepth = 3, filterSize = 4, pad = 1, stride = 2, inSz = 8, batchSize = 1;
        Tensor input{xt::random::randn<double>({batchSize, inDepth, inSz, inSz})};
        auto m = Sequential(ConvTranspose2d{inDepth, outDepth, filterSize, stride, pad, false}, ReLU{0.1});
        auto lossFn = [&]() { return m(input).sum(); };
        gradCheckModule(m, lossFn, input);
    });

    runTest("BCELoss with Linear", [&]() {
        Tensor x{xt::xarray<double>{{0.5, 0.3, 0.1, -2},
                                    {0.2342, 0.1, -0.06, 0}}};
        Tensor y{xt::xarray<double>{1, 0}};
        Linear l{4, 1};
        auto crit = BCELoss();
        auto lossFn = [&]() {
            return crit(l(x).sigmoid().reshape({2}), y);
        };
        gradCheckModule(l, lossFn, x);
    });
}

}; // namespace xtorch

    // runTest("Sigmoid Activation on Tensor", [&]() {
    //     Tensor x{xt::xarray<double>{{0.5, 0.3, 0.1, -2},
    //                                 {0.2342, 0.1, -0.06, 0}}};
    //     auto lossFn = [&]() { return x.sigmoid().sum(); };
    //     // Create a dummy module with no parameters.
    //     Module dummy;
    //     gradCheckModule(dummy, lossFn, x);
    // });

    // runTest("Tanh Activation on Tensor", [&]() {
    //     Tensor x{xt::xarray<double>{{0.5, 0.3, 0.1, -2},
    //                                 {0.2342, 0.1, -0.06, 0}}};
    //     auto lossFn = [&]() { return x.tanh().sum(); };
    //     Module dummy;
    //     gradCheckModule(dummy, lossFn, x);
    // });

    // runTest("Power, Sum, and Square Activation on Tensor", [&]() {
    //     Tensor x{xt::xarray<double>{{0.5, 0.3, 0.1, 2},
    //                                 {0.2342, 0.1, 0.06, 0}}};
    //     auto lossFn = [&]() { return x.pow(3.141592).sum().square(); };
    //     Module dummy;
    //     gradCheckModule(dummy, lossFn, x, true);
    // });



    // runTest("Dot Product and Multiple Dot", [&]() {
    //     Tensor x{xt::xarray<double>{{0.5, 0.3, 0.1, -2},
    //                                 {3, 2, -6, 0}}};
    //     Tensor A{xt::random::randn<double>({4, 4})};
    //     Tensor b{xt::random::randn<double>({2, 4})};
    //     auto lossFn = [&]() {
    //         auto z = x.dot(A) + b;
    //         auto y = z.dot(A) + b;
    //         return (y.sin().square() * 3 * z).sum();
    //     };
    //     // Create a dummy module to hold A and b.
    //     Module dummy;
    //     dummy.addParameter(A);
    //     dummy.addParameter(b);
    //     gradCheckModule(dummy, lossFn, x);
    // });