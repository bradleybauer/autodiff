#pragma once

#include <fstream>
#include <stdexcept>
#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xio.hpp"

#include "xtorch.h"
using namespace xtorch;

#include <xtensor-io/ximage.hpp>

#include <iostream>
using std::cout;
using std::endl;
#include <filesystem>
namespace fs = std::filesystem;

const int imgSz = 64;
const int imgDepth = 3;

struct DataLoader {
    vector<vector<std::string>> batches;
    const int batchSize;

    DataLoader(const int batchSize) : batchSize(batchSize) {
        std::string folder = "/home/bradley/Downloads/archive/50k/";
        vector<std::string> batch;
        for (const auto& entry : fs::directory_iterator(folder)) {
            batch.push_back(entry.path());
            if (batch.size() == batchSize) {
                batches.push_back(batch);
                batch.clear();
            }
        }
    }

    int size() { return batches.size(); }

    Tensor loadBatch(int i) {
        vector<std::string>& batch = batches[i];
        xt::xarray<double> ret = xt::zeros<double>({batchSize, imgSz, imgSz, imgDepth});
        for (int j = 0; j < batchSize; ++j) {
            auto img = xt::load_image(batch[j]);
            xt::view(ret, j) = 2. * (img / 255.) - 1.;
        }
        return {xt::transpose(ret, {0, 3, 1, 2})};
    }
};

void dump_img(xt::xarray<double> x, const std::string& filename)
{
    if (x.dimension() != 3)
    {
        throw std::runtime_error("dump_img_ppm expects a 3D image (channels, height, width)");
    }
    
    // Normalize from [-1, 1] to [0, 255]
    xt::xarray<double> normalized = (((x + 1.0) / 2.0) * 255.0);
    // Cast to unsigned char
    xt::xarray<unsigned char> image = xt::cast<unsigned char>(normalized);
    // Transpose from (channels, height, width) to (height, width, channels)
    xt::xarray<unsigned char> transposed = xt::transpose(image, {1, 2, 0});
    
    auto shape = transposed.shape();
    int height   = static_cast<int>(shape[0]);
    int width    = static_cast<int>(shape[1]);
    int channels = static_cast<int>(shape[2]);
    
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
    {
        throw std::runtime_error("Could not open file for writing");
    }
    
    if (channels == 3)
    {
        // Write binary PPM header for a color image
        ofs << "P6\n" << width << " " << height << "\n255\n";
        ofs.write(reinterpret_cast<const char*>(transposed.data()), width * height * 3);
    }
    else if (channels == 1)
    {
        // Write binary PGM header for a grayscale image
        ofs << "P5\n" << width << " " << height << "\n255\n";
        ofs.write(reinterpret_cast<const char*>(transposed.data()), width * height);
    }
    else
    {
        throw std::runtime_error("Unsupported number of channels");
    }
    ofs.close();
}
