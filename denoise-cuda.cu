#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <cmath>
#include <chrono>  // for high_resolution_clock
#include <cuda_runtime.h>

using namespace std;

__global__ void calculateAndDenoiseKernel(const uchar3* src, uchar3* dst, int cols, int rows, int neighborhoodSize, float factorRatio) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int halfSize = neighborhoodSize / 2;
        int xStart = max(0, x - halfSize);
        int yStart = max(0, y - halfSize);
        int xEnd = min(cols, x + halfSize);
        int yEnd = min(rows, y + halfSize);

        float3 mean = make_float3(0.0f, 0.0f, 0.0f);
        float3 cov = make_float3(0.0f, 0.0f, 0.0f);

        for (int j = yStart; j < yEnd; ++j) {
            for (int i = xStart; i < xEnd; ++i) {
                uchar3 pixel = src[j * cols + i];
                mean.x += pixel.x;
                mean.y += pixel.y;
                mean.z += pixel.z;
            }
        }

        int count = (xEnd - xStart) * (yEnd - yStart);
        mean.x /= count;
        mean.y /= count;
        mean.z /= count;

        for (int j = yStart; j < yEnd; ++j) {
            for (int i = xStart; i < xEnd; ++i) {
                uchar3 pixel = src[j * cols + i];
                float3 diff = make_float3(pixel.x - mean.x, pixel.y - mean.y, pixel.z - mean.z);
                cov.x += diff.x * diff.x;
                cov.y += diff.y * diff.y;
                cov.z += diff.z * diff.z;
            }
        }

        cov.x /= count;
        cov.y /= count;
        cov.z /= count;

        float determinant = cov.x * cov.y * cov.z;

        int kernelSize;
        if (determinant != 0) {
            kernelSize = static_cast<int>(round(factorRatio / determinant));
            kernelSize = kernelSize % 2 == 0 ? kernelSize + 1 : kernelSize;
        } else {
            kernelSize = neighborhoodSize;
        }

        kernelSize = max(1, kernelSize);
        kernelSize |= 1; // Ensure it's odd

        float3 sum = make_float3(0.0f, 0.0f, 0.0f);
        count = 0;

        for (int j = y - kernelSize / 2; j <= y + kernelSize / 2; ++j) {
            for (int i = x - kernelSize / 2; i <= x + kernelSize / 2; ++i) {
                if (i >= 0 && i < cols && j >= 0 && j < rows) {
                    uchar3 pixel = src[j * cols + i];
                    sum.x += pixel.x;
                    sum.y += pixel.y;
                    sum.z += pixel.z;
                    ++count;
                }
            }
        }

        sum.x /= count;
        sum.y /= count;
        sum.z /= count;

        dst[y * cols + x] = make_uchar3(static_cast<unsigned char>(sum.x), static_cast<unsigned char>(sum.y), static_cast<unsigned char>(sum.z));
    }
}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


void processCUDA(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int neighborhoodSize, double factorRatio) {
    dim3 block(32, 8);
    dim3 grid(divUp(src.cols, block.x), divUp(src.rows, block.y));

    calculateAndDenoiseKernel<<<grid, block>>>(
        reinterpret_cast<uchar3*>(const_cast<unsigned char*>(src.ptr())), 
        reinterpret_cast<uchar3*>(dst.ptr()), 
        src.cols, src.rows, neighborhoodSize, factorRatio);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <image_path> <neighborhood_size> <factor_ratio>" << endl;
        return -1;
    }

    cv::Mat input_img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (input_img.empty()) {
        cerr << "Error: Unable to load image." << endl;
        return -1;
    }

    int neighborhoodSize = atoi(argv[2]);
    double factorRatio = atof(argv[3]);

    if (!factorRatio || !neighborhoodSize) {
        cerr << "Error: Invalid input." << endl;
        cerr << "Error: Neighborhood size must be an odd number." << endl;
        cerr << "Error: Factor ratio must be greater than 0." << endl;
        return -1;
    }
    
    if (neighborhoodSize % 2 == 0) {
        cerr << "Error: Neighborhood size must be an odd number." << endl;
        return -1;
    }

    if (factorRatio <= 0) {
        cerr << "Error: Factor ratio must be greater than 0." << endl;
        return -1;
    }

    cv::Mat denoised_image;

    cv::cuda::GpuMat d_input_img, d_denoised_image;

    // Start the timer
    auto begin = chrono::high_resolution_clock::now();

    // Number of iterations
    const int iter = 1;

    for (int it = 0; it < iter; it++) {
        d_input_img.upload(input_img);
        d_denoised_image.upload(input_img);
        processCUDA(d_input_img, d_denoised_image, neighborhoodSize, factorRatio);
        d_denoised_image.download(denoised_image);
    }

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the time difference
    chrono::duration<double> diff = end - begin;

    // Display performance metrics
    cout << "Total time for " << iter << " iterations: " << diff.count() << " s" << endl;
    cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    // Display the original and processed images
    // cv::imshow("Original Image", input_img);
    // cv::imshow("Denoised Image", denoised_image);

    // Save the anaglyph image
    std::string filename =  "output/denoised/denoised-cuda.jpg";
    cv::imwrite(filename, denoised_image);

    // cv::waitKey();

    return 0;
}
