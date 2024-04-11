#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <string>
#include <cmath>
#include <chrono>  

using namespace std;
using namespace cv;

__device__ float getTheta(float x, float y) {
    float rtn = 0;
    if (y < 0) {
        rtn = atan2f(y, x) * -1;
    } else {
        rtn = M_PI + (M_PI - atan2f(y, x));
    }
    return rtn;
}

__global__ void equirectangularToCubeMap(const cv::cuda::PtrStepSz<uchar3> src, cv::cuda::PtrStepSz<uchar3> dst,
                                         int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x < outputWidth && dst_y < outputHeight) {
        float sqr = inputWidth / 4.0f;
        float tx, ty, normTheta, normPhi;
        float tempX, tempY, tempZ;

        if (dst_y < sqr + 1) {
            if (dst_x < sqr + 1) {
                tx = dst_x;
                ty = dst_y;
                tempX = tx - 0.5f * sqr;
                tempY = 0.5f * sqr;
                tempZ = ty - 0.5f * sqr;
            } else if (dst_x < 2 * sqr + 1) {
                // top middle [X+]
                tx = dst_x - sqr;
                ty = dst_y;
                tempX = 0.5f * sqr;
                tempY = (tx - 0.5f * sqr) * -1;
                tempZ = ty - 0.5f * sqr;
            } else {
                // top right [Y-]
                tx = dst_x - 2 * sqr;
                ty = dst_y;
                tempX = (tx - 0.5f * sqr) * -1;
                tempY = -0.5f * sqr;
                tempZ = ty - 0.5f * sqr;
            }
        } else {
            if (dst_x < sqr + 1) {
                // bottom left box [X-]
                tx = dst_x;
                ty = dst_y - sqr;
                tempX = -0.5f * sqr;
                tempY = tx - 0.5f * sqr;
                tempZ = ty - 0.5f * sqr;
            } else if (dst_x < 2 * sqr + 1) {
                // bottom middle [Z-]
                tx = dst_x - sqr;
                ty = dst_y - sqr;
                tempX = (ty - 0.5f * sqr) * -1;
                tempY = (tx - 0.5f * sqr) * -1;
                tempZ = 0.5f * sqr;
            } else {
                // bottom right [Z+]
                tx = dst_x - 2 * sqr;
                ty = dst_y - sqr;
                tempX = ty - 0.5f * sqr;
                tempY = (tx - 0.5f * sqr) * -1;
                tempZ = -0.5f * sqr;
            }
        }

        // Normalize theta and phi
        float rho = sqrtf(tempX * tempX + tempY * tempY + tempZ * tempZ);
        normTheta = getTheta(tempX, tempY) / (2 * M_PI);
        normPhi = (M_PI - acosf(tempZ / rho)) / M_PI;

        // Calculate input coordinates
        float iX = normTheta * inputWidth;
        float iY = normPhi * inputHeight;

        // Handle possible overflows
        if (iX >= inputWidth) {
            iX -= inputWidth;
        }
        if (iY >= inputHeight) {
            iY -= inputHeight;
        }

        // Copy pixel value from input to output
        dst(dst_y, dst_x) = src((int)iY, (int)iX);
    }
}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <inputImage>" << endl;
        return 1;
    }

    string inputPath = argv[1];
    Mat inputImage = imread(inputPath);

    if (inputImage.empty()) {
        cerr << "Error: Couldn't open the inputImage image!" << endl;
        return 1;
    }

    size_t lastSlash = inputPath.find_last_of("/");
    string fileName = inputPath.substr(lastSlash + 1);
    fileName = fileName.substr(0, fileName.find_last_of("."));
    cout << "Processing " << fileName << "..." << endl;

    int inputWidth = inputImage.cols;
    int inputHeight = inputImage.rows;
    float sqr = inputWidth / 4.0f;
    int outputWidth = sqr * 3;
    int outputHeight = sqr * 2;

    // Upload inputImage image to GPU
    cv::cuda::GpuMat d_input;
    d_input.upload(inputImage);

    // Create output GPU matrix
    cv::cuda::GpuMat d_output(outputHeight, outputWidth, CV_8UC3);

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid(divUp(inputImage.cols, block.x), divUp(inputImage.rows, block.y));
    equirectangularToCubeMap<<<grid, block>>>(d_input, d_output, inputWidth, inputHeight, outputWidth, outputHeight);

    // Download output from GPU
    Mat output;
    d_output.download(output);

    int faceWidth = outputWidth / 3;
    int faceHeight = outputHeight / 2;

    // Split the output into six different images
    Mat face1 = output(Rect(0, 0, faceWidth, faceHeight));
    Mat face2 = output(Rect(faceWidth, 0, faceWidth, faceHeight));
    Mat face3 = output(Rect(2 * faceWidth, 0, faceWidth, faceHeight));
    Mat face4 = output(Rect(0, faceHeight, faceWidth, faceHeight));
    Mat face5 = output(Rect(faceWidth, faceHeight, faceWidth, faceHeight));
    Mat face6 = output(Rect(2 * faceWidth, faceHeight, faceWidth, faceHeight));

    // Save output images
    imwrite("output/cube/" + fileName + "_cube.jpg", output);
    imwrite("output/cube/" + fileName + "_negx.jpg", face1);
    imwrite("output/cube/" + fileName + "_posz.jpg", face2);

    imwrite("output/cube/" + fileName + "_posx.jpg", face3);
    imwrite("output/cube/" + fileName + "_negz.jpg", face4);

    imwrite("output/cube/" + fileName + "_negy.jpg", face5);
    imwrite("output/cube/" + fileName + "_posy.jpg", face6);

    return 0;
}
