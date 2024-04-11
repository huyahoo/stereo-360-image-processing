#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <cmath>

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

__global__ void equirectangularToCubeMap(const PtrStepSz<uchar3> src, PtrStepSz<uchar3> dst,
                                         int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outputWidth && y < outputHeight) {
        float sqr = inputWidth / 4.0f;
        float outputX = x + 0.5f;
        float outputY = y + 0.5f;
        float tx, ty, normTheta, normPhi;
        float theta, phi;

        if (outputY < sqr + 1) {
            if (outputX < sqr + 1) {
                tx = outputX;
                ty = outputY;
                float tempX = tx - 0.5f * sqr;
                float tempY = 0.5f * sqr;
                float tempZ = ty - 0.5f * sqr;
                theta = getTheta(tempX, tempY);
                if (tempY < 0) {
                    theta = M_PI + theta;
                }
                phi = M_PI - acosf(tempZ / sqrtf(tempX * tempX + tempY * tempY + tempZ * tempZ));
            } else if (outputX < 2 * sqr + 1) {
                // top middle [X+]
                tx = outputX - sqr;
                ty = outputY;
                float tempX = 0.5f * sqr;
                float tempY = (tx - 0.5f * sqr) * -1;
                float tempZ = ty - 0.5f * sqr;
                theta = getTheta(tempX, tempY);
                if (tempY < 0) {
                    theta = M_PI + theta;
                }
                phi = M_PI - acosf(tempZ / sqrtf(tempX * tempX + tempY * tempY + tempZ * tempZ));
            } else {
                // top right [Y-]
                tx = outputX - 2 * sqr;
                ty = outputY;
                float tempX = (tx - 0.5f * sqr) * -1;
                float tempY = -0.5f * sqr;
                float tempZ = ty - 0.5f * sqr;
                theta = getTheta(tempX, tempY);
                if (tempY < 0) {
                    theta = M_PI + theta;
                }
                phi = M_PI - acosf(tempZ / sqrtf(tempX * tempX + tempY * tempY + tempZ * tempZ));
            }
        } else {
            if (outputX < sqr + 1) {
                // bottom left box [X-]
                tx = outputX;
                ty = outputY - sqr;
                float tempX = -0.5f * sqr;
                float tempY = tx - 0.5f * sqr;
                float tempZ = ty - 0.5f * sqr;
                theta = getTheta(tempX, tempY);
                if (tempY < 0) {
                    theta = M_PI + theta;
                }
                phi = M_PI - acosf(tempZ / sqrtf(tempX * tempX + tempY * tempY + tempZ * tempZ));
            } else if (outputX < 2 * sqr + 1) {
                // bottom middle [Z-]
                tx = outputX - sqr;
                ty = outputY - sqr;
                float tempX = (ty - 0.5f * sqr) * -1;
                float tempY = (tx - 0.5f * sqr) * -1;
                float tempZ = 0.5f * sqr;
                theta = getTheta(tempX, tempY);
                if (tempY < 0) {
                    theta = M_PI + theta;
                }
                phi = M_PI - acosf(tempZ / sqrtf(tempX * tempX + tempY * tempY + tempZ * tempZ));
            } else {
                // bottom right [Z+]
                tx = outputX - 2 * sqr;
                ty = outputY - sqr;
                float tempX = ty - 0.5f * sqr;
                float tempY = (tx - 0.5f * sqr) * -1;
                float tempZ = -0.5f * sqr;
                theta = getTheta(tempX, tempY);
                if (tempY < 0) {
                    theta = M_PI + theta;
                }
                phi = M_PI - acosf(tempZ / sqrtf(tempX * tempX + tempY * tempY + tempZ * tempZ));
            }
        }

        // Normalize theta and phi
        normTheta = theta / (2 * M_PI);
        normPhi = phi / M_PI;

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
        dst(y, x) = src((int)iY, (int)iX);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_image>" << endl;
        return 1;
    }

    Mat input = imread(argv[1]);
    if (input.empty()) {
        cerr << "Error: Couldn't open the input image!" << endl;
        return 1;
    }

    int inputWidth = input.cols;
    int inputHeight = input.rows;
    float sqr = inputWidth / 4.0f;
    int outputWidth = sqr * 3;
    int outputHeight = sqr * 2;

    // Upload input image to GPU
    cuda::GpuMat d_input;
    d_input.upload(input);

    // Create output GPU matrix
    cuda::GpuMat d_output(outputHeight, outputWidth, CV_8UC3);

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((outputWidth + block.x - 1) / block.x, (outputHeight + block.y - 1) / block.y);
    equirectangularToCubeMap<<<grid, block>>>(d_input, d_output, inputWidth, inputHeight, outputWidth, outputHeight);

    // Download output from GPU
    Mat output;
    d_output.download(output);

    // Save output image
    imwrite("cube_map_output.png", output);

    return 0;
}
