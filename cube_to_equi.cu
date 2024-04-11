#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <math.h>
#include <iostream>

using namespace std;
using namespace cv;

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <math.h>
#include <iostream>

using namespace std;
using namespace cv;

__device__ void unit3DToUnit2D(float x, float y, float z, int faceIndex, float &x2D, float &y2D) {
    if(faceIndex == 0) { // X+
        x2D = y + 0.5;
        y2D = z + 0.5;
    } else if(faceIndex == 1) { // Y+
        x2D = (-x) + 0.5;
        y2D = z + 0.5;
    } else if(faceIndex == 2) { // X-
        x2D = (-y) + 0.5;
        y2D = z + 0.5;
    } else if(faceIndex == 3) { // Y-
        x2D = x + 0.5;
        y2D = z + 0.5;
    } else if(faceIndex == 4) { // Z+
        x2D = y + 0.5;
        y2D = (-x) + 0.5;
    } else { // Z-
        x2D = y + 0.5;
        y2D = x + 0.5;
    }
    y2D = 1 - y2D;
}

__device__ void project(float theta, float phi, float sign, int axis, float &x, float &y, float &z) {
    float rho;
    if(axis == 0) { // X
        x = sign * 0.5;
        rho = x / (cos(theta) * sin(phi));
        y = rho * sin(theta) * sin(phi);
        z = rho * cos(phi);
    } else if(axis == 1) { // Y
        y = sign * 0.5;
        rho = y / (sin(theta) * sin(phi));
        x = rho * cos(theta) * sin(phi);
        z = rho * cos(phi);
    } else { // Z
        z = sign * 0.5;
        rho = z / cos(phi);
        x = rho * cos(theta) * sin(phi);
        y = rho * sin(theta) * sin(phi);
    }
}

__global__ void convertEquirectUVtoUnit2D(cuda::PtrStepSz<uchar3> posx, cuda::PtrStepSz<uchar3> negx, cuda::PtrStepSz<uchar3> posy, cuda::PtrStepSz<uchar3> negy, cuda::PtrStepSz<uchar3> posz, cuda::PtrStepSz<uchar3> negz, cuda::PtrStepSz<uchar3> output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < output.cols && y < output.rows) {
        float U = float(x) / (output.cols - 1);
        float V = float(y) / (output.rows - 1);
        float theta = U * 2 * M_PI;
        float phi = V * M_PI;
        float x3D = cos(theta) * sin(phi);
        float y3D = sin(theta) * sin(phi);
        float z3D = cos(phi);
        float maximum = max(abs(x3D), max(abs(y3D), abs(z3D)));
        x3D /= maximum;
        y3D /= maximum;
        z3D /= maximum;
        int faceIndex;
        if(x3D == 1 || x3D == -1) {
            faceIndex = (x3D == 1) ? 0 : 2;
            project(theta, phi, x3D, 0, x3D, y3D, z3D);
        } else if(y3D == 1 || y3D == -1) {
            faceIndex = (y3D == 1) ? 1 : 3;
            project(theta, phi, y3D, 1, x3D, y3D, z3D);
        } else {
            faceIndex = (z3D == 1) ? 4 : 5;
            project(theta, phi, z3D, 2, x3D, y3D, z3D);
        }
        float x2D, y2D;
        unit3DToUnit2D(x3D, y3D, z3D, faceIndex, x2D, y2D);
        x2D *= posx.cols;
        y2D *= posx.rows;
        int xPixel = int(x2D);
        int yPixel = int(y2D);
        uchar3 color;
        switch(faceIndex) {
            case 0: color = posx(yPixel, xPixel); break;
            case 1: color = posy(yPixel, xPixel); break;
            case 2: color = negx(yPixel, xPixel); break;
            case 3: color = negy(yPixel, xPixel); break;
            case 4: color = posz(yPixel, xPixel); break;
            case 5: color = negz(yPixel, xPixel); break;
        }
        output(y, x) = color;
    }
}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <inputPath>" << endl;
        return 1;
    }

    string inputPath = argv[1];

    // Load cube face images
    Mat posx = imread(inputPath + "posz.jpg");
    Mat negx = imread(inputPath + "negz.jpg");
    Mat posy = imread(inputPath + "posx.jpg");
    Mat negy = imread(inputPath + "negx.jpg");
    Mat posz = imread(inputPath + "posy.jpg");
    Mat negz = imread(inputPath + "negy.jpg");

    // Check if cube face images are loaded successfully
    if (posx.empty() || negx.empty() || posy.empty() || negy.empty() || posz.empty() || negz.empty()) {
        cerr << "Error: Couldn't open one or more cube face images!" << endl;
        return 1;
    }

    int squareLength = posx.cols;
    int outputWidth = squareLength * 2;
    int outputHeight = squareLength * 1;

    // Upload inputImage image to GPU
    cuda::GpuMat d_posx, d_negx, d_posy, d_negy, d_posz, d_negz;
    d_posx.upload(posx);
    d_negx.upload(negx);
    d_posy.upload(posy);
    d_negy.upload(negy);
    d_posz.upload(posz);
    d_negz.upload(negz);

    cv::cuda::GpuMat d_equirectangular(outputHeight, outputWidth, CV_8UC3);

    dim3 block(16, 16);
    dim3 grid(divUp(outputWidth, block.x), divUp(outputHeight, block.y));
    convertEquirectUVtoUnit2D<<<grid, block>>>(d_posx, d_negx, d_posy, d_negy, d_posz, d_negz, d_equirectangular);

    Mat equirectangular;
    d_equirectangular.download(equirectangular);

    // Save the output image
    imwrite("EQUIRECTANGULAR.png", equirectangular);

    return 0;
}