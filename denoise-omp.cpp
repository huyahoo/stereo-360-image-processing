#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <cmath>
#include <chrono>  // for high_resolution_clock
#include <omp.h> // OpenMP header

using namespace std;

cv::Mat calculateCovarianceMatrix(const cv::Mat& image, int x, int y, int neighborhoodSize) {
    int halfSize = neighborhoodSize / 2;
    int xStart = std::max(0, x - halfSize);
    int yStart = std::max(0, y - halfSize);
    int xEnd = std::min(image.cols, x + halfSize);
    int yEnd = std::min(image.rows, y + halfSize);

    cv::Mat neighborhood = image(cv::Rect(xStart, yStart, xEnd - xStart, yEnd - yStart)).clone();
    cv::Mat reshapedNeighborhood = neighborhood.reshape(1, neighborhood.total());

    cv::Mat mean, covariance;
    cv::calcCovarMatrix(reshapedNeighborhood, covariance, mean, cv::COVAR_NORMAL | cv::COVAR_ROWS | cv::COVAR_SCALE);
    return covariance;
}

cv::Mat denoiseByCovariance(const cv::Mat& src, int neighborhoodSize, double factorRatio) {
    cv::Mat dst(src.size(), src.type());

    #pragma omp parallel for
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            cv::Mat covariance = calculateCovarianceMatrix(src, x, y, neighborhoodSize);
            double determinant = cv::determinant(covariance);

            int kernelSize;
            if (determinant != 0) {
                kernelSize = static_cast<int>(std::round(factorRatio / determinant));
                kernelSize = kernelSize % 2 == 0 ? kernelSize + 1 : kernelSize;
            } else {
                kernelSize = neighborhoodSize;
            }

            // GaussianBlur kernel size should be positive and odd
            kernelSize = std::max(1, kernelSize);
            kernelSize |= 1; // Ensure it's odd

            cv::GaussianBlur(src(cv::Rect(x, y, 1, 1)), dst(cv::Rect(x, y, 1, 1)), cv::Size(kernelSize, kernelSize), 0, 0);

        }
    }

    return dst;
}

int main( int argc, char** argv )
{
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <image_path> <neighborhood_size> <factor_ratio>" << endl;
        return -1;
    }

    // Read the stereo image
    cv::Mat stereo_image = cv::imread(argv[1], cv::IMREAD_COLOR);

    // Check if the image is loaded successfully
    if (stereo_image.empty()) {
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

    // Apply denoising
    cv::Mat denoisedImage;

    // Start the timer
    auto begin = chrono::high_resolution_clock::now();

    // Number of iterations
    const int iter = 1;

    // Perform the operation iter times
    for (int it = 0; it < iter; it++) {
        denoisedImage = denoiseByCovariance(stereo_image, neighborhoodSize, factorRatio);
    }

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the time difference
    std::chrono::duration<double> diff = end - begin;

    // Display the original and denoised images
    // cv::imshow("Original Image", stereo_image);
    // cv::imshow("Denoised Image", denoisedImage);

    // Save the anaglyph image
    std::string filename =  "output/denoised/denoised-omp.jpg";
    cv::imwrite(filename, denoisedImage);

    // Display performance metrics
    cout << "Total time for " << iter << " iterations: " << diff.count() << " s" << endl;
    cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    // Wait for a key press before closing the windows
    // cv::waitKey();

    return 0;
}