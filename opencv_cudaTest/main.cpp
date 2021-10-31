#include <opencv2/core.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

#include "cuda_runtime.h"
#include "myTimer.h"


int main()
{
    std::ofstream out("out.csv");
    out << "a" << "," << std::endl;

    const int ITR = 10000;
    MyTimer timer,timer2,timer3,timer4;
    std::cout << cv::getBuildInformation() << std::endl;
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    cv::Mat src = cv::imread("bunny.png", cv::IMREAD_UNCHANGED);
    cv::Mat src_gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
    }
    else if (src.channels() == 1) {
        src_gray = src;
    }
    cv::Mat canny_dst;
    cv::Mat sobelX_dst;
    cv::Mat sobelY_dst;


    if (src.empty())
    {
        return -1;
    }

    //CannyEdgeDetectorの定義
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(50, 100,3,true);
    cv::Ptr<cv::cuda::Filter> gaussFilter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5), 0, 0, cv::BORDER_REPLICATE);

    //SobelFilterの定義、cudaではFilterクラスで管理されている
    cv::Ptr<cv::cuda::Filter> sobelX = cv::cuda::createSobelFilter(CV_32FC1, CV_32FC1, 1, 0, 3);
    cv::Ptr<cv::cuda::Filter> sobelY = cv::cuda::createSobelFilter(CV_32FC1, CV_32FC1, 0, 1, 3);


    timer.start();
    for (int i = 0;i < ITR;++i) {
        cv::cuda::GpuMat canny_src(src_gray);
        cv::cuda::GpuMat canny_ROI = canny_src(cv::Rect(1, 1, 350, 350));
        gaussFilter->apply(canny_ROI, canny_ROI);
        canny->detect(canny_ROI, canny_ROI);
        canny_src.download(canny_dst); // ホストメモリに転送する
    }

    timer.stop();


    timer2.start();
    for (int i = 0;i < ITR;++i) {
        cv::Mat canny_src2 = src_gray.clone();
        cv::Mat canny_ROI2 = canny_src2(cv::Rect(1, 1, 350, 350));
        cv::GaussianBlur(canny_ROI2, canny_ROI2, cv::Size(5, 5), 0, 0, cv::BORDER_REPLICATE);
        cv::Canny(canny_ROI2, canny_ROI2, 50, 100, 3, true);
    }
    timer2.stop();

    timer3.start();
    for (int i = 0;i < ITR;++i) {
        cv::cuda::GpuMat sobel_src(src_gray), sobelX_gray, sobelY_gray;
        sobel_src.convertTo(sobelX_gray, CV_32FC1,1.0/255.0);
        sobelY_gray = sobelX_gray.clone();
        cv::cuda::GpuMat sobelX_ROI = sobelX_gray(cv::Rect(1, 1, 300, 300));
        cv::cuda::GpuMat sobelY_ROI = sobelY_gray(cv::Rect(1, 1, 300, 300));
        sobelX->apply(sobelX_ROI, sobelX_ROI);
        sobelY->apply(sobelY_ROI, sobelY_ROI);
        sobelX_gray.download(sobelX_dst); // ホストメモリに転送する
        sobelY_gray.download(sobelY_dst); // ホストメモリに転送する
    }

    timer3.stop();

    timer4.start();
    for (int i = 0;i < ITR;++i) {
        cv::Mat sobelX_gray2 = src_gray.clone();
        cv::Mat sobelY_gray2 = src_gray.clone();
        cv::Mat sobelX_gray2_converted;
        cv::Mat sobelY_gray2_converted;
        sobelX_gray2.convertTo(sobelX_gray2_converted, CV_32FC1,1.0/255.0);
        sobelY_gray2_converted = sobelX_gray2_converted.clone();
        cv::Mat sobelX_ROI2 = sobelX_gray2_converted(cv::Rect(1, 1, 300, 300));
        cv::Mat sobelY_ROI2 = sobelY_gray2_converted(cv::Rect(1, 1, 300, 300));
        cv::Sobel(sobelX_ROI2, sobelX_ROI2, CV_8UC1, 1, 0, 3);
        cv::Sobel(sobelY_ROI2, sobelY_ROI2, CV_8UC1, 0, 1, 3);
    }

    timer4.stop();

    cv::namedWindow("normal", cv::WINDOW_AUTOSIZE);
    cv::imshow("normal", canny_dst);
    //cv::namedWindow("sobelX", cv::WINDOW_AUTOSIZE);
    //cv::imshow("sobelX", sobelX_dst);
    //cv::namedWindow("sobelY", cv::WINDOW_AUTOSIZE);
    //cv::imshow("sobelY", sobelY_dst);

    char* managed_src;
    char* managed_dst;

    cudaMallocManaged(&managed_src, 1920 * 1080 * 3);
    cudaMallocManaged(&managed_dst, 300 * 300 * 3);

    cv::Mat cpu_mat_src(cv::Size(1920, 1080), CV_8UC3, managed_src);
    cv::Mat cpu_mat_dst(cv::Size(300, 300), CV_8UC3, managed_dst);
    cv::cuda::GpuMat gpu_mat_src(cv::Size(1920, 1080), CV_8UC3, managed_src);
    cv::cuda::GpuMat gpu_mat_dst(cv::Size(300, 300), CV_8UC3, managed_dst);

    std::cout << timer.MSec()/ITR << std::endl;
    std::cout << timer2.MSec()/ITR << std::endl;
    std::cout << timer3.MSec()/ITR << std::endl;
    std::cout << timer4.MSec()/ITR << std::endl;

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}