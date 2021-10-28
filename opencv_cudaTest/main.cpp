#include <opencv2/core.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include "myTimer.h"

int main()
{
    MyTimer timer,timer2,timer3;
    std::cout << cv::getBuildInformation() << std::endl;
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    cv::Mat src = cv::imread("room_living.png", cv::IMREAD_UNCHANGED);
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
    cv::Ptr<cv::cuda::Filter> sobelX = cv::cuda::createSobelFilter(CV_8UC1, CV_8UC1, 1, 0, 3);
    cv::Ptr<cv::cuda::Filter> sobelY = cv::cuda::createSobelFilter(CV_8UC1, CV_8UC1, 0, 1, 3);

    timer.start();
    cv::cuda::GpuMat canny_src(src), canny_gray;
    cv::cuda::cvtColor(canny_src, canny_gray, cv::COLOR_BGR2GRAY);
    cv::cuda::GpuMat canny_ROI = canny_gray(cv::Rect(1, 1, 350, 350));
    gaussFilter->apply(canny_ROI, canny_ROI);
    canny->detect(canny_ROI, canny_ROI);
    canny_gray.download(canny_dst); // ホストメモリに転送する
    timer.stop();

    cv::Mat canny_src2;
    cv::cvtColor(src, canny_src2, cv::COLOR_BGR2GRAY);
    timer2.start();
    cv::Mat canny_ROI2 = canny_src2(cv::Rect(1, 1, 350, 350));
    cv::GaussianBlur(canny_ROI2, canny_ROI2, cv::Size(5, 5), 0, 0, cv::BORDER_REPLICATE);
    cv::Canny(canny_ROI2, canny_ROI2, 50, 100,3,true);
    timer2.stop();

    timer3.start();
    cv::cuda::GpuMat sobel_src(src), sobelX_gray, sobelY_gray;
    cv::cuda::cvtColor(sobel_src, sobelX_gray, cv::COLOR_BGR2GRAY);
    sobelY_gray = sobelX_gray.clone();
    cv::cuda::GpuMat sobelX_ROI = sobelX_gray(cv::Rect(1, 1, 300, 300));
    cv::cuda::GpuMat sobelY_ROI = sobelY_gray(cv::Rect(1, 1, 300, 300));
    sobelX->apply(sobelX_ROI, sobelX_ROI);
    sobelY->apply(sobelY_ROI, sobelY_ROI);

    sobelX_gray.download(sobelX_dst); // ホストメモリに転送する
    sobelY_gray.download(sobelY_dst); // ホストメモリに転送する
    timer3.stop();


    cv::namedWindow("normal", cv::WINDOW_AUTOSIZE);
    cv::imshow("normal", canny_dst);
    cv::namedWindow("sobelX", cv::WINDOW_AUTOSIZE);
    cv::imshow("sobelX", sobelX_dst);
    cv::namedWindow("sobelY", cv::WINDOW_AUTOSIZE);
    cv::imshow("sobelY", sobelY_dst);


    std::cout << timer.MSec() << std::endl;
    std::cout << timer2.MSec() << std::endl;
    std::cout << timer3.MSec() << std::endl;

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}