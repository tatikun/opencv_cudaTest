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
#include <iomanip>

#include "cuda_runtime.h"
#include "myTimer.h"


std::string getDatetimeStr() {
    time_t t = time(nullptr);
    errno_t error;
    struct tm localTime;
    error = localtime_s(&localTime, &t);
    std::stringstream s;
    s << "20" << localTime.tm_year - 100;
    // setw(),setfill()で0詰め
    s << std::setw(2) << std::setfill('0') << localTime.tm_mon + 1;
    s << std::setw(2) << std::setfill('0') << localTime.tm_mday;
    s << std::setw(2) << std::setfill('0') << localTime.tm_hour;
    s << std::setw(2) << std::setfill('0') << localTime.tm_min;
    s << std::setw(2) << std::setfill('0') << localTime.tm_sec;
    // std::stringにして値を返す
    return s.str();
}


int main()
{
    std::ofstream out(getDatetimeStr()+"_out.csv");

    const int ITR = 3000;
    const int ROI_X = 0;
    const int ROI_Y = 0;
    const int ROI_W = 300;
    const int ROI_H = 300;
    MyTimer timer,timer2,timer3,timer4,timer5,timer6;
    //std::cout << cv::getBuildInformation() << std::endl;
    //cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

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
        cv::cuda::GpuMat canny_ROI = canny_src(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        gaussFilter->apply(canny_ROI, canny_ROI);
        canny->detect(canny_ROI, canny_ROI);
        canny_src.download(canny_dst); // ホストメモリに転送する
    }

    timer.stop();


    timer2.start();
    for (int i = 0;i < ITR;++i) {
        cv::Mat canny_src2 = src_gray.clone();
        cv::Mat canny_ROI2 = canny_src2(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        cv::GaussianBlur(canny_ROI2, canny_ROI2, cv::Size(5, 5), 0, 0, cv::BORDER_REPLICATE);
        cv::Canny(canny_ROI2, canny_ROI2, 50, 100, 3, true);
    }
    timer2.stop();

    cv::cuda::Stream stream[2];
    timer3.start();
    for (int i = 0;i < ITR;++i) {
        cv::cuda::GpuMat sobel_src(src_gray), sobelX_gray, sobelY_gray;

        sobel_src.convertTo(sobelX_gray, CV_32FC1, 1.0 / 255.0, stream[0]);
        sobel_src.convertTo(sobelY_gray, CV_32FC1,1.0/255.0,stream[1]);
        cv::cuda::GpuMat sobelX_ROI = sobelX_gray(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        cv::cuda::GpuMat sobelY_ROI = sobelY_gray(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        //sobelY_gray = sobelX_gray.clone();
        sobelX->apply(sobelX_ROI, sobelX_ROI,stream[0]);
        sobelY->apply(sobelY_ROI, sobelY_ROI,stream[1]);
        sobelX_gray.download(sobelX_dst,stream[0]); // ホストメモリに転送する
        sobelY_gray.download(sobelY_dst,stream[1]); // ホストメモリに転送する
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
        cv::Mat sobelX_ROI2 = sobelX_gray2_converted(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        cv::Mat sobelY_ROI2 = sobelY_gray2_converted(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        cv::Sobel(sobelX_ROI2, sobelX_ROI2, CV_32FC1, 1, 0, 3);
        cv::Sobel(sobelY_ROI2, sobelY_ROI2, CV_32FC1, 0, 1, 3);
    }

    timer4.stop();

    cv::cuda::Stream stream2[3];
    timer5.start();
    for (int i = 0;i < ITR;++i) {
        cv::cuda::GpuMat sobel_src(src_gray), sobelX_gray, sobelY_gray;
        cv::cuda::GpuMat canny_src = sobel_src.clone();

        sobel_src.convertTo(sobelX_gray, CV_32FC1, 1.0 / 255.0, stream2[0]);
        sobel_src.convertTo(sobelY_gray, CV_32FC1, 1.0 / 255.0, stream2[1]);
        cv::cuda::GpuMat sobelX_ROI = sobelX_gray(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        cv::cuda::GpuMat sobelY_ROI = sobelY_gray(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        cv::cuda::GpuMat canny_ROI = canny_src(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));

        //sobelY_gray = sobelX_gray.clone();
        sobelX->apply(sobelX_ROI, sobelX_ROI, stream2[0]);
        sobelY->apply(sobelY_ROI, sobelY_ROI, stream2[1]);
        sobelX_gray.download(sobelX_dst, stream2[0]); // ホストメモリに転送する
        sobelY_gray.download(sobelY_dst, stream2[1]); // ホストメモリに転送する
        
        gaussFilter->apply(canny_ROI, canny_ROI,stream2[2]);
        canny->detect(canny_ROI, canny_ROI,stream2[2]);
        canny_src.download(canny_dst,stream2[2]); // ホストメモリに転送する
    }

    timer5.stop();

/*
    char* managed_canny_src;
    int width = src_gray.cols;
    int height = src_gray.rows;
    cudaMallocManaged(&managed_canny_src, width * height);
    cv::Mat cpu_mat_src(cv::Size(width, height), CV_8UC1, managed_canny_src);
    cv::cuda::GpuMat gpu_mat_src(cv::Size(width, height), CV_8UC1, managed_canny_src);
    managed_canny_src = (char*)malloc(sizeof(uchar) * width * height);
    timer5.start();
    for (int i = 0;i < ITR;++i) {
        cv::cuda::GpuMat a(src_gray);
        cv::cuda::GpuMat canny_ROI2 = a(cv::Rect(1, 1, 350, 350));
        gaussFilter->apply(canny_ROI2, canny_ROI2);
        canny->detect(canny_ROI2, canny_ROI2);
        a.copyTo(gpu_mat_src);
    }
    timer5.stop();
*/

    out << "ITR," << ITR << std::endl;
    out << "W:" << ROI_W << "," << "H:" << ROI_H << std::endl;
    out << "canny(GPU)," << timer.MSec() / ITR << std::endl;
    out << "canny(CPU)," << timer2.MSec() / ITR << std::endl;
    out << "sobel(GPU)," << timer3.MSec()/ITR << std::endl;
    out << "sobel(CPU)," << timer4.MSec() / ITR << std::endl;
    out << "stream(GPU)," << timer5.MSec() / ITR << std::endl;


    cv::namedWindow("normal", cv::WINDOW_AUTOSIZE);
    cv::imshow("normal", canny_dst);
    //cv::imshow("a", cpu_mat_src);
    //cv::namedWindow("sobelX", cv::WINDOW_AUTOSIZE);
    //cv::imshow("sobelX", sobelX_dst);
    //cv::namedWindow("sobelY", cv::WINDOW_AUTOSIZE);
    //cv::imshow("sobelY", sobelY_dst);

    cv::imwrite("out.png", canny_dst);
    cv::waitKey(0);
    cv::destroyAllWindows();
    //cudaFree(managed_canny_src);

    return 0;
}