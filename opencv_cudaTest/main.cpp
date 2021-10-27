#include <opencv2/core.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

int main(int argc, char* argv[])
{
    cv::Mat src = cv::imread("room_living.png", cv::IMREAD_UNCHANGED), dst;
    if (src.empty())
    {
        return -1;
    }

    cv::cuda::GpuMat d_src(src), d_dst;
    cv::cuda::cvtColor(d_src, d_dst, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(50, 100);
    cv::cuda::GpuMat edge;
    cv::cuda::GpuMat roi = d_dst(cv::Rect(1, 1, 200, 200));
    canny->detect(roi, roi);
    //canny->detect(d_dst, edge);

    d_dst.download(dst); // ÉzÉXÉgÉÅÉÇÉäÇ…ì]ëóÇ∑ÇÈ
    cv::namedWindow("normal", cv::WINDOW_AUTOSIZE);
    cv::imshow("normal", dst);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}