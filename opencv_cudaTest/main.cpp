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

#include<stdio.h>
#include <omp.h>

#include <cuda_runtime.h>
#include "myTimer.h"
#include "MyUtils.h"




int main()
{
    const std::string outdataName = "_out648.csv"; //�o�͂���csv�t�@�C���̖��O
    const std::string inputImgName = "imgs/1.png"; //���͂���摜�̃p�X
    const std::string inputImgName2 = "imgs/2.png"; //���͂���摜�̃p�X
    const int ITR = 6000; //�J��Ԃ����s��
    const int ROI_X = 0; //ROI�̍�����W
    const int ROI_Y = 0;
    const int ROI_W = 648; //ROI�̃T�C�Y
    const int ROI_H = 488;
    MyTimer timer[6];

    std::ofstream out(myutils::getDatetimeStr() + outdataName);


    cv::Mat src = cv::imread(inputImgName, 0);
    cv::Mat src2 = cv::imread(inputImgName2, 0);
    cv::Mat canny_dst;
    cv::Mat sobelX_dst;
    cv::Mat sobelY_dst;

    if (src.empty())
    {
        std::cerr << "can't load src image" << std::endl;
        return -1;
    }

    //CannyEdgeDetector�̒�`
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(50, 100, 3, true);
    cv::Ptr<cv::cuda::Filter> gaussFilter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(5, 5), 0, 0, cv::BORDER_REPLICATE);

    //SobelFilter�̒�`�Acuda�ł�Filter�N���X�ŊǗ�����Ă���
    cv::Ptr<cv::cuda::Filter> sobelX = cv::cuda::createSobelFilter(CV_32FC1, CV_32FC1, 1, 0, 3);
    cv::Ptr<cv::cuda::Filter> sobelY = cv::cuda::createSobelFilter(CV_32FC1, CV_32FC1, 0, 1, 3);

    //std::cout << cv::getBuildInformation() << std::endl;
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    std::cout << "canny(GPU)�v����..." << std::endl;
    timer[0].start();
    for (int i = 0; i < ITR; ++i) {
        cv::cuda::GpuMat canny_src(src);
        cv::cuda::GpuMat canny_ROI = canny_src(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        gaussFilter->apply(canny_ROI, canny_ROI);
        canny->detect(canny_ROI, canny_ROI);
        canny_src.download(canny_dst); // �z�X�g�������ɓ]������
    }
    timer[0].stop();

    std::cout << "canny(CPU)�v����..." << std::endl;
    timer[1].start();
    for (int i = 0; i < ITR; ++i) {
        cv::Mat canny_src = src.clone();
        cv::Mat canny_ROI = canny_src(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        cv::GaussianBlur(canny_ROI, canny_ROI, cv::Size(5, 5), 0, 0, cv::BORDER_REPLICATE);
        cv::Canny(canny_ROI, canny_ROI, 50, 100, 3, true);
    }
    timer[1].stop();

    std::cout << "sobel(GPU)�v����..." << std::endl;
    cv::cuda::Stream stream[2];
    timer[2].start();
    for (int i = 0; i < ITR; ++i) {
        cv::cuda::GpuMat sobel_src(src), sobelX_gray, sobelY_gray;

        sobel_src.convertTo(sobelX_gray, CV_32FC1, 1.0 / 255.0, stream[0]);
        sobel_src.convertTo(sobelY_gray, CV_32FC1, 1.0 / 255.0, stream[1]);
        cv::cuda::GpuMat sobelX_ROI = sobelX_gray(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        cv::cuda::GpuMat sobelY_ROI = sobelY_gray(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        //sobelY_gray = sobelX_gray.clone();
        sobelX->apply(sobelX_ROI, sobelX_ROI, stream[0]);
        sobelY->apply(sobelY_ROI, sobelY_ROI, stream[1]);
        sobelX_gray.download(sobelX_dst, stream[0]); // �z�X�g�������ɓ]������
        sobelY_gray.download(sobelY_dst, stream[1]); // �z�X�g�������ɓ]������
    }
    timer[2].stop();

#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
        {
            cv::Mat canny_src = src.clone();
            cv::Mat canny_ROI = canny_src(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
            cv::GaussianBlur(canny_ROI, canny_ROI, cv::Size(5, 5), 0, 0, cv::BORDER_REPLICATE);
            cv::Canny(canny_ROI, canny_ROI, 50, 100, 3, true);
            ;
        }
#pragma omp section
        {
            cv::Mat sobelX_gray2 = src.clone();
            cv::Mat sobelY_gray2 = src.clone();
            cv::Mat sobelX_gray2_converted;
            cv::Mat sobelY_gray2_converted;
            sobelX_gray2.convertTo(sobelX_gray2_converted, CV_32FC1, 1.0 / 255.0);
            sobelY_gray2_converted = sobelX_gray2_converted.clone();
            cv::Mat sobelX_ROI2 = sobelX_gray2_converted(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
            cv::Mat sobelY_ROI2 = sobelY_gray2_converted(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
            cv::Sobel(sobelX_ROI2, sobelX_ROI2, CV_32FC1, 1, 0, 3);
            cv::Sobel(sobelY_ROI2, sobelY_ROI2, CV_32FC1, 0, 1, 3);
            ;
        }
    }
    /*
    * Stream���g��Ȃ�������
    std::cout << "sobel(GPU)�v����..." << std::endl;
    timer[2].start();
    for (int i = 0; i < ITR; ++i) {
        cv::cuda::GpuMat sobel_src(src), sobelX_gray, sobelY_gray;

        sobel_src.convertTo(sobelX_gray, CV_32FC1, 1.0 / 255.0);
        sobelY_gray = sobelX_gray.clone();
        cv::cuda::GpuMat sobelX_ROI = sobelX_gray(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        cv::cuda::GpuMat sobelY_ROI = sobelY_gray(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        sobelX->apply(sobelX_ROI, sobelX_ROI);
        sobelY->apply(sobelY_ROI, sobelY_ROI);
        sobelX_gray.download(sobelX_dst); // �z�X�g�������ɓ]������
        sobelY_gray.download(sobelY_dst); // �z�X�g�������ɓ]������
    }
    timer[2].stop();
    */

    std::cout << "sobel(CPU)�v����..." << std::endl;
    timer[3].start();
    for (int i = 0;i < ITR;++i) {
        cv::Mat sobelX_gray2 = src.clone();
        cv::Mat sobelY_gray2 = src.clone();
        cv::Mat sobelX_gray2_converted;
        cv::Mat sobelY_gray2_converted;
        sobelX_gray2.convertTo(sobelX_gray2_converted, CV_32FC1,1.0/255.0);
        sobelY_gray2_converted = sobelX_gray2_converted.clone();
        cv::Mat sobelX_ROI2 = sobelX_gray2_converted(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        cv::Mat sobelY_ROI2 = sobelY_gray2_converted(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        cv::Sobel(sobelX_ROI2, sobelX_ROI2, CV_32FC1, 1, 0, 3);
        cv::Sobel(sobelY_ROI2, sobelY_ROI2, CV_32FC1, 0, 1, 3);
    }
    timer[3].stop();

    //canny��sobel��GPU�ŕ��񏈗�����e�X�g����
    std::cout << "canny+sobel(GPUstream)�v����..." << std::endl;
    cv::cuda::Stream stream2[3];
    timer[4].start();
    for (int i = 0;i < ITR;++i) {
        cv::cuda::GpuMat sobel_src(src), sobelX_gray, sobelY_gray;
        cv::cuda::GpuMat canny_src = sobel_src.clone();

        sobel_src.convertTo(sobelX_gray, CV_32FC1, 1.0 / 255.0, stream2[0]);
        sobel_src.convertTo(sobelY_gray, CV_32FC1, 1.0 / 255.0, stream2[1]);
        cv::cuda::GpuMat sobelX_ROI = sobelX_gray(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        cv::cuda::GpuMat sobelY_ROI = sobelY_gray(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        cv::cuda::GpuMat canny_ROI = canny_src(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));

        //sobelY_gray = sobelX_gray.clone();
        sobelX->apply(sobelX_ROI, sobelX_ROI, stream2[0]);
        sobelY->apply(sobelY_ROI, sobelY_ROI, stream2[1]);

        gaussFilter->apply(canny_ROI, canny_ROI, stream2[2]);
        canny->detect(canny_ROI, canny_ROI, stream2[2]);

        sobelX_gray.download(sobelX_dst, stream2[0]); // �z�X�g�������ɓ]������
        sobelY_gray.download(sobelY_dst, stream2[1]); // �z�X�g�������ɓ]������
        canny_src.download(canny_dst,stream2[2]); // �z�X�g�������ɓ]������
    }
    timer[4].stop();

/*
    //Shared Memory���g���ĂȂ񂩂��悤�Ƃ������Ǎ���

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
    cv::cuda::HostMem srcMem;
    cv::cuda::HostMem src2Mem;
    cv::cuda::HostMem inputMem;
    cv::cuda::HostMem outputMemx;
    cv::cuda::HostMem outputMemy;
    cv::Mat outputTest;
    srcMem = cv::cuda::HostMem(src.size(), CV_8UC1, cv::cuda::HostMem::AllocType::PAGE_LOCKED);
    src2Mem = cv::cuda::HostMem(src.size(), CV_8UC1, cv::cuda::HostMem::AllocType::PAGE_LOCKED);
    outputMemx = cv::cuda::HostMem(src.size(), CV_32FC1, cv::cuda::HostMem::AllocType::PAGE_LOCKED);
    outputMemy = cv::cuda::HostMem(src.size(), CV_32FC1, cv::cuda::HostMem::AllocType::PAGE_LOCKED);
    inputMem = cv::cuda::HostMem(src.size(), CV_32FC1, cv::cuda::HostMem::AllocType::PAGE_LOCKED);
    cv::cuda::GpuMat sobelTmp,sx,sy;
    cv::Mat srcMemMat;
    cv::Mat src2MemMat;
    srcMemMat = srcMem.createMatHeader();
    src2MemMat = src2Mem.createMatHeader();
    cv::Mat outputMemMat;
    outputMemMat = outputMemx.createMatHeader();
    src.copyTo(srcMemMat);
    src2.copyTo(src2MemMat);

    cv::cuda::GpuMat cannyTmp;


    std::cout << "sobel(GPU)2�v����..." << std::endl;
    cv::cuda::Stream streams[3];
    MyTimer atimer;
    atimer.start();
    for (int i = 0; i < ITR; ++i) {
        if (i % 2 == 0) {
            sobelTmp.upload(srcMem, streams[0]);
        }
        else {
            sobelTmp.upload(src2Mem, streams[0]);
        }
        stream[0].waitForCompletion();
        cannyTmp = sobelTmp.clone();
        sobelTmp.convertTo(sx, CV_32FC1, 1.0 / 255.0, streams[0]);
        sobelTmp.convertTo(sy, CV_32FC1, 1.0 / 255.0, streams[1]);
        cv::cuda::GpuMat sobelX_ROI = sx(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        cv::cuda::GpuMat sobelY_ROI = sy(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        //sobelY_gray = sobelX_gray.clone();
        sobelX->apply(sobelX_ROI, sobelX_ROI, streams[0]);
        sobelY->apply(sobelY_ROI, sobelY_ROI, streams[1]);
        sx.download(outputMemx, streams[0]); // �z�X�g�������ɓ]������
        sy.download(outputMemy, streams[1]); // �z�X�g�������ɓ]������
     
        if (i % 2 == 0) {
            cannyTmp.upload(srcMem, streams[2]);
        }
        else {
            cannyTmp.upload(src2Mem, streams[2]);
        }
        cv::cuda::GpuMat canny_ROI = cannyTmp(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
        gaussFilter->apply(canny_ROI, canny_ROI,streams[2]);
        canny->detect(canny_ROI, canny_ROI,streams[2]);
        cannyTmp.download(canny_dst,streams[2]); // �z�X�g�������ɓ]������
        stream->waitForCompletion();
    }
    atimer.stop();

    //cv::cuda::GpuMat cannyTmp;
    //std::cout << "canny(GPU)�v����..." << std::endl;
    //MyTimer btimer;
    //btimer.start();
    //for (int i = 0; i < ITR; ++i) {
    //    cannyTmp.upload(srcMem, streams[0]);
    //    cv::cuda::GpuMat canny_ROI = cannyTmp(cv::Rect(ROI_X, ROI_Y, ROI_W, ROI_H));
    //    gaussFilter->apply(canny_ROI, canny_ROI);
    //    canny->detect(canny_ROI, canny_ROI);
    //    cannyTmp.download(canny_dst); // �z�X�g�������ɓ]������
    //}
   // btimer.stop();

    out << "ITR," << ITR << std::endl;
    out << "ROI_W:" << ROI_W << "," << "ROI_H:" << ROI_H << std::endl;
    out << "canny(GPU)," << timer[0].MSec() / ITR << std::endl;
    out << "canny(CPU)," << timer[1].MSec() / ITR << std::endl;
    out << "sobel(GPU)," << timer[2].MSec() << std::endl;
    out << "sobel(CPU)," << timer[3].MSec() / ITR << std::endl;
    out << "stream(GPU)," << timer[4].MSec() << std::endl;
    out << "sobel(GPU)2," << atimer.MSec()  << std::endl;
   // out << "canny(GPU)2," << btimer.MSec() / ITR << std::endl;


    cv::Mat testMat;
    outputMemMat.convertTo(testMat, CV_8UC1, 255);
    //cv::namedWindow("normal", cv::WINDOW_AUTOSIZE);
    cv::imshow("normal", testMat);

    cv::waitKey(0);
    cv::destroyAllWindows();
    //cudaFree(managed_canny_src);

    std::cout << "�v���I��" << std::endl;

    return 0;
}