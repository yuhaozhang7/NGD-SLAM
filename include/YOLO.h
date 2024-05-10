#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "Tracking.h"

#ifndef YOLO_H
#define YOLO_H

namespace ORB_SLAM3
{

    class Tracking;

    class YOLO
    {
        public:
            YOLO(const float confThresholdIn, const float nmsThresholdIn, const int inpWidthIn, const int inpHeightIn,
                 const std::string& classesFileIn, const std::string& modelConfigurationIn, const std::string& modelWeightsIn, const std::string& netnameIn);
            ~YOLO() = default;

            void Run();
            cv::Mat Detect(const cv::Mat& imRGB, const cv::Mat& imDepth, const float maxBoxRatio);
            void SetTracker(Tracking* pTracker);
            void InsertInput(const cv::Mat& imRGB, const cv::Mat& imDepth);
            std::vector<cv::Mat> GetOutput();

            bool CheckNewInput();
            bool CheckNewOutput();
            void RequestFinish();
            bool CheckFinish();

            Tracking* mpTracker;
            bool mbIsYOLOInitialized = false;

            std::vector<cv::Mat> mInputPair;
            std::vector<cv::Mat> mOutputPair;

            std::mutex mInputMutex;
            std::mutex mOutputMutex;
            std::mutex mMutexFinish;

            bool mbFinishRequested;

        private:
            float confThreshold;
            float nmsThreshold;
            int inpWidth;
            int inpHeight;
            std::string netname;
            std::vector<std::string> classes;
            cv::dnn::Net net;
            cv::Mat postprocess(const cv::Mat& imRGB, const cv::Mat& imDepth, const std::vector<cv::Mat>& outs, const float maxBoxRatio);
    };

}// namespace ORB_SLAM

#endif // YOLO_H
