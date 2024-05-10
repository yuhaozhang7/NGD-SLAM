#include "YOLO.h"

namespace ORB_SLAM3
{

	YOLO::YOLO(const float confThresholdIn, const float nmsThresholdIn, const int inpWidthIn, const int inpHeightIn,
           	   const std::string& classesFileIn, const std::string& modelConfigurationIn, const std::string& modelWeightsIn, const std::string& netnameIn):
		   confThreshold(confThresholdIn), nmsThreshold(nmsThresholdIn), inpWidth(inpWidthIn), inpHeight(inpHeightIn),
		   mbIsYOLOInitialized(true), netname(netnameIn), net(cv::dnn::readNetFromDarknet(modelConfigurationIn, modelWeightsIn))
	{
		std::cout << "Net use " << netname << std::endl;

		// Load class names
		std::ifstream ifs(classesFileIn);
		std::string line;
		while (getline(ifs, line)) classes.push_back(line);

		// Set net preferences
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}

	cv::Mat YOLO::postprocess(const cv::Mat& imRGB, const cv::Mat& imDepth, const std::vector<cv::Mat>& outs, const float maxBoxRatio)   // Remove the bounding boxes with low confidence using non-maxima suppression
	{
		std::vector<int> classIds;
		std::vector<float> confidences;
		std::vector<cv::Rect> boxes;

		for (size_t i = 0; i < outs.size(); ++i)
		{
			// Scan through all the bounding boxes output from the network and keep only the
			// ones with high confidence scores. Assign the box's class label as the class
			// with the highest score for the box.
			float* data = (float*)outs[i].data;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
			{
				cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				cv::Point classIdPoint;
				double confidence;
				// Get the value and location of the maximum score
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
				if (confidence > this->confThreshold)
				{
					int centerX = (int)(data[0] * imRGB.cols);
					int centerY = (int)(data[1] * imRGB.rows);
					int width = (int)(data[2] * imRGB.cols);
					int height = (int)(data[3] * imRGB.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(cv::Rect(left, top, width, height));
				}
			}
		}

		// Perform non maximum suppression to eliminate redundant overlapping boxes with
		// lower confidences
		std::vector<int> indices;
		cv::Mat globalMask = cv::Mat::zeros(imRGB.rows, imRGB.cols, CV_8UC1);
		cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
		for (size_t i = 0; i < indices.size(); ++i)
		{
			int idx = indices[i];

			if (classIds[idx] == 0)
			{
				cv::Mat localMask = cv::Mat::zeros(imRGB.rows, imRGB.cols, CV_8UC1);
				cv::Rect box = boxes[idx];
				cv::Rect boxTight = box;
				
				float aspectRatio = static_cast<float>(box.width) / static_cast<float>(box.height);
				if (aspectRatio > maxBoxRatio)
				{
					float newWidth = box.width * 0.7f;
					float newHeight = box.height * 0.7f;
					float diffWidth = box.width - newWidth;
					float diffHeight = box.height - newHeight;

					boxTight.x += diffWidth / 2;
					boxTight.y += diffHeight / 2;
					boxTight.width = newWidth;
					boxTight.height = newHeight;
				}

				cv::Mat imDepthLocal = imDepth;

				// Make sure box doesn't exceed image bound
				boxTight.x = std::max(boxTight.x, 0);
				boxTight.y = std::max(boxTight.y, 0);
				boxTight.width = std::min(boxTight.width, imDepthLocal.cols - boxTight.x);
				boxTight.height = std::min(boxTight.height, imDepthLocal.rows - boxTight.y);

				cv::Mat imBoxTight = imDepthLocal(boxTight);

				// Compute median in box
				std::vector<float> validPixelsInBox;
				for (int y = 0; y < imBoxTight.rows; y++)
				{
					for (int x = 0; x < imBoxTight.cols; x++)
					{
						float val = imBoxTight.at<float>(y, x);
						if (val >= 0.05) validPixelsInBox.push_back(val);
					}
				}
				if(validPixelsInBox.size() / (float)(boxTight.width * boxTight.height) < 0.7) continue;

				std::sort(validPixelsInBox.begin(), validPixelsInBox.end());
				float median = validPixelsInBox[validPixelsInBox.size() / 2];
				float third = validPixelsInBox[validPixelsInBox.size() / 3];
				float value = (median + third) / 2.0;


				int expansion = 30;
				for (int y = 0; y < imDepthLocal.rows; ++y)
				{
					for (int x = 0; x < imDepthLocal.cols; ++x)
					{
						if (std::abs(imDepthLocal.at<float>(y, x) - value) <= 0.33 &&
							x >= box.x - expansion && x < box.x + box.width + expansion && 
							y >= box.y - expansion && y < box.y + box.height + expansion) localMask.at<uchar>(y, x) = 1;
					}
				}

				// ========== Find connected components ==========
				cv::Mat labels, stats, centroids;
				int nLabels = cv::connectedComponentsWithStats(localMask, labels, stats, centroids, 4, CV_32S);

				// Identify the largest component
				int largestLabel = 0;
				int largestArea = 0;
				for (int label = 1; label < nLabels; ++label)
				{
					int area = stats.at<int>(label, cv::CC_STAT_AREA);
					if (area > largestArea)
					{
						largestArea = area;
						largestLabel = label;
					}
				}

				for (int y = 0; y < labels.rows; ++y)
				{
					for (int x = 0; x < labels.cols; ++x)
					{
						if (labels.at<int>(y, x) == largestLabel) globalMask.at<uchar>(y, x) = 1;
					}
				}
				// ===============================================

			}
		}

		int dilationSize = 3;
		cv::Mat dilationElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilationSize + 1, 2 * dilationSize + 1), cv::Point(dilationSize, dilationSize));
		cv::dilate(globalMask, globalMask, dilationElement);

		int totalPixels = globalMask.rows * globalMask.cols;
		int countOne = cv::countNonZero(globalMask);
		if ((double)countOne / totalPixels > 0.8) globalMask = cv::Mat::zeros(imRGB.rows, imRGB.cols, CV_8UC1); 

		return globalMask;
	}

	cv::Mat YOLO::Detect(const cv::Mat& imRGB, const cv::Mat& imDepth, const float maxBoxRatio)
	{
		cv::Mat blob;
		cv::dnn::blobFromImage(imRGB, blob, 1 / 255.0, cv::Size(this->inpWidth, this->inpHeight), cv::Scalar(0, 0, 0), true, false);
		this->net.setInput(blob);
		std::vector<cv::Mat> outs;
		this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
		cv::Mat mask = this->postprocess(imRGB, imDepth, outs, maxBoxRatio);

		return mask;
	}

	void YOLO::SetTracker(Tracking *pTracker)
	{
		mpTracker=pTracker;
	}

	bool YOLO::CheckNewInput()
	{
		std::unique_lock<std::mutex> lock(mInputMutex);
		return(!mInputPair.empty());
	}

	bool YOLO::CheckNewOutput()
	{
		std::unique_lock<std::mutex> lock(mOutputMutex);
		return(!mOutputPair.empty());
	}

	void YOLO::InsertInput(const cv::Mat& imRGB, const cv::Mat& imDepth)
	{
		std::unique_lock<std::mutex> lock(mInputMutex);
		mInputPair.clear();
		mInputPair.push_back(imRGB);
		mInputPair.push_back(imDepth);
	}

	std::vector<cv::Mat> YOLO::GetOutput()
	{
		std::unique_lock<std::mutex> lock(mOutputMutex);
		std::vector<cv::Mat> returnedOutput;
		if (mOutputPair.size() == 2)
		{
			returnedOutput = std::move(mOutputPair);
			mOutputPair.clear();
		}
		return returnedOutput;
	}

	void YOLO::Run()
	{
		int counter = 0;
		while(1)
		{
			std::unique_lock<std::mutex> inputLock(mInputMutex);
			if(mInputPair.size() == 2)
			{
				cv::Mat imRGB = mInputPair[0];
				cv::Mat imDepth = mInputPair[1];
				mInputPair.clear();
				inputLock.unlock();

				cv::Mat imMask = Detect(imRGB, imDepth, 0);
				cv::Mat imGray;
				cv::cvtColor(imRGB, imGray, cv::COLOR_BGR2GRAY);

				std::unique_lock<std::mutex> outputLock(mOutputMutex);
				mOutputPair.clear();
				mOutputPair.push_back(imGray);
				mOutputPair.push_back(imMask);
			}
			else
			{
				inputLock.unlock();
				// std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}

			if(CheckFinish()) break;
		}
	}

	void YOLO::RequestFinish()
	{
		unique_lock<mutex> lock(mMutexFinish);
		mbFinishRequested = true;
	}

	bool YOLO::CheckFinish()
	{
		unique_lock<mutex> lock(mMutexFinish);
		return mbFinishRequested;
	}

} //namespace ORB_SLAM
