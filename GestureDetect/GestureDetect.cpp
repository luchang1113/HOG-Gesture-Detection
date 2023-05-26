#include <iostream>
#include <fstream>
#include<filesystem>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
using namespace cv;
using namespace std;

#define ENABLE_TRAIN 1
#define TRAIN_DATA_SIZE 40
#define USE_WEBCAM 0

const char* labels[4] = { "A","C","Five","V" };

Mat fvector2fmat(vector<float> output) {
	Mat out_result(1, output.size(), CV_32FC1, cv::Scalar(0));
	memcpy(out_result.data, output.data(), output.size() * sizeof(float));
	return out_result;
}

//提取图像HOG特征
void get_hog_descriptors(Mat img, vector<float>& fv)
{
	Size winSize(96, 96);
	Size blockSize(16, 16);
	Size blockStride(8, 8);
	Size cellSize(8, 8);
	int nbins = 9;
	int derivAperture = 1;
	double winSigma = 4.;
	int histogramNormType = 0;
	double L2HysThreshold = 2.0000000000000001e-01;
	bool gammaCorrection = false;
	int nlevels = 64;

	normalize(img, img, 0, 1, NORM_MINMAX, CV_32F);
	img.convertTo(img, CV_8U, 255, 0);
	resize(img, img, Size(96, 96), 0, 0, INTER_AREA);
	//imshow("HOG", img);
	//waitKey(0);

	HOGDescriptor hog;
	hog = HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
		HOGDescriptor::L2Hys, L2HysThreshold, gammaCorrection);
	hog.compute(img, fv, Size(0, 0), Size(0, 0));
}

//生成训练集
void generate_dataset(Mat& train_data, Mat& train_label)
{
	vector<string> images;
	vector<vector<float>> vecDec;
	vector<float> fv;
	Mat hogv;
	glob("./Hand/A", images);
	for (int i = 0; i < TRAIN_DATA_SIZE; i++)
	{
		Mat image = imread(images[i].c_str(), cv::IMREAD_GRAYSCALE);
		get_hog_descriptors(image, fv);
		printf("image path : %s, feature data length: %lld\r\n", images[i].c_str(), fv.size());
		hogv = fvector2fmat(fv);
		train_data.push_back(hogv);
		train_label.push_back(1);
	}
	images.clear();
	glob("./Hand/C", images);
	for (int i = 0; i < TRAIN_DATA_SIZE; i++)
	{
		Mat image = imread(images[i].c_str(), cv::IMREAD_GRAYSCALE);
		get_hog_descriptors(image, fv);
		printf("image path : %s, feature data length: %lld\r\n", images[i].c_str(), fv.size());
		hogv = fvector2fmat(fv);
		train_data.push_back(hogv);
		train_label.push_back(2);
	}
	images.clear();
	glob("./Hand/Five", images);
	for (int i = 0; i < TRAIN_DATA_SIZE; i++)
	{
		Mat image = imread(images[i].c_str(), cv::IMREAD_GRAYSCALE);
		get_hog_descriptors(image, fv);
		printf("image path : %s, feature data length: %lld\r\n", images[i].c_str(), fv.size());
		hogv = fvector2fmat(fv);
		train_data.push_back(hogv);
		train_label.push_back(3);
	}
	images.clear();
	glob("./Hand/V", images);
	for (int i = 0; i < TRAIN_DATA_SIZE; i++)
	{
		Mat image = imread(images[i].c_str(), cv::IMREAD_GRAYSCALE);
		get_hog_descriptors(image, fv);
		printf("image path : %s, feature data length: %lld\r\n", images[i].c_str(), fv.size());
		hogv = fvector2fmat(fv);
		train_data.push_back(hogv);
		train_label.push_back(4);
	}

}

void svm_train(Mat& trainData, Mat& labels) {
	printf("\n start SVM training... \n");
	Ptr< ml::SVM > svm = ml::SVM::create();
	/* Default values to train SVM */
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setC(2.67);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));

	printf("Data row:%d col:%d Class row:%d\r\n", trainData.rows, trainData.cols, labels.rows);


	svm->train(trainData, ml::ROW_SAMPLE, labels);
	clog << "...[done]" << endl;

	// save xml
	svm->save("./hog_elec.yml");
}

void main()
{
#if ENABLE_TRAIN
	Mat trainData;
	Mat trainLabel;
	generate_dataset(trainData, trainLabel);
	printf("data size:%d,%d\r\n", trainData.rows, trainData.cols);
	svm_train(trainData, trainLabel);
#endif
#if USE_WEBCAM
	VideoCapture cap(1);
	Ptr<ml::SVM> mySVM = Algorithm::load<ml::SVM>("./hog_elec.yml");
	Mat frame;

	while (1)

	{
		cap >> frame;

		frame = frame(Range(140, 340), Range(300, 500));
		flip(frame, frame, 1);


		resize(frame, frame, Size(96, 96), 0, 0, INTER_AREA);
		cvtColor(frame, frame, COLOR_RGB2GRAY);

		//equalizeHist(frame, frame);

		imshow("webcam", frame);
		vector<float> fv;
		get_hog_descriptors(frame, fv);
		Mat one_row = Mat::zeros(Size(fv.size(), 1), CV_32F);
		for (int i = 0; i < fv.size(); i++) {
			one_row.at<float>(0, i) = fv[i];
		}
		float label = mySVM->predict(one_row);
		cout << labels[(int)label - 1] << endl;

		waitKey(200);
	}

	return;

#else
	vector<string> images;
	glob("./Hand/V", images);
	int cNum = images.size();
	Ptr<ml::SVM> mySVM = Algorithm::load<ml::SVM>("./hog_elec.yml");
	for (int i = TRAIN_DATA_SIZE; i < cNum; i++)
	{
		Mat image = imread(images[i].c_str(), cv::IMREAD_GRAYSCALE);

		resize(image, image, Size(96, 96), 0, 0, INTER_AREA);
		equalizeHist(image, image);

		vector<float> fv;
		get_hog_descriptors(image, fv);
		Mat one_row = Mat::zeros(Size(fv.size(), 1), CV_32F);
		for (int i = 0; i < fv.size(); i++) {
			one_row.at<float>(0, i) = fv[i];
		}
		float label = mySVM->predict(one_row);
		cout << labels[(int)label - 1] << endl;
	}
#endif
}