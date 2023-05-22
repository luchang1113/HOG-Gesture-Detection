#include <iostream>
#include <fstream>
#include<filesystem>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
using namespace cv;
using namespace std;

#define ENABLE_TRAIN 0
#define PRE_PROCESS_LEVEL 1
#define TRAIN_DATA_SIZE 40

const char* labels[4] = { "V","A","Five","C" };

Mat compensate(Mat src, Mat bg) {
	Mat dst = Mat::zeros(src.size(), src.type());
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			double k;
			uint8_t y;
			uint8_t xb = bg.at<uint8_t>(i, j);
			uint8_t x = src.at<uint8_t>(i, j);
			if (xb < 20) {
				k = 2.5;
			}
			else if (xb <= 100 && xb >= 20) {
				k = 1 + 1.5 * (100 - xb) / 80;
			}
			else if (xb < 200 && xb > 100) {
				k = 1.0;
			}
			else {
				k = 1 + 1.0 * (xb - 200) / 35;
			}
			if (xb > x) {
				y = 255 - k * (xb - x);
				if (y < 255 * 0.75) { y = 255 * 0.75; }
			}
			else {
				y = 255;
			}
			dst.at<uint8_t>(i, j) = y;
		}
	}
	return dst;
}
Mat get_background(Mat src, int win_2)
{
	int winsize = 2 * win_2 + 1;
	Mat src_tmp;
	//为了使原图所有点都能取到完整的矩形邻域，首先对原图进行边缘填充
	copyMakeBorder(src, src_tmp, win_2, win_2, win_2, win_2, BORDER_CONSTANT, Scalar(0));
	//中值滤波去噪，减少背景杂志
	medianBlur(src_tmp, src_tmp, 7);

	Mat dst(src.size(), CV_8UC1);
	for (int i = win_2; i < src_tmp.rows - win_2; i++)
	{
		uchar* pd = dst.ptr<uchar>(i - win_2);
		for (int j = win_2; j < src_tmp.cols - win_2; j++)
		{
			Mat tmp;
			//截取每一个点周围的矩形邻域
			src_tmp(Rect(j - win_2, i - win_2, winsize, winsize)).copyTo(tmp);
			//将二维矩阵转换成一维数据
			tmp.reshape(1, 1).copyTo(tmp);
			//从大到小排序
			cv::sort(tmp, tmp, SORT_EVERY_ROW | SORT_ASCENDING);

			uchar* p = tmp.ptr<uchar>(0);
			//取排序之后的前5个像素值计算均值作为背景值
			pd[j - win_2] = (uchar)((p[tmp.cols - 1] + p[tmp.cols - 2] + p[tmp.cols - 3] + p[tmp.cols - 4] + p[tmp.cols - 5]) * 0.2);
		}
	}
	return dst;
}

Mat pre_process_img(Mat img)
{
#if PRE_PROCESS_LEVEL == 1
	Mat YCrCb_img;
	cvtColor(img, YCrCb_img, COLOR_RGB2YCrCb);
	Mat channels[3];
	split(YCrCb_img, channels);

	cvtColor(img, img, COLOR_BGR2GRAY);
	img = 0.5 * channels[0] + 0.5 * channels[2];
#endif
#if PRE_PROCESS_LEVEL >= 2
	Mat bg = get_background(img, 3);
	img = compensate(img, bg);
#if PRE_PROCESS_LEVEL >= 3
	fastNlMeansDenoising(img, img, 3, 7, 25);
#if PRE_PROCESS_LEVEL >= 4
	threshold(img, img, 0, 255, THRESH_BINARY | THRESH_TRIANGLE);
	bitwise_not(img, img);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(img, img, MORPH_OPEN, kernel);
	morphologyEx(img, img, MORPH_CLOSE, kernel);
	Mat fill;
	img.copyTo(fill);
	floodFill(fill, Point(0, 0), Scalar::all(255));
	bitwise_not(fill, fill);
	img = img | fill;
#endif
#endif
#endif
	Mat res;
	resize(img, res, Size(64, 64));
	return res;
}

void get_hog_descriptors(Mat img, vector<float>& fv)
{
	// HOGDescriptor hog(Size(64, 64), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	HOGDescriptor hog(Size(64, 64), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	hog.compute(img, fv, Size(0, 0), Size(0, 0));
}

void generate_dataset(Mat& train_data, Mat& train_label)
{
	vector<string> images;
	vector<vector<float>> vecDec;
	vector<float> fv;
	glob("./Hand/V", images);
	int vNum = images.size();
	for (int i = 0; i < TRAIN_DATA_SIZE; i++)
	{
		Mat image = imread(images[i].c_str());
		image = pre_process_img(image);
		vector<float> fv;
		get_hog_descriptors(image, fv);
		printf("image path : %s, feature data length: %d \n", images[i].c_str(), fv.size());
		vecDec.push_back(fv);
	}
	images.clear();
	glob("./Hand/A", images);
	int aNum = images.size();
	for (int i = 0; i < TRAIN_DATA_SIZE; i++)
	{
		fv.clear();
		Mat image = imread(images[i].c_str());
		image = pre_process_img(image);
		get_hog_descriptors(image, fv);
		printf("image path : %s, feature data length: %d \n", images[i].c_str(), fv.size());
		vecDec.push_back(fv);
	}
	images.clear();
	glob("./Hand/Five", images);
	int fiveNum = images.size();
	for (int i = 0; i < TRAIN_DATA_SIZE; i++)
	{
		fv.clear();
		Mat image = imread(images[i].c_str());
		image = pre_process_img(image);
		get_hog_descriptors(image, fv);
		printf("image path : %s, feature data length: %d \n", images[i].c_str(), fv.size());
		vecDec.push_back(fv);
	}
	images.clear();
	glob("./Hand/C", images);
	int cNum = images.size();
	for (int i = 0; i < TRAIN_DATA_SIZE; i++)
	{
		fv.clear();
		Mat image = imread(images[i].c_str());
		image = pre_process_img(image);
		get_hog_descriptors(image, fv);
		printf("image path : %s, feature data length: %d \n", images[i].c_str(), fv.size());
		vecDec.push_back(fv);
	}

	int trainDataNum = TRAIN_DATA_SIZE * 4;
	int trainDataLen = fv.size();


	Mat trainDataTemp(trainDataNum, trainDataLen, CV_32FC1);
	Mat trainLabel(trainDataNum, 1, CV_32SC1);


	for (int i = 0; i < trainDataNum; i++)
	{
		for (int j = 0; j < trainDataLen; j++)
		{
			trainDataTemp.at<float>(i, j) = vecDec[i][j];
		}
		if (0 <= i && i < vNum)
		{
			trainLabel.at<int>(i) = 1;
		}
		else if (vNum <= i && i < vNum + aNum)
		{
			trainLabel.at<int>(i) = 2;
		}
		else if (vNum + aNum <= i && i < vNum + aNum + fiveNum)
		{
			trainLabel.at<int>(i) = 3;
		}
		else
		{
			trainLabel.at<int>(i) = 4;
		}
	}

	trainDataTemp.copyTo(train_data);
	trainLabel.copyTo(train_label);

	return;
}

void svm_train(Mat& trainData, Mat& labels) {
	printf("\n start SVM training... \n");
	Ptr< ml::SVM > svm = ml::SVM::create();
	/* Default values to train SVM */
	svm->setGamma(5.383);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setC(2.67);
	svm->setType(ml::SVM::C_SVC);
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
	vector<string> images;
	glob("./Hand/C", images);
	int cNum = images.size();
	Ptr<ml::SVM> mySVM = Algorithm::load<ml::SVM>("./hog_elec.yml");
	for (int i = TRAIN_DATA_SIZE; i < cNum; i++)
	{
		Mat image = imread(images[i].c_str());
		resize(image, image, Size(64, 64));
		vector<float> fv;
		get_hog_descriptors(image, fv);
		Mat one_row = Mat::zeros(Size(fv.size(), 1), CV_32F);
		for (int i = 0; i < fv.size(); i++) {
			one_row.at<float>(0, i) = fv[i];
		}
		float label = mySVM->predict(one_row);
		cout << labels[(int)label-1] << endl;
		//imshow("test image", image);
		//waitKey(0);
	}

	/*
	for (auto& i : filesystem::directory_iterator(".\\Hand\\C")) {
		Mat input = imread(i.path().string());
		Mat img;

		//resize(input, img, Size(256, 256));
		input.copyTo(img);

		Mat YCrCb_img;
		cvtColor(img, YCrCb_img, COLOR_RGB2YCrCb);
		Mat channels[6];
		split(YCrCb_img, channels);

		channels[3] = 0.5 * channels[0] + 0.5 * channels[2];

		//Mat kernel = (Mat_<int>(3, 3) <<
		//	0, -1, 0,
		//	-1, 4, -1,
		//	0, -1, 0
		//	);
		//filter2D(channels[3], channels[4], channels[3].depth(), kernel);
		//channels[4] += channels[3];
		////threshold(channels[3], channels[4], 0, 255, THRESH_OTSU);

		Mat test_inv;
		threshold(channels[3], test_inv, 0, 255, THRESH_OTSU);
		int sum = 0;
		for (int i = 0; i < test_inv.rows; i++)
		{
			uint8_t* ptr = test_inv.ptr<uint8_t>(i);
			for (int j = 0; j < test_inv.cols; j++)
			{
				if (ptr[j] != 0)
				{
					sum++;
				}
			}
		}
		if (sum * 2 > test_inv.rows * test_inv.cols)
		{
			bitwise_not(channels[3], channels[3]);
		}
		medianBlur(channels[3], channels[3], 5);

		equalizeHist(channels[3], channels[3]);

		Mat Edge_1;
		Canny(channels[0], Edge_1, 100, 200);

		//threshold(channels[3], channels[4], 0, 255, THRESH_OTSU);
		//adaptiveThreshold(channels[3], channels[4], 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 0);
		float p = (channels[3].rows < 200) ? 0.8 : 0.9;
		float rate = 0;
		int value = 0;
		while (rate < p)
		{
			int sum = 0;
			for (int i = 0; i < channels[3].rows; i++)
			{
				uint8_t* ptr = channels[3].ptr<uint8_t>(i);
				for (int j = 0; j < channels[3].cols; j++)
				{
					if (ptr[j] < value)
					{
						sum++;
					}
				}
			}
			rate = (float)sum / (channels[3].rows * channels[3].cols);
			value++;
		}
		threshold(channels[3], channels[4], value, 255, THRESH_BINARY);

		Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
		morphologyEx(channels[4], channels[4], MORPH_CLOSE, kernel, Point(-1, -1));
		Canny(channels[4], channels[5], 100, 200);
		channels[5] += Edge_1;
		//Canny(channels[3], Edge_1, 100, 200);
		//channels[5] += Edge_1;
		//morphologyEx(channels[5], channels[5], MORPH_CLOSE, kernel, Point(-1, -1));

		//imshow("Edges", channels[5]);

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(channels[5], contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		channels[5] = Mat::zeros(channels[5].rows, channels[5].cols, channels[5].type());
		sort(contours.begin(), contours.end(), [](vector<Point>a, vector<Point>b) {
			return contourArea(a) > contourArea(b);
			});
		Scalar color = Scalar::all(255);
		drawContours(channels[5], contours, 0, color, 1, LINE_AA, hierarchy, 0);
		Mat fill;
		Mat ori;
		channels[5].copyTo(fill);
		channels[5].copyTo(ori);
		floodFill(fill, Point(0, 0), Scalar::all(255));
		bitwise_not(fill, fill);
		channels[5] = channels[5] | fill;

		findContours(channels[5], contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		vector<Point> hull, approx_1;
		convexHull(contours[0], hull);
		double hullRate = contourArea(contours[0]) / contourArea(hull);
		double arcRate = arcLength(contours[0], true) / arcLength(hull, true);
		cout << hullRate << ' ' << arcRate << ' ';
		double epsilon = 0.01 * arcLength(contours[0], true);
		approxPolyDP(contours[0], approx_1, epsilon, true);
		Mat poly = Mat::zeros(channels[5].rows, channels[5].cols, channels[5].type());
		polylines(poly, approx_1, true, Scalar::all(255));

		Canny(channels[5], channels[5], 100, 200);

		int r_hit_count[10] = { 0 };
		int c_hit_count[10] = { 0 };
		for (int i = 0; i < channels[5].rows * 0.8; i++)
		{
			uint8_t* ptr = channels[5].ptr<uint8_t>(i);
			int hit = 0;
			bool l = false;
			for (int j = 0; j < channels[5].cols; j++)
			{
				if (l == false && ptr[j] > 0)
				{
					hit++;
					l = true;
				}
				else if (ptr[j] == 0)
				{
					l = false;
				}
			}
			r_hit_count[hit]++;
		}
		for (int i = 0; i < channels[5].cols; i++)
		{
			int hit = 0;
			bool l = false;
			for (int j = 0; j < channels[5].rows; j++)
			{
				if (l == false && channels[5].at<uint8_t>(j, i) > 0)
				{
					hit++;
					l = true;
				}
				else if (channels[5].at<uint8_t>(j, i) == 0)
				{
					l = false;
				}
			}
			c_hit_count[hit]++;
		}
		//cout << c_hit_count[3] + c_hit_count[4] << ' ';
		bool peak[3] = { false };
		int peakRate = 0;
		if (r_hit_count[2] > r_hit_count[1] && r_hit_count[2] > r_hit_count[3])
		{
			peak[0] = true;
			peakRate = 1;
		}
		if (r_hit_count[4] > r_hit_count[3] && r_hit_count[4] > r_hit_count[5])
		{
			peak[1] = true;
			peakRate = 2;
		}
		if (r_hit_count[6] > r_hit_count[5] && r_hit_count[6] > r_hit_count[7])
		{
			peak[2] = true;
			peakRate = 3;
		}
		if (peak[2] && r_hit_count[6] > 4)
		{
			cout << "Five" << endl;
		}
		else if (peak[1] && r_hit_count[4] > 5)
		{
			if (arcRate > 1.3 && hullRate < 0.7)
			{
				cout << "Five" << endl;
			}
			else
			{
				cout << "V" << endl;
			}
		}
		else if (peak[0] && r_hit_count[2] > 6)
		{
			if (arcRate > 1.2)
			{
				cout << "Five" << endl;
			}
			else if (hullRate < 0.7)
			{
				cout << "Not A" << endl;
			}
			else
			{
				cout << "A" << endl;
			}
		}
		else
		{
			cout << "CCCCC" << endl;
		}
#if OUT_FILE 1
		if (f_out.is_open())
		{
			// cout << r_hit_count[i] << ' ';
			f_out << hullRate << ", " << arcRate << ", " << peakRate << ", ";
			// cout << endl;
			f_out << endl;
		}
#endif

		//Mat Y;
		//hconcat(channels, 6, Y);
		//imshow("display", Y);
		//waitKey(0);
	}
#if OUT_FILE 1
	f_out.close();
#endif
*/
}