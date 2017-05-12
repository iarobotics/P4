// testOpenCV.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cstdio>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <cmath>
//#include <opencv2\features2d.hpp>

using namespace cv;
using namespace std;

int droneSize = 25;
int drone_max_altitude = 40;

int skyTrainingSamples = 128;
int skyTestSamples = 16;

int landTrainingSamples = 289;


Mat grayscale(Mat input);
Mat grayscale_unweighted(Mat input);
Mat binary(Mat gray);
void toFile(double area, double height, double width);
Mat binary_sky(Mat input);
Mat binary_land(Mat input);
Mat morphology_sky(Mat& bin);
Mat morphology_land(Mat& bin);
void stuff(Mat& morph);
void land_stuff(Mat& morph);
void land_stuff_mod(Mat& morph);
void stuff_sky_mod(Mat& morph);

void drone_lift_off(int& max_altitude);
void drone_hover(int& altitude);
void drone_cruise(Mat& morph, Point2f& center, int& droneSize);
void drone_move(String direction);
void drone_descend(int altitude);

bool destination_reached();
bool destination_not_reached();

//Resize and apply morphology to the input sky image
//Initiate flight mode
void do_sky(Mat& img_sky) {
	Size size(800, 600);
	resize(img_sky, img_sky, size);

	///Convert to grayscale with modified Blue weight
	Mat gray_sky = Mat(img_sky.rows, img_sky.cols, CV_8U);
	gray_sky = grayscale(img_sky);

	///thresholding and returning a binary image
	Mat bin_sky = Mat(gray_sky.rows, gray_sky.cols, CV_8U);
	bin_sky = binary(gray_sky);

	Mat morph_sky = morphology_sky(bin_sky);

	/*
	imshow("Gray", gray_sky);
	imshow("Input", img_sky);
	imshow("Binary", bin_sky);

	imwrite("D:\\img\\other\\input.jpg", img_sky);
	imwrite("D:\\img\\other\\binary.jpg", bin_sky);
	*/

	//stuff(morph_sky);
	stuff_sky_mod(morph_sky);
}

//Resize and apply morphology to the input land image
//Initiate landing mode
void do_land(Mat& img_land) {
	Size size(800, 600);
	resize(img_land, img_land, size);

	Mat gray_land = Mat(img_land.rows, img_land.cols, CV_8U);
	gray_land = grayscale(img_land);

	Mat bin_land = Mat(gray_land.rows, gray_land.cols, CV_8U);
	bin_land = binary_land(gray_land);

	Mat morph_land = morphology_land(bin_land);

	land_stuff_mod(morph_land);

	//imshow("Gray", gray_land);
	//imshow("Input", img_land);
	//imshow("Binary", bin_land);
	//imwrite("D:\\img\\other\\gray_land.jpg", gray_land);
	//imwrite("D:\\img\\other\\img_land.jpg", img_land);
	//imwrite("D:\\img\\other\\bin_land.jpg", bin_land);
}

int main()
{
	Mat img_sky;
	Mat img_land;
	drone_lift_off(drone_max_altitude);

	for (int k = 1; k <= landTrainingSamples; k++) {
		string a = to_string(k);

		/// Load colour image and create empty images for output:
		//img_land = imread("D:\\Downloads\\image_data\\landing_training\\training (999).jpg", IMREAD_ANYCOLOR);
		img_land = imread("D:\\Downloads\\image_data\\landing_training\\training (" + a + ").jpg", IMREAD_ANYCOLOR);

		//img_sky = imread("D:\\Downloads\\image_data\\training_data\\" + a + ".jpg", IMREAD_ANYCOLOR);
		//img_sky = imread("D:\\Downloads\\image_data\\testing_data\\t (" + a + ").jpg", IMREAD_ANYCOLOR);
		img_sky = imread("D:\\Downloads\\image_data\\training_data\\5.jpg", IMREAD_ANYCOLOR);


		//do_land(img_land);
		//do_sky(img_sky);
		//}

		if (destination_reached()) {
			cout << "DESTINATION REACHED" << endl;
			do_land(img_land);
		}
		else {
			do_sky(img_sky);
		}
	}

	//Sleep(10);

	


	cvWaitKey(0);
	return 0;
}

Mat grayscale(Mat input) {
	Mat output = Mat(input.rows, input.cols, CV_8U);
	const float WR = 0.0;
	const float WG = 0.0;
	const float WB = 1.0;

	for (int x = 0; x < input.cols; ++x) {
		for (int y = 0; y < input.rows; ++y) {
			int red = (input.at<Vec3b>(Point(x, y))[2])*WR;
			int green = (input.at<Vec3b>(Point(x, y))[1])*WG;
			int blue = (input.at<Vec3b>(Point(x, y))[0])*WB;
			output.at<uchar>(Point(x, y)) = red + green + blue;
		}
	}
	return output;
}

Mat grayscale_unweighted(Mat input) {
	Mat output = Mat(input.rows, input.cols, CV_8U);
	const float WR = 0.299;
	const float WG = 0.587;
	const float WB = 0.114;

	for (int x = 0; x < input.cols; ++x) {
		for (int y = 0; y < input.rows; ++y) {
			int red = (input.at<Vec3b>(Point(x, y))[2])*WR;
			int green = (input.at<Vec3b>(Point(x, y))[1])*WG;
			int blue = (input.at<Vec3b>(Point(x, y))[0])*WB;
			output.at<uchar>(Point(x, y)) = red + green + blue;
		}
	}
	return output;
}


Mat binary(Mat gray) {
	Mat bin = Mat(gray.rows, gray.cols, CV_8UC1);

	for (int x = 0; x < gray.cols; x++) {
		for (int y = 0; y < gray.rows; y++) {
			if (gray.at<uchar>(Point(x, y)) < 120) {
				bin.at<uchar>(Point(x, y)) = 0;
			}
			else {
				bin.at<uchar>(Point(x, y)) = 255;
			}
		}
	}
	return bin;
}

Mat binary_sky(Mat input) {
	Mat bin = Mat(input.rows, input.cols, CV_8UC1);

	for (int x = 0; x < input.cols; x++) {
		for (int y = 0; y < input.rows; y++) {
			if (input.at<Vec3b>(Point(x, y))[2] > 0) {
				bin.at<uchar>(Point(x, y)) = 0;
			}
			else {
				bin.at<uchar>(Point(x, y)) = 255;
			}
		}
	}
	return bin;
}

Mat binary_land(Mat gray) {
	Mat bin = Mat(gray.rows, gray.cols, CV_8UC1);

	for (int x = 0; x < gray.cols; x++) {
		for (int y = 0; y < gray.rows; y++) {
			if (gray.at<uchar>(Point(x, y)) < 150) {
				bin.at<uchar>(Point(x, y)) = 0;
			}
			else {
				bin.at<uchar>(Point(x, y)) = 255;
			}
		}
	}
	return bin;
}

void toFile(double area, double height, double width) {
	ofstream outfile;
	outfile.open("D:\\Downloads\\results.m", ios::app);
	outfile << area << "    " << height << "    " << width << endl;
	outfile.close();
}


Mat morphology_sky(Mat& bin) {
	Mat morph = Mat(bin.rows, bin.cols, CV_8U);

	Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(22, 22));
	morphologyEx(bin, morph, MORPH_ERODE, element1);
	//imshow("Morph", morph);
	//imwrite("D:\\img\\other\\morph1.jpg", morph);

	Mat element2 = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));
	morphologyEx(morph, morph, MORPH_DILATE, element2);
	//zimshow("Morph2", morph);
	//imwrite("D:\\img\\other\\morph2.jpg", morph);
	

	return morph;
}

Mat morphology_land(Mat& bin) {
	Mat morph = Mat(bin.rows, bin.cols, CV_8U);

	Mat element1 = getStructuringElement(MORPH_RECT, Size(4, 4));
	morphologyEx(bin, morph, MORPH_ERODE, element1);
	//imshow("Erode", morph);

	Mat element2 = getStructuringElement(MORPH_RECT, Size(40,25	));
	morphologyEx(morph, morph, MORPH_DILATE, element2);
	//imshow("Dilate", morph);


	//Mat element2 = getStructuringElement(MORPH_RECT, Size(10, 10));
	//morphologyEx(bin, morph, MORPH_DILATE, element1);
	//imshow("Erode", morph);

	/*
	///Consider using very high dilation such as to avoid small objects in the light
	//Mat element1 = getStructuringElement(MORPH_RECT, Size(2, 2));
	Mat element1 = getStructuringElement(MORPH_RECT, Size(20, 15));
	morphologyEx(bin, morph, MORPH_ERODE, element1);
	imshow("Erode", morph);
	Mat element2 = getStructuringElement(MORPH_RECT, Size(90, 60));
	morphologyEx(morph, morph, MORPH_DILATE, element2);
	imshow("ERODE_DILATE", morph);

	morphologyEx(morph, morph, MORPH_ERODE, element2);
	imshow("Erode2", morph);

	morphologyEx(morph, morph, MORPH_DILATE, element1);
	imshow("ERODE_DILATE2", morph);
	*/

	//imwrite("D:\\img\\other\\morph_dilate.jpg", morph);

	//Mat element2 = getStructuringElement(MORPH_RECT, Size(10, 10));
	//morphologyEx(morph, morph, MORPH_DILATE, element1);
	//imshow("Morph2", morph);

	return morph;
}


void stuff(Mat& morph) {

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	///RETR_CCOMP - This flag retrieves all the contours and arranges them to a 2-level hierarchy.
	findContours(morph, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	///Find all contours within an image 
	for (int i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i]);
		if (area > 100000) {
			Mat blankImg = Mat(morph.rows, morph.cols, CV_8UC3);
			//Scalar color(rand() & 255, rand() & 255, rand() & 255);
			Scalar color(0, 0, 255);
			//drawContours(blankImg, contours, i, Scalar(0, 0, 255), 1);
			drawContours(blankImg, contours, i, color, CV_FILLED, 8, hierarchy);

			///Draw a rectangle around the found contour
			vector<vector<Point> > contours_poly(contours.size());
			vector<Rect> boundRect(contours.size());
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
			Scalar color_rect(255, 0, 0);
			rectangle(blankImg, boundRect[i].tl(), boundRect[i].br(), color_rect, 1, 8, 0);

			string b = to_string(i);
			//imshow(b, blankImg);

			//Crop the picture to the size of the rectangle enclosing the sky
			Mat output = Mat(boundRect[i].height - 2, boundRect[i].width - 2, CV_8UC3);
			for (int x = 0; x < output.cols; ++x) {
				for (int y = 0; y < output.rows; ++y) {
					output.at<Vec3b>(Point(x, y))[2] = blankImg.at<Vec3b>(Point(x + 1, y + 1))[2];
					output.at<Vec3b>(Point(x, y))[1] = blankImg.at<Vec3b>(Point(x + 1, y + 1))[1];
					output.at<Vec3b>(Point(x, y))[0] = blankImg.at<Vec3b>(Point(x + 1, y + 1))[0];
				}
			}


			//TODO: Change the thresholding function here
			Mat grays = grayscale(output);
			//imshow("OUT", grays);
			Mat binar = binary(grays);
			//imshow("BIN", binar);

			//imwrite("D:\\img\\contours\\"+a+"_"+b+"_contour.jpg", blankImg);
			vector<vector<Point>> contours_2;
			vector<Vec4i> hierarchy_2;

			///RETR_CCOMP - This flag retrieves all the contours and arranges them to a 2-level hierarchy.
			findContours(binar, contours_2, hierarchy_2, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			Mat blankImg_2;
			for (int x = 0; x < contours_2.size(); x++) {
				double area_2 = contourArea(contours_2[x]);
				//Mat blankImg_2;

				///vector<vector<Point> > contours_poly_2(contours_2.size());
				///vector<Rect> boundRect_2(contours_2.size());
				///approxPolyDP(Mat(contours_2[x]), contours_poly_2[x], 3, true);
				///boundRect_2[x] = boundingRect(Mat(contours_poly_2[x]));
				Rect boundRect_2 = boundingRect(contours_2[x]);
				double height_2 = boundRect_2.height;
				double width_2 = boundRect_2.width;

				Scalar color_rect(255, 0, 0);
				rectangle(blankImg_2, boundRect_2.tl(), boundRect_2.br(), color_rect, 1, 8, 0);
				//double height_2 = boundRect_2[x].height;
				//double width_2 = boundRect_2[x].width;

				cout << "A: " << area_2 << endl;
				cout << "H: " << height_2 << endl;
				cout << "W: " << width_2 << endl;

				//toFile(k, area_2, height_2, width_2);

			}

			///Areas for objects in the sky
			//cout << "area" << i << " =" << area << endl;
			toFile(area, output.rows, output.cols);
		}
		///All areas
		//cout << "area"<<i<<" =" << area << endl;
	}
}

void stuff_sky_mod(Mat& morph) {

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	///RETR_CCOMP - This flag retrieves all the contours and arranges them to a 2-level hierarchy.
	findContours(morph, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	///Find all contours within an image

	/// Get the moments
	vector<Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}

	///  Get the mass centers:
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}

	Mat blankImg = Mat(morph.rows, morph.cols, CV_8UC3);

	if (!contours.empty() && !hierarchy.empty()) {
		for (int i = 0; i < contours.size(); i++) {
			double area = contourArea(contours[i]);
			//Scalar color(rand() & 255, rand() & 255, rand() & 255);

			///Draw a rectangle around the found contour
			Scalar color_rect(255, 0, 0);
			Rect boundRect = boundingRect(contours[i]);

			double width = boundRect.width;
			double height = boundRect.height;
			//toFile(area, height, width);

			if (area > 10000 && width >= 600) {
				///Draw the sky and its included objects, hierarchy 0 displays only the sky and the horizon line
				Scalar color(0, 0, 255);
				drawContours(blankImg, contours, i, color, 1, 8, hierarchy,1);

				///Draw a rectangle around the sky object - enclosing the horizon and objects on the sky
				//rectangle(blankImg, boundRect.tl(), boundRect.br(), color_rect, 1, 8, 0);

				///Draw a circle at the mass center of the sky
				Point2f center = mc[i];
				circle(blankImg, center, 4, color, -1, 8, 0);

				///Draw a square that shows the area where the drone flies
				cv::rectangle(
					blankImg,
					cv::Point(center.x - droneSize, center.y - droneSize),
					cv::Point(center.x + droneSize, center.y + droneSize),
					cv::Scalar(255, 255, 255)
				);
				

				///Fly and avoid obstacles
				drone_cruise(morph, center, droneSize);
				
				
				//drawContours(blankImg, contours, i, color, CV_FILLED, 8, hierarchy, 0);
			}
		}
	}
	//namedWindow("Sky", CV_WINDOW_AUTOSIZE);
	//imshow("Sky", blankImg);
}

void land_stuff_mod(Mat& morph) {

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	///Find all contours within an image
	///RETR_CCOMP - This flag retrieves all the contours and arranges them to a 2-level hierarchy.
	//findContours(morph, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	findContours(morph, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	
	/// Get the moments
	vector<Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}

	///  Get the mass centers:
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}

	Mat blankImg = Mat(morph.rows, morph.cols, CV_8UC3);
	

	if (!contours.empty() && !hierarchy.empty()) {

		bool landing_pad_found = false;
		double max_area = 0;
		int index = 0;
		double land_width = 0;
		double land_height = 0;

		for (int i = 0; i < contours.size(); i++) {	
			double area = contourArea(contours[i]);
			

			//Scalar color(rand() & 255, rand() & 255, rand() & 255);

			///Draw a rectangle around the found contour
			Scalar color_rect(255, 0, 0);
			Rect boundRect = boundingRect(contours[i]);

			double width = boundRect.width;
			double height = boundRect.height;
			//toFile(area, height, width);

			///Draw the sky and its included objects, hierarchy 0 displays only the sky and the horizon line
			
			if (area > 1000 && area < 10000 ) {
				if (area > max_area) {
					max_area = area;
					index = i;
					land_width = width;
					land_height = height;
				}
				landing_pad_found = true;
				//drawContours(blankImg, contours, i, color, 1, 8, hierarchy);
				//cout << area << endl;
				
			}
			///Draw a rectangle around the sky object - enclosing the horizon and objects on the sky
			//rectangle(blankImg, boundRect.tl(), boundRect.br(), color_rect, 1, 8, 0);

			///Draw a circle at the mass center of every object
			//Point2f center = mc[i];
			//circle(blankImg, center, 4, color, -1, 8, 0);

			//imshow("Contours", blankImg);
			
		}
		

		if (landing_pad_found) {

			toFile(max_area, land_height, land_width);
			max_area = 0;
			Scalar color(0, 0, 255);
			drawContours(blankImg, contours, index, color, 1, 8, hierarchy);

			///Draw the center of mass for the landing pad
			Point2f center = mc[index];
			circle(blankImg, center, 4, (255, 0, 0), -1, 8, 0);

			///Draw the landing area of interest
			cv::rectangle(
				blankImg,
				cv::Point(375, 275),
				cv::Point(425,325),
				cv::Scalar(255, 255, 255)
			);
			int x_min = 375;
			int x_max = 425;
			int y_min = 275;
			int y_max = 325;

			//while (center.x < 375 || center.x > 425 || center.y < 275 || center.y > 325) {

				if (center.x < 375) {
					drone_move("RIGHT");
				}
				else if (center.x > 425) {
					drone_move("LEFT");
				}

				if (center.y < 275) {
					drone_move("BACK");
				}
				else if (center.y > 325) {
					drone_move("FORWARD");
				}
				else {
					drone_descend(drone_max_altitude);
				}
			//}
		}
		

		
	}
	//namedWindow("Sky", CV_WINDOW_AUTOSIZE);
	//imshow("Sky", blankImg);
}

void land_stuff(Mat& morph) {

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	///RETR_CCOMP - This flag retrieves all the contours and arranges them to a 2-level hierarchy.
	findContours(morph, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	///Find all contours within an image
	Mat blankImg = Mat(morph.rows, morph.cols, CV_8UC3);

	double min_area = 0;
	int index = 0;

	for (int i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i]);
		if (area>0) {
			//Scalar color(rand() & 255, rand() & 255, rand() & 255);
			Scalar color(0, 0, 255);
			//drawContours(blankImg, contours, i, Scalar(0, 0, 255), 1);
			drawContours(blankImg, contours, i, color, CV_FILLED, 8, hierarchy);

			///Draw a rectangle around the found contour
			vector<vector<Point> > contours_poly(contours.size());
			vector<Rect> boundRect(contours.size());

			/*
			vector<RotatedRect> minRect(contours.size());
			minRect[i] = minAreaRect(Mat(contours[i]));

			double height_2 = minRect[i].size[];
			double width_2 = boundRect_2.width;

			Point2f rect_points[4]; minRect[i].points(rect_points);
			for (int j = 0; j < 4; j++)
			line(blankImg, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
			*/


			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
			double width = boundRect[i].width;
			double height = boundRect[i].height;

			Scalar color_rect(255, 0, 0);
			rectangle(blankImg, boundRect[i].tl(), boundRect[i].br(), color_rect, 1, 8, 0);

			//string b = to_string(i);
			//imshow(b, blankImg);

			//cout << "area" << i << " =" << area << endl;
			//toFile(area, height, width);
		}

		///Areas for objects in the sky
		//cout << "area" << i << " =" << area << endl;
	}
	//imshow("Blank", blankImg);
	//imwrite("D:\\img\\other\\objects_land.jpg", blankImg);
}

void drone_lift_off(int& max_altitude) {
	cout << "DRONE: LIFT OFF" << endl;
	cout << "DRONE: ALTITUDE "<<max_altitude<<"m REACHED" << endl;
	drone_hover(max_altitude);
}

void drone_hover(int& altitude) {
	cout << "DRONE: HOVERING AT: " << altitude << "m" << endl;
}
void drone_cruise(Mat& morph, Point2f& center, int& droneSize) {
	bool found = false;
	for (int x = center.x - droneSize; x < center.x + droneSize; x++) {
		for (int y = center.y - droneSize; y < center.y + droneSize; y++) {
			if (morph.at<uchar>(Point(x, y)) == 0) {
				cout << "DRONE: Obstacle Found " << x << " " << y << endl;
				/// Order the drone to move  
				///TODO: move accoridng to where the obstacle was detected - store all points in a vector<Point>
				found = true;
				break;
			}
		}if (found == true) { break; }
	}
	if (found == true){
		drone_move("LEFT");
	}
	else {
		drone_move("FORWARD");
	}
		

}

void drone_descend(int altitude) {
	for (int i = altitude; i > 0; i -= 5) {
		cout << "DRONE: DESCNEDING  ALTITUDE: " << i << endl;
	}
}
///Function to move the drone while it's hovering
void drone_move(String direction) {
	cout << "DRONE: MOVING: " << direction<<endl;
	//drone_hover(drone_max_altitude);
}

bool destination_reached() {
	//Query is GPS coordinates in range, return true is so, otherwise return false
	return true;
}

bool destination_not_reached() {
	return false;
}
