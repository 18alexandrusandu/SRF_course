// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"

#define MAXT 1000

struct weaklearner {
	int c;
	int class_label;
	float error;
	int classify(Mat X) {



		if (X.at<float>(0) + X.at<float>(1) > c || X.at<float>(0) - X.at<float>(1) < c)
			return class_label;
		else
			return -class_label;
	   }



};

struct classifier {
	int T;
	float alphas[MAXT];
	weaklearner hs[MAXT];
	int classify(Mat X) {
      	
		float H = 0;
		for (int i = 0; i < T; i++)
		{
			H += alphas[i] * hs[i].classify(X);
		}
		return (H > 0) ? 1 : -1;

	}
};


weaklearner findWeakLearner(Mat img, Mat X, Mat y, Mat w)
{
	int classL[2] = {-1, 1};


	weaklearner best_h;
	int best_err = img.rows * img.cols + 1;

	for (int c = -img.cols; c < img.cols; c++)
	{
		float zi = 0;
		for (int cls = 0; cls < 2; cls++)
		{
			float err = 0;
			for (int i = 1; i < X.rows; i++)
			{
				
					if ((X.at<float>(i, 0) + X.at<float>(i, 1) - c < 0) || (X.at<float>(i, 0) - X.at<float>(i, 1) - c) > 0)
						zi = classL[cls];
					else
						zi = -classL[cls];
				

				if (zi * y.at<int>(i, 0) < 0)
					err += w.at<float>(i, 0);
			}

			if (err < best_err)
			{
				printf("asta %f\n", err);
				best_err = err;
				best_h = { c, classL[cls],err };

			}
		}
	}
	std::cout << "bestf" << best_h.c << " " << best_h.class_label << " " << best_h.error << std::endl;
	return best_h;

}

classifier Train(Mat img,Mat X,Mat y,Mat* w,int T)
{   
	classifier Boost;
	Boost.T = T;
	for (int i = 0; i < X.rows; i++)
	{
		(*w).at<float>(i, 0) = 1.0 / X.rows;
	}

	for (int t = 0; t < T; t++)
	{

		weaklearner  wl = findWeakLearner(img, X, y, *w);
		float at = 0.5 * log((1.0 - wl.error) / wl.error);
		Boost.alphas[t] = at;
		Boost.hs[t] = wl;
		float s = 0;
		for (int i = 0; i < X.rows; i++)
		{
			(*w).at<float>(i, 0)= (*w).at<float>(i, 0) * exp(-at * y.at<int>(i,0) * wl.classify(X.row(i)));
			s += (*w).at<float>(i,0);

		}
		for (int i = 0; i < X.rows; i++)
			(*w).at<float>(i, 0) = (*w).at<float>(i, 0) / s;

	}
	 
	return Boost;


}
void drawBoundary(Mat img, classifier cls)
{

	Mat img2 = img.clone();
	for(int i=0;i<img.rows;i++)
		for (int j = 0; j < img.cols; j++)
		{

			if (img.at<Vec3b>(i, j)[0] == 255 && img.at<Vec3b>(i, j)[2] == 255)
			{
				Mat X(1,2, CV_64FC1);
				X.at<float>(0,0) = j;
				X.at<float>(0,1) = i;
				int classP = cls.classify(X);
				if (classP > 0)
				{
					img2.at<Vec3b>(i, j) = { 0,255,255 };
				}
				else
				{
					img2.at<Vec3b>(i, j) = { 255,255,0 };
				}

		     }


		}
	imshow("classified", img2);
	waitKey(100);
}




int main()
{
         Mat img = imread("inputB.png", IMREAD_COLOR);
		 std::vector<std::pair<Point, int>> tmp;
		 imshow("original", img);
		 waitKey();



	 for(int i=0;i<img.rows;i++)
			 for (int j = 0; j < img.cols; j++)
			 {
				 if (img.at<Vec3b>(i, j)[0] == 255 && img.at<Vec3b>(i, j)[1] == 0 && img.at<Vec3b>(i, j)[2] == 0)
					 tmp.push_back({ { j,i }, 1 });
				 
				 if (img.at<Vec3b>(i, j)[0] == 0 && img.at<Vec3b>(i, j)[1] == 0 && img.at<Vec3b>(i, j)[2] == 255)
					 tmp.push_back({ { j,i }, -1 });

			 }
	 Mat X(tmp.size(), 2, CV_32FC1);
	 Mat Y(tmp.size(), 1,CV_32SC1);
	 for (int i = 0; i < tmp.size(); i++)
	 {
		 X.at<float>(i, 0) = tmp.at(i).first.x;
		 X.at<float>(i, 1) = tmp.at(i).first.y;
		 Y.at<int>(i, 0) = tmp.at(i).second;


	 }
	 std::cout << Y;


	 Mat W = Mat(tmp.size() , 1, CV_32FC1);


	 int t = 1;

	
	 boolean no_error = false;

	 while (!no_error)
	 {
		 printf("t=%d", t);
		 classifier b = Train(img, X, Y, &W, t);
		 int err = 0;
		 for (int i = 0; i < X.rows; i++)
		 {
			 if (b.classify(X.row(i)) != Y.at<int>(i))
				 err += 1;
		 }
		 
		 drawBoundary(img, b);


		 if (err == 0)
			 no_error = true;
		 t++;
	 }
	 printf("the minimum t for no error is:", t);
	
}

