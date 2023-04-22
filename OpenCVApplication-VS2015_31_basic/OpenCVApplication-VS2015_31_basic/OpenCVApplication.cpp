// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey(0);
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey(0)==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey(0);
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey(0);
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey(0);
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey(0);
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey(0);
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey(0);
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}



void first_2_methods()
{


	for (int k = 0; k < 5; k++)
	{


		char source[100] = "prs_res_LeastSquares/points0.txt";
		source[27] = '1' + k;
		system("cls");

		destroyAllWindows();
		FILE* f = fopen(source, "r");
		int n;
		float x, y;
		fscanf(f, "%d", &n);
		int height = 500, width = 500;






		float sumx = 0, sumy = 0, sumxy = 0, sumx2 = 0, sumx2y2 = 0;
		float beta = 0, fi = 0;
		int i = 0;
		float theta0 = 0, theta1 = 0;


		float minx = 0, miny = 0;
		while (i < n)
		{

			fscanf(f, "%f%f", &y, &x);
			printf("x=%f y=%f \n", x, y);
			sumx += x;
			sumy += y;
			sumxy += x * y;
			sumx2 += x * x;
			i++;
			if (x < minx)
				minx = x;
			if (y < miny)
				miny = y;


			sumx2y2 += y * y - x * x;
			if (ceil(y) > height)
				height = ceil(y);
			if (ceil(x) > width)
				width = ceil(x);




		}
		theta1 = (n * sumxy - sumx * sumy) / (n * sumx2 - sumx * sumx);
		theta0 = (sumy - theta1 * sumx) / n;

		beta = -atan2(2 * sumxy - 2.0 / n * sumx * sumy, sumx2y2 + 1.0 / n * sumx * sumx - 1.0 / n * sumy * sumy) / 2;
		fi = 1.0 / n * (cos(beta) * sumx + sin(beta) * sumy);




		printf("t1=%f t0=%f\n", theta1, theta0);
		printf("w=%d h=%d\n", width, height);

		float finaly = theta0 + theta1 * width;

		printf("w=%d,fy=%f\n", width, theta0 + theta1 * width);
		printf("b=%f fi=%f\n", beta, fi);

		Mat img(height, width, CV_8UC3);
		Mat img2(height, width, CV_8UC3);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
			{
				img.at<Vec3b>(i, j)[0] = 255; //blue
				img.at<Vec3b>(i, j)[1] = 255; //green
				img.at<Vec3b>(i, j)[2] = 255;

				img2.at<Vec3b>(i, j)[0] = 255; //blue
				img2.at<Vec3b>(i, j)[1] = 255; //green
				img2.at<Vec3b>(i, j)[2] = 255;

			}


		line(img, Point(0, theta0), Point(width, finaly), Scalar(0, 0, 255));


		fclose(f);
		f = fopen(source, "r");
		fscanf(f, "%d", &n);
		for (int i = 0; i < n; i++)
		{
			fscanf(f, "%f%f", &y, &x);
			if (x > 0 & y > 0)
			{
				circle(img, Point(x, y), 3, Scalar(0, 255, 0));
				circle(img2, Point(x, y), 3, Scalar(0, 255, 0));
			}
		}
		fclose(f);
		if (abs(sin(beta)) > 0.01)
		{

			for (int j = 0; j < width; j++)
			{
				int i = (fi - j * cos(beta)) / sin(beta);


				if (i >= 0 && i < height)
				{
					img2.at<Vec3b>(i, j)[0] = 0; //blue
					img2.at<Vec3b>(i, j)[1] = 0; //green
					img2.at<Vec3b>(i, j)[2] = 255;
				}


			}
		}
		else
		{


			for (int i = 0; i < height; i++)
			{
				int j = (fi - i * sin(beta)) / cos(beta);

				if (j >= 0 && j < height)
				{

					img2.at<Vec3b>(i, j)[0] = 0; //blue
					img2.at<Vec3b>(i, j)[1] = 0; //green
					img2.at<Vec3b>(i, j)[2] = 255;
				}



			}


		}



		char imgs1[50] = "my img0.1";
		char imgs2[50] = "my img0.2";
		imgs1[6] = '0' + k;
		imgs2[6] = '0' + k;
		imshow(imgs1, img);
		imshow(imgs2, img2);

		waitKey(0);


	}






}

void ransac(){
	for (int k = 1; k <= 5; k++)
	{

		int n, N, t = 10, a, b, c, aOptim = 0, bOptim = 0, cOptim = 0, cnt = 0, max_cnt = -1, s = 2;
		float p = 0.99, q, T;


		char source[100] = "prs_res_Ransac/points0.bmp";
		source[21] = '0' + k;
		printf("%s\n", source);

		Mat img = imread(source, CV_LOAD_IMAGE_GRAYSCALE);
		std::vector<Point> points;



		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
			{
				if (img.at<uchar>(i, j) == 0) {
					Point p; p.x = j; p.y = i;
					points.push_back(p);
				}

			}
		n = points.size();
		printf("n=%d\n", n);
		for (int i = 0; i < n; i++)
			printf("%d ,%d\n", points[i].x, points[i].y);


		//compute N,T,t,p

		if (k == 5 || k == 1)
			q = 0.3;
		else
			q = 0.8;
		N = log(1 - p) / log(1 - pow(q, s));

		printf("N is %d\n", N);
		T = q * n;

		printf("N is %d and T is %f\n", N, T);
		max_cnt = 0;
		for (int tries = 1; tries <= N; tries++)
		{
			int i1 = rand() % n;
			int i2 = rand() % n;
			while (i1 == i2)
				i2 = rand() % n;

			a = points[i1].y - points[i2].y;
			b = points[i2].x - points[i1].x;
			c = points[i1].x * points[i2].y - points[i2].x * points[i1].y;


			cnt = 0;
			for (int point = 0; point < n; point++)
			{
				float den = sqrt(a * a + b * b);


				float dist = abs(a * points[point].x + points[point].y * b + c) / den;

				if (dist < t)
					cnt++;
			}

			if (cnt > max_cnt)
			{
				max_cnt = cnt;
				aOptim = a;
				bOptim = b;
				cOptim = c;
			}

			if (cnt > T)
				break;

		}


		Mat newimg = img.clone();
		printf("boptim=%d\n", bOptim);

		if (abs(bOptim) > 5)
		{
			for (int j = 0; j < newimg.cols; j++)
			{
				int i = (-cOptim - aOptim * j) / bOptim;
				if (i >= 0 && i < newimg.rows)
					newimg.at<uchar>(i, j) = 0;
			}
		}
		else
		{

			for (int i = 0; i < newimg.rows; i++)
			{
				int j = (-cOptim - bOptim * i) / aOptim;
				if (j >= 0 && j < newimg.cols)
					newimg.at<uchar>(i, j) = 0;
			}

		}




		imshow("img", img);
		imshow("imagine with fitting line", newimg);
		waitKey(0);



	}


}


float convRad(int a)
{
	return (float)a * PI / 180;
}


struct peak {
	int theta, ro, hval;
	bool operator < (const peak& o) const {
		return hval > o.hval;
	}
};





void Hough(char* source1,char* source2)
{
	//char source1[50] = "prs_res_Hough/edge_simple.bmp";
	//char source2[50] = "prs_res_Hough/image_simple.bmp";
	char source3[50] = "prs_res_Hough/edge_complex.bmp";
	char source4[50] = "prs_res_Hough/image_complex.bmp";

	Mat img1 = imread(source2, CV_LOAD_IMAGE_GRAYSCALE);
;


	Mat img3 = imread(source1);
	

	int width = img1.cols;
	int height = img1.rows;



	int deltafi = 1;
	int deltatheta = 1;

	int D = sqrt(img1.rows * img1.cols + img1.cols * img1.cols);
	Mat H(D + 1, 360, CV_32SC1);

	H.setTo(0);


	for (int i = 0; i < img1.rows; i++)
		for (int j = 0; j < img1.cols; j++)
		{

			if (img1.at<uchar>(i, j) == 255)
			{

				for (int theta = 0; theta < 360; theta++)
				{



					float theta2 = convRad(theta);
					int ro = ceil(j * cos(theta2) + i * sin(theta2));
					if (ro >= 0 && ro < D + 1)
						H.at<int>(ro, theta)++;

				}


			}
		}
	int  maxHough = -1;

	for (int i = 0; i < H.rows; i++)
		for (int j = 0; j < H.cols; j++)
			if (H.at<int>(i, j) > maxHough)
			{
				maxHough = H.at<int>(i, j);
			}

	Mat houghImg;
	H.convertTo(houghImg, CV_8UC1, 255.0f / maxHough);
	imshow("imagine linii", houghImg);
	waitKey(0);

	std::vector<peak> v;



	for (int ro = 0; ro < H.rows; ro++)
		for (int theta = 0; theta < H.cols; theta++)
		{
			int ok = 1;
			int w = 3;
			if (H.at<int>(ro, theta) == 0)
				ok = 0;

			for (int Dro = -w; Dro <= w; Dro++)
				for (int Dtheta = -w; Dtheta <= w; Dtheta++)
					if (ro + Dro >= 0 && ro + Dro < H.rows && theta + Dtheta < H.cols && theta + Dtheta >= 0)
						if (H.at<int>(ro + Dro, theta + Dtheta) >H.at<int>(ro, theta))
							ok = 0;

			if (ok == 1)
				//if(H.at<int>(ro, theta)>5)
					v.push_back({ theta,ro,H.at<int>(ro, theta) });

		}




	sort(v.begin(), v.end());

	for (int k = 0; k < v.size(); k++)
		printf(" k=%d ,ro=%d theta=%d H=%d\n", k, v.at(k).ro, v.at(k).theta, v.at(k).hval);

	//desenare linii
	for (int k = 0; k < min(9, v.size()); k++)
	{
		int ro = v.at(k).ro;
		int theta = v.at(k).theta;
		float radtheta = convRad(theta);

		if (abs(sin(radtheta)) > 0.1)
		{








			for (int j = 0; j < width; j++)
			{
				int i = (ro - j * cos(radtheta)) / sin(radtheta);


				if (i >= 0 && i < height)
				{
					img3.at<Vec3b>(i, j)[0] = 0; //blue
					img3.at<Vec3b>(i, j)[1] = 255; //green
					img3.at<Vec3b>(i, j)[2] = 0;
				}


			}
		}
		else
		{


			for (int i = 0; i < height; i++)
			{
				int j = (ro - i * sin(radtheta)) / cos(radtheta);

				if (j >= 0 && j < height)
				{

					img3.at<Vec3b>(i, j)[0] = 0; //blue
					img3.at<Vec3b>(i, j)[1] = 255; //green
					img3.at<Vec3b>(i, j)[2] = 0;
				}



			}


		}
	}
	imshow("image without edges", img1);
	imshow("image with edges", img3);
	waitKey(0);


	


	Hough(source4, source3);
}


int weight(int k,int  l,int dim_masc)
{

	if (k == 0 && l == 0)
		return 0;

	int whV = 2;
	int wD = 3;
	if (k == 0 || l == 0 )
	{
		return whV;
	}

	if (k == l)
	{
		return wD;
	}
	if (k + l == dim_masc - 1)
		return wD;



}


void calcul_DT(Mat* DT,Mat img)
{
   (* DT) = Mat(img.rows, img.cols, CV_32SC1);
	int dim_masca = 1;
	int weightD = 3;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			(*DT).at<int>(i, j) = img.at<uchar>(i, j);

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{

			for (int k = -dim_masca; k <= 0; k++)
				for (int l = -dim_masca; l <= dim_masca; l++)
				{

					if ((k == 0 && l < 0) || k != 0)
						if (i + k >= 0 && (i + k) < (*DT).rows && j + l >= 0 && j + l < (*DT).cols)
						{
							//printf("%d %d %d %d\n", i, j, i + k, j + l);
							(*DT).at<int>(i, j) = min((*DT).at<int>(i, j), (*DT).at<int>(i + k, j + l) + weight(k, l, dim_masca));

						}


				}

		}
	for (int i = img.rows - 1; i >= 0; i--)
		for (int j = img.cols - 1; j >= 0; j--)
		{
			int mind = img.rows * img.cols;
			for (int k = 0; k <= dim_masca; k++)
				for (int l = -dim_masca; l <= dim_masca; l++)
				{
					if ((k == 0 && l > 0) || k != 0)
						if (i + k >= 0 && i + k < (*DT).rows && j + l >= 0 && j + l < (*DT).cols)
							(*DT).at<int>(i, j) = min((*DT).at<int>(i, j), (*DT).at<int>(i + k, j + l) + weight(k, l, dim_masca));
				}
		}

	Mat img_GrayD = Mat((*DT).rows, (*DT).cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			img_GrayD.at<uchar>(i, j) = min(255, (*DT).at<int>(i, j));
	imshow("gray Image", img_GrayD);
	waitKey();
}


void process_DT_similaritate(char* model, char* incercare)
{
	Mat img2 = imread(model, IMREAD_GRAYSCALE);
	Mat img3 = imread(incercare, IMREAD_GRAYSCALE);
	std::vector<Point> v, q;
	imshow("template", img2);
	imshow("new Image", img3);

	Mat DT1;
	Mat img5 = img3.clone();

	calcul_DT(&DT1, img2);
	
	for (int i = 0; i < img2.rows; i++)
		for (int j = 0; j < img2.cols; j++)
			if (img2.at<uchar>(i, j) == 0)
				v.push_back({ j,i });


	int cx = 0, cy = 0;
	for (int i = 0; i < v.size(); i++)
	{
		cx += v.at(i).x;
		cy += v.at(i).y;
	}
	cx = cx / v.size();
	cy = cy / v.size();


	for (int i = 0; i < img3.rows; i++)
		for (int j = 0; j < img3.cols; j++)
			if (img3.at<uchar>(i, j) == 0)
				q.push_back({ j,i });
	

	int cx2 = 0, cy2 = 0;
	for (int i = 0; i < q.size(); i++)
	{
		cx2 += q.at(i).x;
		cy2 += q.at(i).y;
	}
	cx2 = cx2 / q.size();
	cy2 = cy2 / q.size();

	//centrare
	int displacementX = cx - cx2;
	int displacementY = cy- cy2;

	//Calcul Scor
	int scor = 0;
	



	for (int i = 0; i < q.size(); i++)
	{
		if (q.at(i).x + displacementX >= 0 && q.at(i).y + displacementY >= 0 && q.at(i).x + displacementX < DT1.cols && q.at(i).y + displacementY < DT1.rows)
			scor += DT1.at<int>(q.at(i).y + displacementY, q.at(i).x + displacementX);
	}
	scor = scor / q.size();
	printf("score cu shiftare este %d\n", scor);

}




typedef struct
{
	int x, y;

}Punct;


using namespace std;
void statistics()
{
	char folder[256] = "faces";
	char fname[256];
	Mat I = Mat(400, 19 * 19, CV_8UC1);



	for (int i = 1; i <= 400; i++) {
		sprintf(fname, "%s/face%05d.bmp", folder, i);

		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("img1", img);

		for (int j = 0; j < img.rows; j++)
			for (int k = 0; k < img.cols; k++)
			{
				I.at<uchar>(i - 1, j * 19 + k) = img.at<uchar>(j, k);
			}
	}
	float avg[400] = { 0 };

	for (int k = 0; k < 19 * 19; k++)
	{
		avg[k] = 0;

		for (int j = 0; j < 400; j++)
		{
			avg[k] += I.at<uchar>(j, k);
		}
		avg[k] = avg[k] / 400;

	}
	FILE* F = fopen("averages.csv", "w");
	for (int k = 0; k < 19 * 19; k++)
	{
		printf("%d:%f ;", k, avg[k]);

		fprintf(F, "%f,", avg[k]);
	}
	fclose(F);
	Mat C = Mat(19 * 19, 19 * 19, CV_32FC1);
	F = fopen("covalenta.csv", "w");
	for (int i = 0; i < 19 * 19; i++)
	{


		for (int j = 0; j < 19 * 19; j++)
		{

			C.at<float>(i, j) = 0;
			for (int k = 0; k < 400; k++)
			{
				C.at<float>(i, j) += (I.at<uchar>(k, i) - avg[i]) * (I.at<uchar>(k, j) - avg[j]);
			}
			C.at<float>(i, j) = C.at<float>(i, j) / 400;
			fprintf(F, "%f,", C.at<float>(i, j));

		}
		fprintf(F, ";");
	}
	fclose(F);
	for (int i = 0; i < 19 * 19; i++)
	{


		for (int j = 0; j < 19 * 19; j++)
		{
			printf("%f ", C.at<float>(i, j));

		}
		printf("\n");

	}
	Mat Coef = Mat(19 * 19, 19 * 19, CV_32FC1);
	fclose(F);

	F = fopen("corelatia.csv", "w");

	for (int i = 0; i < 19 * 19; i++)
	{
		for (int j = 0; j < 19 * 19; j++)
		{


			Coef.at<float>(i, j) = C.at<float>(i, j) / (sqrt(C.at<float>(i, i)) * sqrt(C.at<float>(j, j)));

			printf("%f ", Coef.at<float>(i, j));
			fprintf(F, "%f ", Coef.at<float>(i, j));
		}
		printf("\n");
		fprintf(F, "\n");
	}
	fclose(F);
	Mat Graf = Mat(255, 255, CV_8UC1);
	Graf.setTo(255);
	int r1 = 5;
	int c1 = 4;
	int r2 = 5;
	int c2 = 14;



	int ii = r1 * 19 + c1;
	int ij = r2 * 19 + c2;
	for (int i = 0; i < 400; i++)
	{

		Graf.at<uchar>(I.at<uchar>(i, ii), I.at<uchar>(i, ij)) = 0;

	}
	printf("ro: %f", Coef.at<float>(ii, ij));


	imshow("Graf de corelatie", Graf);
	waitKey();




}
int aprox(double x)
{
	if (x - (int)x >= 0.5)
		return (int)x + 1;

	return (int)x;


}
void PCA()
{

	FILE* f;
	f = fopen("PCA/pca2d.txt", "r");
	if (f)
	{
		int n, d;
		fscanf(f, "%d", &n);
		fscanf(f, "%d", &d);
		Mat_<double> X(n, d, CV_64FC1);
		std::vector<double> miu;
		for (int i = 0; i < n; i++)
		{


			for (int j = 0; j < d; j++)
			{
				fscanf(f, "%lf", &(X(i, j)));


			}

		}
		//find means
		for (int j = 0; j < d; j++)
		{
			double mean = 0;
			for (int i = 0; i < n; i++)
				mean += X.at<double>(i, j);

			mean = mean / n;
			miu.push_back(mean);

		}

		Mat X2(n, d, CV_64FC1);

		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < d; j++)
				X2.at<double>(i, j) = X.at<double>(i, j) - miu.at(j);

		}
		Mat C = X2.t() * X2 / (n - 1);

		Mat Lambda, Q;
		eigen(C, Lambda, Q);
		std::cout << Lambda;
		Q = Q.t();

		Mat Xcoef = X * Q;

		Mat Xk = Mat(n, d, CV_64FC1);
		Xk.setTo(0);
		int k = 1;

		for (int i = 0; i < k; i++)
		{
			Mat prod = X * Q.col(i) * (Q.col(i).t());
			Xk += prod;

		}


		Mat E = (X - Xk);
		float mean_error = 0;
		for (int i = 0; i < E.rows; i++)
			for (int j = 0; j < E.cols; j++)
				mean_error += abs(E.at<double>(i, j));

		mean_error /= (E.rows * E.cols);
		printf("\n eroarea medie este: %f\n", mean_error);

		std::vector<double> maxime, minime;
		for (int i = 0; i < d; i++)
		{
			int max = Xcoef.at<double>(0, i), min = Xcoef.at<double>(0, i);
			for (int j = 0; j < n; j++) {
				if (Xcoef.at<double>(j, i) < min)
					min = Xcoef.at<double>(j, i);

				if (Xcoef.at<double>(j, i) > max)
					max = Xcoef.at<double>(j, i);


			}
			maxime.push_back(max);
			minime.push_back(min);

		}
		printf("maxime: %f %f", maxime.at(1) - minime.at(1) + 20, maxime.at(0) - minime.at(0) + 20);
		Mat imagine_alb(maxime.at(0) - minime.at(0) + 20, maxime.at(1) - minime.at(1) + 20, CV_8UC1);
		imagine_alb.setTo(255);
		for (int i = 0; i < n; i++)
		{

			imagine_alb.at<uchar>(Xcoef.at<double>(i, 0) - minime.at(0) + 10, Xcoef.at<double>(i, 1) - minime.at(1) + 10) = 0;

		}

		imshow("img", imagine_alb);
		waitKey();
		k = 0;
		float p = 0.99;
		float sumd = 0;
		printf(" lAmbda: %d %d", Lambda.rows, Lambda.cols);
		for (int i = 0; i < d; i++)
			sumd += Lambda.at<double>(i, 0);

		float sumk = 0;
		do {
			k++;
			sumk = 0;
			for (int i = 0; i < k; i++)
				sumk += Lambda.at<double>(i, 0);

		} while ((sumk / sumd) < p);
		printf("k optim is %d", k);





	}


	//3DDDDDDDDDDDDDDDDDDDDDDDDDDDD
	f = fopen("PCA/pca3d.txt", "r");
	if (f)
	{
		int n, d;
		fscanf(f, "%d", &n);
		fscanf(f, "%d", &d);
		Mat_<double> X(n, d, CV_64FC1);
		std::vector<double> miu;
		for (int i = 0; i < n; i++)
		{


			for (int j = 0; j < d; j++)
			{
				fscanf(f, "%lf", &(X(i, j)));


			}

		}
		//find means
		for (int j = 0; j < d; j++)
		{
			double mean = 0;
			for (int i = 0; i < n; i++)
				mean += X.at<double>(i, j);

			mean = mean / n;
			miu.push_back(mean);

		}

		Mat X2(n, d, CV_64FC1);

		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < d; j++)
				X2.at<double>(i, j) = X.at<double>(i, j) - miu.at(j);

		}
		Mat C = X2.t() * X2 / (n - 1);

		Mat Lambda, Q;
		eigen(C, Lambda, Q);
		std::cout << Lambda;
		Q = Q.t();

		Mat Xcoef = X * Q;

		Mat Xk = Mat(n, d, CV_64FC1);
		Xk.setTo(0);
		int k = 1;

		for (int i = 0; i < k; i++)
		{
			Mat prod = X * Q.col(i) * (Q.col(i).t());
			Xk += prod;

		}


		Mat E = (X - Xk);
		float mean_error = 0;
		for (int i = 0; i < E.rows; i++)
			for (int j = 0; j < E.cols; j++)
				mean_error += abs(E.at<double>(i, j));

		mean_error /= (E.rows * E.cols);
		printf("\n eroarea medie este: %f\n", mean_error);

		std::vector<double> maxime, minime;
		for (int i = 0; i < d; i++)
		{
			int max = Xcoef.at<double>(0, i), min = Xcoef.at<double>(0, i);
			for (int j = 0; j < n; j++) {
				if (Xcoef.at<double>(j, i) < min)
					min = Xcoef.at<double>(j, i);

				if (Xcoef.at<double>(j, i) > max)
					max = Xcoef.at<double>(j, i);


			}
			maxime.push_back(max);
			minime.push_back(min);

		}



		printf("maxime: %f %f", maxime.at(1) - minime.at(1) + 20, maxime.at(0) - minime.at(0) + 20);
		Mat imagine_alb(maxime.at(1) - minime.at(1) + 20, maxime.at(0) - minime.at(0) + 20, CV_8UC1);
		imagine_alb.setTo(255);
		for (int i = 0; i < n; i++)
		{

			imagine_alb.at<uchar>(aprox(Xcoef.at<double>(i, 1) - minime.at(1)) + 10, aprox(Xcoef.at<double>(i, 0) - minime.at(0)) + 10) = aprox(255.0 - ((Xcoef.at<double>(i, 2) - minime.at(2)) / (maxime.at(2) - minime.at(2))) * 255.0);

		}

		imshow("img2", imagine_alb);
		waitKey();
		k = 0;
		float p = 0.99;
		double sumd = 0;
		printf(" lAmbda: %d %d", Lambda.rows, Lambda.cols);
		for (int i = 0; i < d; i++)
			sumd += Lambda.at<double>(i, 0);

		double sumk = 0;
		do {
			k++;
			sumk = 0;
			for (int i = 0; i < k; i++)
				sumk += Lambda.at<double>(i, 0);

		} while ((sumk / sumd) < p);
		printf("k optim is %d", k);


	}


}
double euclidean(Mat C1,Mat C2)
{
	double sum = 0;
	//cout << "dimensiune coloana si linie" << min(C1.cols, C2.cols) << " " << min(C1.rows, C2.rows) << endl;
	for(int i=0;i<min(C1.rows,C2.rows);i++)
		for (int j = 0; j < min(C1.cols, C2.cols);j++)
		{             
			sum += pow(C1.at<float>(i, j) - C2.at<float>(i, j), 2);

		}
	return sqrt(sum);
}
#include <random>
default_random_engine gen;

std::pair<Mat, Mat> kmeans(Mat X, int K)
{


	Mat C = Mat(K, X.cols, CV_32FC1);
	Mat L = Mat(X.rows, 1, CV_32SC1);
	L.setTo(0);

	uniform_int_distribution<int> distribution(0,X.rows);


	for (int k = 0; k < K; k++)
	{
		int randint = distribution(gen);
		X.row(randint).copyTo(C.row(k));

	}
	
	boolean changed = true;
	int max_iterations = 20;
	int nr_iter = 0;
	while (changed && nr_iter < max_iterations)
	{
		cout << "iter:" << nr_iter;
		nr_iter++;
		changed = false;
		//asignare

		for (int i = 0; i < X.rows; i++)
		{
			double min_dist = euclidean(X.row(i), C.row(0));

			for (int k = 0; k < K; k++)
			{


				double dk = euclidean(X.row(i), C.row(k));
				if (dk < min_dist)
				{
					min_dist = dk;
					L.at<int>(i, 0) = k;
					changed = true;

				}
			}
		}
		//cout << "nr iter " << nr_iter << " " << C << " " << L << endl;
		//system("pause");
		//actualizare
		for (int k = 0; k < K; k++)
		{

			for (int j = 0; j < C.cols; j++)
			{
				float s = 0;
				int ct = 0;
				for (int i = 0; i < X.rows; i++)
				{
					
					if (L.at<int>(i, 0) == k)
					{
						ct++;
						s = s + X.at<float>(i, j);
					}
				}
				s = s / ct;
				C.at<float>(k, j) = s;
			}
		}



	}

	pair<Mat, Mat> p = pair<Mat, Mat>(C, L);
	return p;


}

void kMeans()
{
	//for grayscale image 
	std::vector<cv::Point> points;
	FILE* F;
	Mat img = imread("KMe/points5.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("imginit", img);
	waitKey();
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{

			if (img.at<uchar>(i, j) < 100)
			{
				points.push_back({ j, i });
			}

		}
	Mat X = Mat(points.size(), 2, CV_32FC1);
	for (int i = 0; i < points.size(); i++)
	{
		X.at<float>(i, 0) = points.at(i).x;
		X.at<float>(i, 1) = points.at(i).y;
	}
	cout << X;
	int K = 5;
	pair<Mat, Mat> p = kmeans(X, K);
	Mat C = p.first;
	Mat L = p.second;

	cout << "C" << C;
	cout << "L:" << L;

	Mat newImg(img.rows, img.cols, CV_8UC3);
	newImg.setTo(Scalar(255, 255, 255));

	for (int i = 0; i < K; i++)
	{
		circle(newImg, { (int)C.at<float>(i,0),(int)C.at<float>(i,1) }, 10, Scalar(255 * (i == 0), 255 * (i == 1), 255 * (i == 2)), 5);
	}


	for (int i = 0; i < points.size(); i++)
	{
		if (L.at<int>(i, 0) == 0)
			newImg.at<Vec3b>(points.at(i).y, points.at(i).x) = Vec3b(255, 0, 0);
		if (L.at<int>(i, 0) == 1)
			newImg.at<Vec3b>(points.at(i).y, points.at(i).x) = Vec3b(0, 255, 0);
		if (L.at<int>(i, 0) == 2)
			newImg.at<Vec3b>(points.at(i).y, points.at(i).x) = Vec3b(0, 0, 255);
		if (L.at<int>(i, 0) == 3)
			newImg.at<Vec3b>(points.at(i).y, points.at(i).x) = Vec3b(255, 255, 0);
		if (L.at<int>(i, 0) == 4)
			newImg.at<Vec3b>(points.at(i).y, points.at(i).x) = Vec3b(0, 255, 255);
	}
	imshow("result img", newImg);
	waitKey();


	//3d imagini colorate



	img = imread("KMe/img04.jpg", CV_LOAD_IMAGE_COLOR);
	Mat newimg = Mat(img.rows, img.cols, CV_8UC3);



	X = Mat(img.rows * img.cols, 3, CV_32FC1);
	int cnt = 0;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			X.at<float>(cnt, 0) = img.at<Vec3b>(i, j)[0];
			X.at<float>(cnt, 1) = img.at<Vec3b>(i, j)[1];
			X.at<float>(cnt, 2) = img.at<Vec3b>(i, j)[2];
			cnt++;
		}


	imshow("imginit", img);
	waitKey();


	K = 10;

	Vec3b colors[10];

	uniform_int_distribution<int> distribution(0, 255);
	for (int i = 0; i < K; i++)
		colors[i] = { (uchar)distribution(gen),
		 (uchar)distribution(gen),
		 (uchar)distribution(gen) };
	cout << "enter in knn";

	p = kmeans(X, K);

	cout << "out from  kmeans";
	C = p.first;
	L = p.second;
	int cnt2 = 0;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			newimg.at<Vec3b>(i, j) = Vec3b(C.at<float>(L.at<int>(cnt2), 0),
				C.at<float>(L.at<int>(cnt2), 1), C.at<float>(L.at<int>(cnt2), 2));
			cnt2++;

		}
	imshow("image color with kmeans", newimg);
	waitKey();



}


using namespace std;

float dist(float* hist, Mat hist2, int len = 3 * 256)
{
	float dist = 0;
	//cout << hist2 << endl;
	for (int i = 0; i < len; i++)
	{
		dist += abs(hist[i] - hist2.at<float>(0,i));
	}
	return sqrt(dist);

}
void calcHist(Mat X,float* hist,int nr_bins=256)
{
	int wchanel = nr_bins;

	for (int i = 0; i < 3*nr_bins; i++)
		hist[i] = 0;
	cout << "not here"<<endl;
	for(int i=0;i<X.rows;i++)
		for (int j = 0; j < X.cols; j++)
		{

			int b = X.at<Vec3b>(i, j)[0];
			int g = X.at<Vec3b>(i, j)[1];
			int r = X.at<Vec3b>(i, j)[2];
			hist[b]++;
			hist[wchanel + g]++;
			hist[2 * wchanel + r]++;

		}
	for (int i = 0; i < 3*nr_bins; i++)
		hist[i] = hist[i] / (X.rows * X.cols);
	cout << "not there" << endl;

}





int classify(Mat img,Mat X,Mat y, int k, int c,int nr_bins=256)
{

	int u = 3 * nr_bins;
	float hist[3*256];

	calcHist(img, hist, nr_bins);
	cout << "classify";


	vector < pair<double, int>> v;
	for (int i = 0; i < X.rows; i++)
	{
		
		v.push_back({ (double)dist(hist, X.row(i)),y.at<uchar>(i)});
	
	}
 
	sort(v.begin(), v.end());
	//for (int i = 0; i < v.size(); i++)
		//cout << v.at(i).first << " " << v.at(i).second<<endl;

	int voturi[6];
	
	for (int i = 0; i < c; i++)
		voturi[i] = 0;
	
	for (int j = 0; j < k; j++)
	{
	
		voturi[v.at(j).second]++;

	}
	
	int index_max=0;
	int max = 0;
	for(int i=0;i<c;i++)
		if (voturi[i] >= max)
		{
			max = voturi[i];
			index_max = i;
		}
	cout << "classified";

	return index_max;

}

void KNN_main()
{
	const int nrclasses = 6;
	char classes[nrclasses][10] = { "beach", "city","desert","forest","landscape","snow" };



	int nrinst = 679;
	int feature_dim = 3 * 256;
	Mat X(nrinst, feature_dim, CV_32FC1);
	Mat y(nrinst, 1, CV_8UC1);
	char fname[50];

	int c = 0, fileNr = 0, rowX = 0;
	float hist[3 * 256 + 1];
	int hist_size = 3 * 256;
	Mat C(nrclasses, nrclasses, CV_32FC1);
	C.setTo(0);
	int im = 0;
	for (int c = 0; c < 6; c++)
	{
		fileNr = 0;
		while (1) {
			im++;

			sprintf(fname, "KNN/train/%s/%06d.jpeg", classes[c], fileNr++);
			Mat img = imread(fname);


			if (img.cols == 0) break;


			calcHist(img, hist);

			for (int d = 0; d < hist_size; d++)
				X.at<float>(rowX, d) = hist[d];
			y.at<uchar>(rowX) = c;
			rowX++;
		}
	}
	cout << "nr files=" << im;
	cout << y;
	//testare
	int k = 5;
	int imt = 0;
	Mat img;

	for (int c = 0; c < 6; c++)
	{
		fileNr = 0;
		while (1)
		{
			imt++;
			sprintf(fname, "KNN/test/%s/%06d.jpeg", classes[c], fileNr++);
			img = imread(fname);

			if (img.cols == 0 || img.rows == 0) break;



			//imshow("img test", img);
			//waitKey();



			int p = classify(img, X, y, k, c);
			//if(p<6)
			C.at<float>(c, p)++;



			rowX++;
		}


	}

	float acc = 0;
	float den = 0;
	float dem = 0;
	cout << C;
	for (int i = 0; i < 6; i++)
	{


		den += C.at<float>(i, i);
		for (int j = 0; j < 6; j++)
			dem += C.at<float>(i, j);
	}
	cout << "den dem" << den << " " << dem << endl;
	acc = den / dem;
	cout << "accuracy of" << acc;

}





Mat read_X()
{

}

Mat binarizare(Mat img,int prag=128)
{
	
	Mat img2 = img.clone();
	for(int i=0;i<img.rows;i++)
		for (int j = 0; j < img.rows; j++)
		{
			if (img.at<uchar>(i, j) > prag)
				img2.at<uchar>(i, j) = 255;
			else
				img2.at<uchar>(i, j) = 0;

		}
	return img2;
}


void Bayes_training(int C, int d, Mat* likelihood, Mat* prior,int limit=100)
{
	char fname[256];

	int index = 0;
	int count_img_total=0;
	if (limit < 0)
		limit = INT_MAX;

	for (int c = 0; c < C; c++)
	{
		
		index = 0;
		while (index < limit) {

			count_img_total++;
			sprintf(fname, "Bayes/train/%d/%06d.png", c, index);
			Mat img = imread(fname, 0);
			img = binarizare(img);
			//imshow("binarised", img);
			//waitKey();

			if (img.cols == 0) break;

			int w = img.cols;


			
			for(int i=0;i<img.rows;i++)
				for (int j = 0; j < img.cols; j++)
				{
					if (img.at<uchar>(i, j) == 255)
					{
						likelihood->at<double>(c, i *w+ j)++;

					}
					
				}

			
			//process img
			index++;
		}
		for (int i = 0; i < d; i++)
			likelihood->at<double>(c, i) = (likelihood->at<double>(c, i) + 1) / (index + C);
		(*prior).at<double>(c, 0) = ((double)index);
	}

	
	cout << "likelihood:" << *likelihood << endl;


	for (int c = 0; c < C; c++)
	{
		(*prior).at<double>(c, 0) = (*prior).at<double>(c, 0) / count_img_total;
	}

}


int  classifyBayes(Mat img, Mat priors, Mat likelihood)
{
	vector<pair<double,int>> probabilitati;
	for (int c = 0; c < priors.rows; c++)
	{
		double probabilitate = log(priors.at<double>(c, 0));
		for (int feature = 0; feature < likelihood.cols; feature++)
			if(img.at<uchar>(feature/28,feature%28)==255)
			probabilitate += log(likelihood.at<double>(c, feature));
			else
			probabilitate += log(1-likelihood.at<double>(c, feature));

		probabilitati.push_back({ probabilitate,c });

	}
	sort(probabilitati.begin(), probabilitati.end(), greater< pair<int, double>>());
	return probabilitati.at(0).second;


}
//returns matricea de confuzie si acuratetea
pair<Mat,double> clasifyBayesALLTests(Mat priors,Mat likelihood,int C=2,int limit = 100)
{

	Mat Confussion = Mat(priors.rows, priors.rows, CV_64FC1);
	Confussion.setTo(0);
	char fname[256];
	if (limit == -1)
		limit = INT_MAX;


	for (int c = 0; c < C; c++)
	{

		int index = 0;
		while (index < limit) {


			sprintf(fname, "Bayes/test/%d/%06d.png", c, index);
			Mat img = imread(fname, 0);
			img = binarizare(img);



			if (img.cols == 0) break;

			int guessed_class = classifyBayes(img, priors, likelihood);
//			imshow("test_img", img);
			cout <<"guess:"<< guessed_class << endl;
	//		waitKey();
			Confussion.at<double>(c, guessed_class)++;



			//process img
			index++;
		}
	}
	int truePositive = 0;
	int all = 0;
	for (int ci = 0; ci < C; ci++)
	{
		truePositive += Confussion.at<double>(ci, ci);
		for (int cj = 0; cj < C; cj++)
			all += Confussion.at<double>(ci, cj);


	}
			

	double Accuracy = ((double)truePositive) / all;
	pair<Mat, double> result = { Confussion,Accuracy };
	return result;
}
void test_Baysian()
{
	int C = 10;
	int d = 28 * 28;
	Mat priors(C, 1, CV_64FC1);
	Mat likelihood(C, d, CV_64FC1);
	Bayes_training(C, d, &likelihood, &priors, -1);
	cout << "training done";
	pair<Mat, double> results = clasifyBayesALLTests(priors, likelihood, C, -1);
	cout << "Confuusion:" << results.first << endl;
	cout << "Accuracy:" << results.second;

}


double OnlinePerceptron(Mat img,Mat X, Mat* W, Mat Y,double l_r, int max_iter,double E_limit, double l_r_t1=-1)
{
	double E = 0;
	for (int iter = 0; iter < max_iter; iter++)
	{



		E = 0;
		if (l_r_t1 < 0)
			l_r_t1 = l_r;




		for (int i = 0; i < X.rows; i++)
		{
			double zi = 0;
		

			for (int j = 0; j < X.cols; j++)
				zi += (*W).at<double>(0, j) * X.at<double>(i, j);


			
			if (zi * Y.at<double>(i, 0) <= 0)
			{
				(*W).at<double>(0, 0) = (*W).at<double>(0, 0) + l_r_t1 * Y.at<double>(i, 0) * X.at<double>(i, 0);
				for (int k = 1; k < X.cols; k++)
					(*W).at<double>(0, k) = (*W).at<double>(0, k) + l_r * Y.at<double>(i, 0) * X.at<double>(i, k);
				
				E = E + 1;
			}


		}
		cout << "online: W:" << *W << endl;
		E = E / X.rows;

		Mat img2 = img.clone();
		double a = (*W).at<double>(0, 0);//*1
		double b = (*W).at<double>(0, 1);//*x
		double c= (*W).at<double>(0, 2);//*y


		if (abs(c) > abs(b))
		{
			double y1 = -a / c;
			double y2 = (-a - b * (img.cols - 1)) / c;


			//compute for x==0 and x== X.rows
			line(img2, { 0,(int)y1 }, { img.cols - 1,(int)y2 }, { 0,255,0 });
		}
		else
		{
			cout << "with x version"<<endl;
			double x1 = -a / b;
			double x2 = (-a - c * (img.rows - 1)) / b;


			//compute for x==0 and x== X.rows
			line(img2, { (int)x1,0}, {(int)x2,img.rows - 1}, {0,255,0});

		}




		namedWindow("imagine_modificiata", WINDOW_KEEPRATIO);
		imshow("imagine_modificiata", img2);
		waitKey(100);

		if (E < E_limit)
			break;

	}
	return E;
}
double BatchPerceptron(Mat img,Mat X, Mat* W, Mat Y, double l_r, int max_iter, double E_limit,double l_r_t1 = -1)
{
	double E = 0, L = 0;
	if (l_r_t1 < 0)
		l_r_t1 = l_r;
	for (int iter = 0; iter < max_iter; iter++)
	{
		E = 0;
		L = 0;
		
		Mat dL = Mat(1, X.cols, CV_64FC1);
		dL.setTo(0);

		for (int i = 0; i < X.rows; i++)
		{
			double zi = 0;
			for (int j = 0; j < X.cols; j++)
				zi += (*W).at<double>(0, j) * X.at<double>(i, j);

			if (zi * Y.at<double>(i, 0) <= 0)
			{

				(*W).at<double>(0, 0) = (*W).at<double>(0, 0) + l_r_t1 * Y.at<double>(i, 0) * X.at<double>(i, 0);
				for (int k = 1; k < X.cols; k++)
					(*W).at<double>(0, k) = (*W).at<double>(0, k) + l_r * Y.at<double>(i, 0) * X.at<double>(i, k);
				
				E = E + 1;
				L = L - Y.at<double>(i, 0) * zi;
				dL = dL - Y.at<double>(i, 0) * X.row(i);
			}


		}
		cout << "batch: W:" << *W << endl;
		E = E *( 1.0 / X.rows);
		L = L * (1.0 / X.rows);
		dL=dL* (1.0 / X.rows);

		Mat img2 = img.clone();

		double a = (*W).at<double>(0, 0);//*1
		double b = (*W).at<double>(0, 1);//*x
		double c = (*W).at<double>(0, 2);//*y
		if (abs(c) > abs(b))
		{
			double y1 = -a / c;
			double y2 = (-a - b * (img.cols - 1)) / c;


			//compute for x==0 and x== X.rows
			line(img2, { 0,(int)y1 }, { img.cols - 1,(int)y2 }, { 0,255,0 });
		}
		else
		{

			double x1 = -a / b;
			double x2 = (-a - c * (img.rows - 1)) / b;


			//compute for x==0 and x== X.rows
			line(img2, { (int)x1,0 }, { (int)x2,img.rows - 1 }, { 0,255,0 });

		}

		namedWindow("imagine_modificiata_batch", WINDOW_KEEPRATIO);
		imshow("imagine_modificiata_batch", img2);
		waitKey(100);







		if (E < E_limit)
			break;
		(*W) = (*W) - l_r * dL;

	
	}
	return E;
}



void Perceptron()
{
	char fname[256];





	for (int index = 0; index <= 6; index++)
	{
		sprintf(fname, "Perceptron/test%02d.bmp", index);
		cout << fname << endl;
		Mat img = imread(fname, IMREAD_COLOR);
		namedWindow("imagine_original", WINDOW_KEEPRATIO);
		imshow("imagine_original", img);
		waitKey(1000);

		vector<pair<Point, int>> temp_xy;
		int nr_points = 0;



		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++)
			{
				//punct rosu
				if (img.at<Vec3b>(i, j)[2] == 255 && img.at<Vec3b>(i, j)[0] == 0 && img.at<Vec3b>(i, j)[1] == 0)
				{
					nr_points++;
					temp_xy.push_back({ {j,i},1 });
				}
				if (img.at<Vec3b>(i, j)[0] == 255 && img.at<Vec3b>(i, j)[2] == 0 && img.at<Vec3b>(i, j)[1] == 0)
				{
					nr_points++;
					temp_xy.push_back({ {j,i},-1 });
				}
			}

		Mat X(nr_points, 3, CV_64FC1);
		Mat Y(nr_points, 1, CV_64FC1);
		for (int i = 0; i < nr_points; i++)
		{
			X.at<double>(i, 0) = 1;
			X.at<double>(i, 1) = temp_xy.at(i).first.x;
			X.at<double>(i, 2) = temp_xy.at(i).first.y;
			Y.at<double>(i, 0) = temp_xy.at(i).second;
		}
		//init W
		Mat W = Mat(1, 3, CV_64FC1);
		W.at<double>(0, 0) = 0.1;
		W.at<double>(0, 1) = 0.1;
		W.at<double>(0, 2) = 0.1;
		double learning_rate = pow(10, -4);
		double E_limit = pow(10, -5);
		double max_iter = pow(10, 5);
		cout << "img" << index << endl;
		OnlinePerceptron(img, X, &W, Y, learning_rate, max_iter, E_limit, 0.1);
		W = Mat(1, 3, CV_64FC1);
		W.at<double>(0, 0) = 0.1;
		W.at<double>(0, 1) = 0.1;
		W.at<double>(0, 2) = 0.1;
		BatchPerceptron(img, X, &W, Y, learning_rate, max_iter, E_limit, 0.1);

	}


}


using namespace cv;

struct weaklearner {
	int feature_i;
	int threshold;
	int class_label;
	float error;
	int classify(Mat X) {
		if (X.at<double>(feature_i) < threshold)
			return class_label;
		else
			return -class_label;
	}
};


weaklearner findWeakLearner(Mat img, Mat X, Mat y, Mat w)
{
	weaklearner best_h = {};
	int classLabels[2] = { -1,1 };
	double best_err = 1000000;
	for (int j = 0; j < X.cols; j++)
		for (int treshhold = 0; treshhold < max(img.cols, img.rows); treshhold++)
			for (int index = 0; index < 2; index++)
			{
				int classLabel = classLabels[index];
				double e = 0;
				double zi = 0;
				for (int i = 0; i < X.rows; i++)
				{
					if (X.at<double>(i, j) < treshhold)
					{
						zi = classLabel;
					}
					else
					{
						zi = -classLabel;
					}
					if (zi * y.at<double>(i) < 0)
					{
						e += w.at<double>(i);
					}
				}
				if (e < best_err)
				{
					best_err = e;
					best_h = { j,treshhold,classLabel,(float)e };


				}

			}

	return best_h;
}











#define MAXT 500
struct classifier {
	int T;
	float alphas[MAXT];
	weaklearner hs[MAXT];


	void train(Mat img,Mat Xs,Mat ys,int t)
	{

		T = t;
		int n = Xs.rows;
		Mat W(n, 1, CV_64FC1);
		W.setTo(1.0 / n);

		for (int i = 0; i < T; i++)
		{
			
			weaklearner wl = findWeakLearner(img, Xs, ys, W);
			hs[i] = wl;

			double err = wl.error;
			alphas[i] = 0.5 * log((1 - err) / err);
			double s = 0;
			for (int j = 0; j < Xs.rows; j++)
			{
				W.at<double>(j, 0) = W.at<double>(j, 0) * exp(-alphas[i] * ys.at<double>(j, 0) * wl.classify(Xs.row(j)));
				s += W.at<double>(j, 0);
			}
			for (int j = 0; j < Xs.rows; j++)
			{
				W.at<double>(j, 0) = W.at<double>(j, 0) / s;
			}
		}





	}







	int classify(Mat X) {
		
		float Ht = 0;
		for (int i = 0; i < T; i++)
		{
			Ht += alphas[i] * hs[i].classify(X);

		}
		return (Ht > 0) ? 1 : -1;

	}
};

void drawBoundary(Mat img, classifier clf)
{
	Mat img2 = img.clone();
	for(int i=0;i<img2.rows;i++)
		for (int j = 0; j < img2.cols; j++)
		{

			if (img.at<Vec3b>(i, j)[0] == 255 && img.at<Vec3b>(i, j)[2] == 255 && img.at<Vec3b>(i, j)[1] == 255) {
				Mat X(1, 2, CV_64FC1);
				X.at<double>(0, 0) = j;
				X.at<double>(0, 1) = i;

                int classg = clf.classify(X);
				if (classg < 0)
				{
					img2.at<Vec3b>(i, j) = { 255,255,0 };
				}
				else
				{
					img2.at<Vec3b>(i, j) = { 0,255,255 };
				}

			}
		}

	  imshow("imagine clasificata", img2);
	  waitKey(1000);
		}
	


int main()
{
	char fname[256];
	char SVM[256] = "SVM/test0";
	char Ada[256] = "Ada/points";


		for (int index = 0; index < 6; index++)
		{
			sprintf(fname, "%s%d.bmp",Ada, index);
			Mat img = imread(fname, IMREAD_COLOR);
			namedWindow("imagine_original", WINDOW_KEEPRATIO);
			imshow("imagine_original", img);
			waitKey(1000);

			vector<pair<Point, int>> temp_xy;
			int nr_points = 0;



			for (int i = 0; i < img.rows; i++)
				for (int j = 0; j < img.cols; j++)
				{
					//punct rosu
					if (img.at<Vec3b>(i, j)[2] == 255 && img.at<Vec3b>(i, j)[0] == 0 && img.at<Vec3b>(i, j)[1] == 0)
					{
						nr_points++;
						temp_xy.push_back({ {j,i},1 });
					}
					if (img.at<Vec3b>(i, j)[0] == 255 && img.at<Vec3b>(i, j)[2] == 0 && img.at<Vec3b>(i, j)[1] == 0)
					{
						nr_points++;
						temp_xy.push_back({ {j,i},-1 });
					}
				}

			Mat X(nr_points, 2, CV_64FC1);
			Mat Y(nr_points, 1, CV_64FC1);
			for (int i = 0; i < nr_points; i++)
			{
				X.at<double>(i, 0) = temp_xy.at(i).first.x;
				X.at<double>(i, 1) = temp_xy.at(i).first.y;
				Y.at<double>(i, 0) = temp_xy.at(i).second;
			}
			//init W
			classifier AdaBoost;
			int t= 200;
			AdaBoost.train(img, X, Y, t);
			drawBoundary(img, AdaBoost);




		}

		waitKey();



	
	
}

