/*
function： 检测柱子托盘中心坐标
date：2016.11.9
author:LHMD.NUC
*/

#include"opencv2/opencv.hpp"
using namespace cv;
using namespace std;

/*---------------------全局变量声明------------------------------*/
Mat src;
Mat ROI, changban;
int low_thresd = 34;//canny阈值
int high_thresd = 74;//100也行貌似

					 /*!!!!!!以下是要根据摄像头来更改的参数!!!!!!*/
					 //检测柱子轮廓ROI区域内的参数；参数从左到右分别是ROI区域左边顶点坐标x、y、宽度dx、高度dy
					 //这个区域需要包括柱子的左右边界
uint16_t x = 536, y = 1003, dx = 536, dy = 115;

//seek_y()中所用参数
uint16_t x_limit = 640;//截取长板ROI区域时x坐标的判断值,
uint16_t x1 = 140;
uint16_t y_cb = 1050, dx_cb = 500, dy_cb = 186;//检测长板截取ROI区域的参数，别是ROI的左顶点的y坐标，ROI长度和ROI的高度

											   /*!!!!!!以上是要根据摄像头来更改的参数!!!!!!*/

Mat abs_grad_x, abs_grad_y, dst;//创建 abs_grad_x 和 abs_grad_y 矩阵
Mat abs_grad_x1, abs_grad_y1, dst1;//创建 abs_grad_x 和 abs_grad_y 矩阵

Mat img_canny;
Mat hough_erode;
Mat midImage;
int central_x = 0;
int central_y = 0;

vector<Vec2i>mate(10);//创建一个容器存储在上面匹配到的【maxik，xi】---i表示数字

					  /*---------------------函数声明------------------------------*/
void draw_ROI(const Mat& src, Mat& ROI, int x1, int y1, int dx, int dy);
void strengthen(const Mat& ROI, Mat& result);
void sobel(const Mat& result, Mat& abs_grad_x, Mat& abs_grad_y, Mat& dst);
static void on_canny(int, void*);
void canny(Mat& img, Mat& img_canny, int low_thresd, int high_thresd);
void ThinImage(void);
void seek_y(int &central_x, Mat& changban);

/*--------------对图像中的线段进行细化，到线段的最小宽度----------------------------*/
cv::Mat thinImage(const Mat & src, const int maxIterations = -1)
{
	assert(src.type() == CV_8UC1);
	cv::Mat dst;
	int width = src.cols;
	int height = src.rows;
	src.copyTo(dst);
	int count = 0;  //记录迭代次数
	while (true)
	{
		count++;
		if (maxIterations != -1 && count > maxIterations) //限制次数并且迭代次数到达
			break;
		std::vector<uchar *> mFlag; //用于标记需要删除的点
									//对点标记
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//如果满足四个条件，进行标记
				//  p9 p2 p3
				//  p8 p1 p4
				//  p7 p6 p5
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
					{
						//标记
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//将标记的点删除
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//直到没有点满足，算法结束
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//将mFlag清空
		}

		//对点标记
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//如果满足四个条件，进行标记
				//  p9 p2 p3
				//  p8 p1 p4
				//  p7 p6 p5
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);

				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0)
					{
						//标记
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//将标记的点删除
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//直到没有点满足，算法结束
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//将mFlag清空
		}
	}
	return dst;
}


int main(int argc, char* argv[])
{
	int Open = 0;//open==1，处理视频；open==0，处理图片
	if (Open == 1)
	{
		int m = 0;
		VideoCapture capture("5.avi");
		if (!capture.isOpened())
		{
			cout << "\nfail to open video!\n" << endl;
			return 0;
		}
		//获取整个帧数
		long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
		cout << "整个视频共" << totalFrameNumber << "帧" << endl;
		double rate = capture.get(CV_CAP_PROP_FPS);
		cout << "帧率为:" << rate << endl;
		while (1)
		{
			if (m == totalFrameNumber)//视频播放完了则跳出循环
			{
				cout << "视频已经播放完了哦哈哈~" << endl;
				break;
			}
			capture >> src;
			m++;
			if ((src.cols < (x + dx)) || (src.rows < (y + dy)))//这里对图片大小有要求，更换相机和图片时可能要改
			{
				printf("图片太小，不满足选取ROI区域的要求，请检查下图片！");
				return false;
			}
			draw_ROI(src, ROI, x, y, dx, dy);//提取ROI区域

											 //增强图片的对比度（现在暂时不加入这步）
											 /*Mat result;strengthen(ROI, result);*/

											 //------------用sobel算子和canny算子对图片进行边缘检测得到柱子轮廓---------------
			sobel(ROI, abs_grad_x, abs_grad_y, dst);
			canny(abs_grad_x, img_canny, low_thresd, high_thresd);
			//imshow("【效果图】Canny边缘检测", img_canny);

			//------------霍夫变换得到柱子的轮廓线段-----------------------------------------
			Mat midImage, dstImage;//临时变量和目标图的定义
			Canny(img_canny, midImage, 50, 80, 3);//进行一此canny边缘检测
			cvtColor(midImage, dstImage, CV_GRAY2BGR);//转化边缘检测后的图为灰度图
			vector<Vec4i> lines(40);//定义一个矢量结构lines用于存放得到的线段矢量集合
			HoughLinesP(midImage, lines, 1, CV_PI / 180, 20, 15, 10);
			printf("\n%d\n", lines.size());
			if (lines.size() == 0)//在第一次霍夫变换中若没有检测到线段则不进行下面的检测
			{
				printf("\n第一次霍夫变换没有直线，不再进行下一步处理!\n");
			}
			else
			{
				for (size_t i = 0; i < lines.size(); i++)//将检测到的线段绘制到图中
				{
					Vec4i l = lines[i];
					line(dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 255, 255), 3, CV_AA);
				}
				Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
				Mat erode1;
				erode(dstImage, erode1, element);
				//namedWindow("erode1", 1);
				//imshow("erode1", erode1);
				imwrite("erode1.jpg", erode1);//将腐蚀后的图片保存到文件夹中
				ThinImage();
				waitKey(1);//这里加个小延时就可以显示出来了，不加waitkey图片无法正常显示

				seek_y(central_x, changban);
				waitKey(1);
				if (central_x != x)
					line(src, Point(central_x, 0), Point(central_x, src.cols), Scalar(0, 0, 255), 3, CV_AA);
				if (central_y != 0)
					line(src, Point(0, central_y), Point(src.rows, central_y), Scalar(0, 255, 0), 3, CV_AA);
				if (central_x != x&&central_y != 0)
					circle(src, Point(central_x, central_y), 15, Scalar(255, 0, 0), -1, CV_AA);//画一个实心圆
				namedWindow("SRC", 0);//估计是这张图片太大，一旦显示原图则在视频中其他图片无法显示
				imshow("SRC", src);
				central_y = 0;
			}

			waitKey(1);
		}
	}
	else//open=0，处理图片
	{
		src = imread("24.jpg");
		if (!src.data)
		{
			printf("读取图片错误，be sure that the picture is in the document.");
			return 0;
		}
		if ((src.cols < (x + dx)) || (src.rows < (y + dy)))
		{
			printf("图片太小，不满足选取ROI区域的要求，请检查下图片！");
			return 0;
		}

		draw_ROI(src, ROI, x, y, dx, dy);		//提取ROI区域
												//imshow("ROI", ROI);

												//增强图片的对比度（现在暂时不加入这步）
												/*Mat result;
												strengthen(ROI, result);*/

		sobel(ROI, abs_grad_x, abs_grad_y, dst);//用sobel算子对图片进行边缘检测
												//imshow("abs_grad_x", abs_grad_x);

		canny(abs_grad_x, img_canny, low_thresd, high_thresd);
		//imshow("【效果图】Canny边缘检测", img_canny);
		///////////////////////////////////////////////////////////////////////////

		////////////////////////////////////////////////////////////////////////
		Mat midImage, dstImage;//临时变量和目标图的定义
		Canny(img_canny, midImage, 50, 80, 3);//进行一此canny边缘检测
		cvtColor(midImage, dstImage, CV_GRAY2BGR);//转化边缘检测后的图为灰度图
												  //imshow("midImage", midImage);
		vector<Vec4i> lines(100);//注意内存的溢出，换不同图片的时候要考虑定义一个矢量结构lines用于存放得到的线段矢量集合
		HoughLinesP(midImage, lines, 1, CV_PI / 180, 20, 15, 10);
		//cout << "sobel、canny之后检测到的线段数目：" << lines.size() << endl;
		if (lines.size() == 0)
		{
			printf("第一次霍夫变换没有直线，不再进行下一步处理!");
		}
		else

		{
			for (size_t i = 0; i < lines.size(); i++)
			{
				Vec4i l = lines[i];
				line(dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 255, 255), 3, CV_AA);
			}
			////只有在前面检测到了直线的情况下才进行腐蚀和和细化
			Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
			Mat erode1;
			erode(dstImage, erode1, element);
			namedWindow("erode", 1);
			//imshow("erode", erode1);
			imwrite("erode1.jpg", erode1);
			ThinImage();
			seek_y(central_x, changban);
			if (central_x != x)
				line(src, Point(central_x, 0), Point(central_x, 1236), Scalar(0, 0, 255), 3, CV_AA);
			if (central_y != 0)
				line(src, Point(0, central_y), Point(1624, central_y), Scalar(0, 0, 255), 3, CV_AA);
			if (central_x != x&&central_y != 0)
				circle(src, Point(central_x, central_y), 15, Scalar(255, 0, 0), -1, CV_AA);//画一个实心圆

			namedWindow("SRC", 0);
			imshow("SRC", src);
		}

	}
	waitKey(0);
	return 0;

}

/*--------------------function:在原图中选取合适ROI区域-----------------------------------------*/
void draw_ROI(const Mat& src, Mat& ROI, int x, int y, int dx, int dy)
{
	ROI = src(Rect(x, y, dx, dy));
	//cout << "截取ROI区域成功！" << endl;
}

/*---------------------function:对ROI图片进行细节增强-------------------------------------------*/
void strengthen(const Mat& ROI, Mat& result)
{

}

/*---------------------function:sobel算子对图片进行轮廓检测--------------------------------------*/
void sobel(const Mat& result, Mat& abs_grad_x, Mat& abs_grad_y, Mat& dst)
{
	Mat grad_x, grad_y;
	Sobel(result, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);//求 X方向梯度
	convertScaleAbs(grad_x, abs_grad_x);//转换为绝对值输出

	Sobel(result, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);//求Y方向梯度
	convertScaleAbs(grad_y, abs_grad_y);

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);//合并梯度(近似)
	cout << "sobel算子检测成功！" << endl;


}
//void on_canny(int, void*,int )
//{
//	canny(abs_grad_x, img_canny, low_thresd, high_thresd);
//	imshow("【效果图】Canny边缘检测", img_canny);
//////////////////////////////////////////////////////////////////////////
//	Mat midImage, dstImage;//临时变量和目标图的定义
//    Canny(img_canny, midImage, 50, 80, 3);//进行一此canny边缘检测
//	cvtColor(midImage, dstImage, CV_GRAY2BGR);//转化边缘检测后的图为灰度图
//	imshow("zaici canny", dstImage);
//	vector<Vec4i> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合
//	HoughLinesP(midImage, lines, 1, CV_PI / 180, 20, 15, 10);
/////////////////////////////////////////////////////////////////////////
//	//midImage = midImage1.clone();
//	//hough_erode = img_canny.clone();          //midImage.copyTo(dstImage);
//
//	//Canny(img_canny, midImage, 50, 80, 3);//进行一此canny边缘检测
//	////cvtColor(midImage, midImage, CV_GRAY2BGR);//转化边缘检测后的图为灰度图
//	//vector<Vec4i> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合
//	//HoughLinesP(midImage, lines, 1, CV_PI / 180, 20, 15, 10);
//	///*【4】依次在图中绘制出每条线段*/
//	//printf("%d,d%",lines.size());
//	//if (lines.size() == 0)
//	//{
//	//	;
//	//}
//	//else
//
//	//{
//	//	for (size_t i = 0; i < lines.size(); i++)
//	//	{
//	//		Vec4i l = lines[i];
//	//		line(midImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
//	//	}
//	//	printf("bugbugbugbugbug");
//	//}
//	//imshow("midImage", midImage);
//	//printf("xinxinxinxxinxixn");
//
//
//	//Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
//	////Mat out;
//	//erode(midImage, midImage, element);
//	//namedWindow("erode", 0);
//	//imshow("erode", midImage);
//	//houghline_erode(img_canny, hough_erode);
//}

/*---------------------function:canny边缘检测-----------------------------------------------------*/
void canny(Mat& img, Mat& img_canny, int low_thresd, int high_thresd)
{
	cvtColor(img, img_canny, CV_BGR2GRAY);
	blur(img_canny, img_canny, Size(3, 3));
	Canny(img_canny, img_canny, low_thresd, high_thresd, 3);
	//imshow("【效果图】Canny边缘检测", img_canny);
}

void houghline_erode(Mat& midImage1, Mat& hough_erode)
{
	//midImage= midImage1.clone();
	//hough_erode = midImage1.clone();          //midImage.copyTo(dstImage);
	Canny(midImage1, midImage1, 50, 80, 3);//进行一此canny边缘检测
	cvtColor(midImage1, hough_erode, CV_GRAY2BGR);//转化边缘检测后的图为灰度图
	vector<Vec4i> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合
	HoughLinesP(midImage1, lines, 1, CV_PI / 180, 20, 15, 10);

	/*【4】依次在图中绘制出每条线段*/
	if (lines.size() == 0)
	{
		printf("There are no such lines.");
	}
	else
	{
		for (size_t i = 0; i < lines.size(); i++)
		{
			Vec4i l = lines[i];
			line(hough_erode, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 255, 255), 3, CV_AA);
		}
	}

}

Vec3i duibi(Vec2i& mate1, Vec2i& mate2);//用于对比,返回两者的xi之差和maxik之和【cha，sumk】
int zhuzi_x();//在这里对数据进行筛选并返回柱子中心坐标

void ThinImage(void)
{
	Mat img = ROI.clone();//需要复制一个ROI区域进行处理，直接在选取的ROI图上处理会改变原图
	Mat src1 = imread("erode1.jpg", cv::IMREAD_GRAYSCALE);
	//imshow("yuantu", src1);//显示原图
	if (src1.empty())
	{
		cout << "读取腐蚀图失败！" << endl;

	}

	threshold(src1, src1, 128, 1, cv::THRESH_BINARY);
	//图像细化
	Mat dst = thinImage(src1);
	dst = dst * 255;

	Mat midImage, dstImage;//临时变量和目标图的定义
	midImage = dst.clone();
	/*进行边缘检测和转化为灰度图*/
	Canny(dst, midImage, 50, 80, 3);//进行一此canny边缘检测
	cvtColor(midImage, dstImage, CV_GRAY2BGR);//转化边缘检测后的图为灰度图

											  //--------------以下为另外一种判断轮廓线的方法-----------------------------------
											  /***这种方法不强健，若是ROI区域内出现太多干扰的杂线段则很有可能判断出错***************************/
											  /***目前采用的是这种方法，检测托盘中心y坐标是在这步基础上的，得不到x则无法得到y**********************/

	vector<Vec4i> lines4(100);//
	int num[100] = { 0 };//用来统计线段的x坐标的
	int w = 0;
	for (size_t i = 0; i < 5; i++)
	{
		Mat roi = midImage(Rect(0, i*(dy / 5), dx, dy / 5));//将ROI区域再细分成5个区域
		HoughLinesP(roi, lines4, 1, CV_PI / 180, 10, 10, 10);
		cout << "第" << i + 1 << "个ROI区域内检测到的线段数：  " << "   " << lines4.size() << endl;
		if (lines4.size() != 0)
		{
			for (size_t m = 0; m < lines4.size(); m++)
			{
				Vec4i l = lines4[m];
				int dy, dx;
				dy = l[3] - l[1];
				dx = l[2] - l[0];
				dx = (dx >= 0) ? dx : (-dx);
				if (5 >= dx)//只有斜率满足的线段才会被统计到数组中
				{
					int n = (int)((l[0] + l[2]) / 2);//n是线段两点的平均x坐标
					num[w] = n;//将x坐标均值存入这个数组中，最多存储100个
					w++;//在5个ROI小区域检测到的符合要求的线段都会存储到同一个数组中，后面会对这些坐标分类
				}

			}
		}
	}
	int num1[500] = { 0 };//这个数组的大小还是有待商榷，太占用内存了
	int num2[500] = { 0 };
	int num3[500] = { 0 };
	int num4[500] = { 0 };
	int num5[500] = { 0 };
	static int k1 = 0;
	static int k2 = 0;
	static int k3 = 0;
	static int k4 = 0;
	static int k5 = 0;
	//以下对存储在num[]中的数据进行分类，目前最多只能分五类
	for (size_t i = 0; i < w; i++)
	{
		cout << "num[]中的x坐标为：  " << num[i] << endl;//将存储在数组中的x坐标一一输出

		int x = num[i];

		if (i == 0 || ((x - num1[0]) <= 10 && (x - num1[0]) >= -10))//
		{
			num1[k1] = x;
			k1++;
		}

		if (((x - num1[0]) >= 10 || (x - num1[0]) <= -10))
		{
			if (num2[0] != 0)//当该数组已经进行过赋值以后，x需要对该数组的第一个值进行比较再确定分类
			{
				if ((x - num2[0]) <= 10 && (x - num2[0]) >= -10)//只有差满足在【-10，,10】之间才满足
				{
					num2[k2] = x;
					k2++;
				}
			}
			else//当x不属于数组1时而且此时数组2还没有进行赋值时就将x赋给数组2
			{
				num2[k2] = x;
				k2++;
			}

		}

		if (((x - num1[0]) >= 10 || (x - num1[0]) <= -10) && ((x - num2[0]) >= 10 || (x - num2[0]) <= -10))
		{
			if (num3[0] != 0)
			{
				if ((x - num3[0]) <= 10 && (x - num3[0]) >= -10)//只有差满足在【-10，,10】之间才满足
				{
					num3[k3] = x;
					k3++;
				}
			}
			else
			{
				num3[k3] = x;
				k3++;
			}
		}

		if (((x - num1[0]) >= 10 || (x - num1[0]) <= -10) && ((x - num2[0]) >= 10 || (x - num2[0]) <= -10) && ((x - num3[0]) >= 10 || (x - num3[0]) <= -10))
		{
			if (num4[0] != 0)
			{
				if ((x - num4[0]) <= 10 && (x - num4[0]) >= -10)
				{
					num4[k4] = x;
					k4++;
				}
			}
			else
			{
				num4[k4] = x;
				k4++;
			}
		}

		if (((x - num1[0]) >= 10 || (x - num1[0]) <= -10) && ((x - num2[0]) >= 10 || (x - num2[0]) <= -10) && ((x - num3[0]) >= 10 || (x - num3[0]) <= -10) && ((x - num4[0]) >= 10 || (x - num4[0]) <= -10))
		{
			if (num5[0] != 0)
			{
				if ((x - num5[0]) <= 10 && (x - num5[0]) >= -10)
				{
					num5[k5] = x;
					k5++;
				}
			}
			else
			{
				num5[k5] = x;
				k5++;
			}
		}

		if (((x - num1[0]) >= 10 || (x - num1[0]) <= -10) && ((x - num2[0]) >= 10 || (x - num2[0]) <= -10) && ((x - num3[0]) >= 10 || (x - num3[0]) <= -10) && ((x - num4[0]) >= 10 || (x - num4[0]) <= -10) && ((x - num5[0]) >= 10 || (x - num5[0]) <= -10))
			cout << "此数无处安放~   " << x << endl;
	}
	cout << "num1   " << num1[0] << " " << num1[1] << " " << num1[2] << " " << num1[3] << " " << num1[4] << endl;
	cout << "num2   " << num2[0] << " " << num2[1] << " " << num2[2] << " " << num2[3] << " " << num2[4] << endl;
	cout << "num3   " << num3[0] << " " << num3[1] << " " << num3[2] << " " << num3[3] << " " << num3[4] << endl;
	cout << "num4   " << num4[0] << " " << num4[1] << " " << num5[2] << " " << num4[3] << " " << num4[4] << endl;
	k1 = 0, k2 = 0, k3 = 0, k4 = 0;
	//下面五个循环是为了分别求出五个数组中线段的个数
	for (size_t i = 0; i < 50; i++)
	{
		if (num1[i] == 0)
			break;
		k1++;
	}
	cout << "k1  " << k1 << endl;
	for (size_t i = 0; i < 50; i++)
	{
		if (num2[i] == 0)
			break;
		k2++;
	}
	for (size_t i = 0; i < 50; i++)
	{
		if (num3[i] == 0)
			break;
		k3++;
	}
	for (size_t i = 0; i < 50; i++)
	{
		if (num4[i] == 0)
			break;
		k4++;
	}
	for (size_t i = 0; i < 50; i++)
	{
		if (num5[i] == 0)
			break;
		k5++;
	}
	int numk[5] = { k1,k2,k3,k4,k5 };//将五个线段集合里面的线段数目按顺序存入
	int max = 0, max1k = -5, max2k = -5, max3k = -5, max4k = -5;
	//cout << "numk[]   " << numk[0] << " " << numk[1] << " " << numk[2] << endl;
	////下面si个for循环是为了找出在五个数组中线段数目排前si的4个数组的序号-1
	for (int j = 0; j < 5; j++)
	{
		if (numk[j] > max)
		{
			max = numk[j];
			max1k = j;//将线段数目最多的序号赋值给max1k
					  //cout <<"max1k  " << max1k <<"max "<< max << endl;
		}
	}
	numk[max1k] = 0; max = 0;
	for (int j = 0; j < 5; j++)//这里筛选出线段数目第二多的
	{
		if (numk[j] > max)
		{
			max = numk[j];
			max2k = j;
		}
	}
	numk[max2k] = 0; max = 0;
	for (int j = 0; j < 5; j++)
	{
		if (numk[j] > max)
		{
			max = numk[j];
			max3k = j;
		}
	}
	numk[max3k] = 0; max = 0;
	for (int j = 0; j < 5; j++)
	{
		if (numk[j] > max)
		{
			max = numk[j];
			max4k = j;
		}
	}
	numk[max4k] = 0; max = 0;
	//cout <<"maxnk" << " " <<max1k <<" " << max2k << " " << max3k<< endl;
	int x1 = 0, x2 = 0, x3 = 0, x4 = 0;
	////这里只针对线段数目排在前4的进行比对
	switch (max1k)//这里maxl1k表示的是含线段数目最多的数组序号数-1
	{
	case 0:x1 = num1[0]; break;		case 1:x1 = num2[0]; break;		case 2:x1 = num3[0]; break;		case 3:x1 = num4[0]; break;		case 4:x1 = num5[0]; break; default:;
	}
	switch (max2k)
	{
	case 0:x2 = num1[0]; break;		case 1:x2 = num2[0]; break;		case 2:x2 = num3[0]; break;		case 3:x2 = num4[0]; break;		case 4:x2 = num5[0]; break; default:;
	}
	switch (max3k)//这里存在bug，如果只有两个集合有线段，3,4,5数组里面都是0，而且一开始给max3k赋值为0
	{            //的话，这里就会跳入第一个case，所以赋值不能为0
	case 0:x3 = num1[0]; break;		case 1:x3 = num2[0]; break;		case 2:x3 = num3[0]; break;		case 3:x3 = num4[0]; break;		case 4:x3 = num5[0]; break; default:;
	}
	switch (max4k)
	{
	case 0:x4 = num1[0]; break;		case 1:x4 = num2[0]; break;		case 2:x4 = num3[0]; break;		case 3:x4 = num4[0]; break;		case 4:x4 = num5[0]; break; default:;
	}

	int cha = ((x1 - x2) >= 0) ? (x1 - x2) : (x2 - x1);
	cout << "两者只差：  " << cha << endl;
	//下面是对筛选出来的几个不同的线段间距进行合理性分析
	//vector<Vec2i>mate(10);//创建一个容器存储在上面匹配到的【maxik，xi】---i表示数字
	mate[0][0] = max1k; mate[0][1] = x1;
	mate[1][0] = max2k; mate[1][1] = x2;
	mate[2][0] = max3k; mate[2][1] = x3;
	mate[3][0] = max4k; mate[3][1] = x4;
	cout << "mate[0]的两个值" << mate[0][0] << " " << mate[0][1] << endl;
	cout << "mate[1]的两个值" << mate[1][0] << " " << mate[1][1] << endl;
	cout << "mate[2]的两个值" << mate[2][0] << " " << mate[2][1] << endl;
	cout << "mate[3]的两个值" << mate[3][0] << " " << mate[3][1] << endl;

	central_x = (zhuzi_x() + x);//在这里对数据进行筛选并返回柱子中心坐标

								//---------------------以上为另外一种方法--------------------------------/

								/*进行霍夫线变换*/
	vector<Vec4i> lines2(100);//定义一个矢量结构lines用于存放得到的线段矢量集合
							  //vector<Vec4i> lines3;//用于存放经斜率排除后的线段
	HoughLinesP(midImage, lines2, 1, CV_PI / 180, 20, 15, 10);
	cout << "细化图霍夫变换中的线段数目：" << lines2.size() << endl;
	//int m = 0;//用于统计满足斜率的线段数目
	if (lines2.size() != 0)
	{
		for (size_t i = 0; i < lines2.size(); i++)
		{
			Vec4i l = lines2[i];
			int dy, dx;
			dy = l[3] - l[1];
			dx = l[2] - l[0];
			dx = (dx >= 0) ? dx : (-dx);
			dy = (dy >= 0) ? dy : (-dy);
			//float k = dy / dx;
			if (6 >= dx)//线段两个端点的dx过大判断为噪声，不大于5时才会画出来
			{          //这步判断可以放在第一次霍夫变换中						
					   //lines3.push_back(m);
				line(img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, CV_AA);
				//m++;
			}
			else;
		}
		//在ROI图中画出中心线
		line(img, Point(central_x - x, 0), Point(central_x - x, 115), Scalar(0, 255, 0), 1, CV_AA);

	}
	else
		cout << "细化图中没有提取到直线" << endl;

	namedWindow("ROI细化图", 1);
	imshow("ROI细化图", img);

}

//------function：用于对比,返回两者的xi之差、maxik之和、xi之和的一半【cha，sumk，central】-----
Vec3i duibi(Vec2i& mate1, Vec2i& mate2)
{
	Vec3i mate3;
	mate3[0] = 0;
	mate3[1] = 0;
	mate3[2] = 0;
	if (mate1[1] == 0 || mate2[1] == 0)//只要有一个匹配点的xi为0就不进行这些比较了，因为没有必要了
	{
		return mate3;
	}
	int cha = ((mate1[1] - mate2[1])>0) ? (mate1[1] - mate2[1]) : (mate2[1] - mate1[1]);//取两者差的绝对值
	int sumk = mate1[0] + mate2[0];
	int central = (int)((mate1[1] + mate2[1]) / 2);
	if ((143 <= cha&&cha <= 163) || (235 <= cha&&cha <= 255))//三米远柱子轮廓线段间距245，六米远的为153
	{
		mate3[0] = cha;
		mate3[1] = sumk;
		mate3[2] = central;

	}
	return mate3;
}
//---------------function：对数据进行筛选并返回柱子中心坐标-----------------
int zhuzi_x()
{
	vector<Vec3i>mate1(10);//创建一个容器存储这些对比结果
	vector<Vec3i>mate2;//创建一个容器存储满足要求的对比结果
	int max = 0, j = 0;
	mate1[0] = duibi(mate[0], mate[1]);
	mate1[1] = duibi(mate[0], mate[2]);
	mate1[2] = duibi(mate[0], mate[3]);
	mate1[3] = duibi(mate[1], mate[2]);
	mate1[4] = duibi(mate[1], mate[3]);
	mate1[5] = duibi(mate[2], mate[3]);
	for (int i = 0; i < 6; i++)
	{
		if (mate1[i][0] > 0)
		{
			cout << "差值、sumk/central：  " << mate1[i][0] << "  " << mate1[i][1] << "  " << mate1[i][2] << endl;
			mate2.push_back(mate1[i]);
			//num_pipei++;
		}
	}
	cout << "mate2.size() :  " << mate2.size() << endl;
	if (mate2.size() == 1)//只有一个匹配结果合理，直接返回
	{
		return (mate2[0][2]);
	}
	else  if (mate2.size() > 1)//多个匹配结果满足要求，进行筛选
	{
		max = mate2[0][1];
		for (int i = 0; i < mate2.size(); i++)
		{
			if (mate2[i][1]>max)
			{
				max = mate2[i][1];
				j = i;
			}
		}
		return (mate2[j][2]);
	}
	else//没有满足要求的匹配点，返回0
	{
		cout << "没有满足要求的匹配结果，请考虑是否有bug或图像是否满足检测要求！" << "  " << endl;
		return 0;
	}
}

//----------------function：检测托盘中心y坐标-----------------------------------------------------
void seek_y(int &central_x, Mat& changban)
{
	if (central_x > x)//只有在xihua（）中求到了柱子中心x坐标才进行求y坐标的操作
	{
		Mat g_grayImage1;
		cout << "central_x  " << central_x << endl;
		//这里存在风险，如果柱子过于靠边，还是往左边去roi区域来检测长板的话就会溢界保错
		if (central_x < x_limit)
		{
			draw_ROI(src, changban, (central_x + x1), y_cb, dx_cb, dy_cb);

		}
		else
			draw_ROI(src, changban, (central_x - x_limit), y_cb, dx_cb, dy_cb);
		imwrite("changban.jpg", changban);
		Mat changban_cy = changban.clone();

		//imshow("changban_cy", changban_cy);
		cvtColor(changban_cy, g_grayImage1, CV_RGB2GRAY);
		//imshow("cvtColor", g_grayImage1);

		threshold(g_grayImage1, g_grayImage1, 80, 255, 1);
		//imshow("threshold", g_grayImage1);
		/***************************以下：将阈值化图先腐蚀在霍夫线变换***********************/
		Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
		Mat erode1;
		erode(g_grayImage1, erode1, element);
		//imshow("erode1", erode1);
		Canny(erode1, erode1, 50, 80, 3);//进行一此canny边缘检测
		vector<Vec4i> lines5(100);//定义一个矢量结构lines用于存放得到的线段矢量集合
		vector<size_t> aggregation_y1(100);
		HoughLinesP(erode1, lines5, 1, CV_PI / 180, 10, 50, 10);
		if (lines5.size() != 0)
		{
			int j1 = 0;
			for (size_t i = 0; i < lines5.size(); i++)
			{
				Vec4i l = lines5[i];
				int dy1, dx1;
				dy1 = l[3] - l[1];
				dx1 = l[2] - l[0];
				dx1 = (dx1 >= 0) ? dx1 : (-dx1);
				dy1 = (dy1 >= 0) ? dy1 : (-dy1);
				if (dy1 <= 10)//保证平行度，如果相机摆歪了这里就gg了
				{
					aggregation_y1[j1] = (int)(l[3] + l[1]) / 2;
					j1++;
					//将提取到的直线画出来
					line(changban_cy, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 2, CV_AA);
				}
			}
		}
		int min1 = aggregation_y1[0];
		//寻找最小的y坐标，也即是长板的上边缘，一旦上面的步骤出问题，在地板上检测到了直线，这里就出错
		for (size_t i = 0; i < aggregation_y1.size(); i++)
		{
			if (aggregation_y1[i] < min1&&aggregation_y1[i] != 0)
			{
				min1 = aggregation_y1[i];
			}
			//cout << "aggregation_y1[i] " << aggregation_y1[i] << endl;

		}
		//line(changban_cy, Point(0, min), Point(500, min), Scalar(0, 255, 0), 4, CV_AA);

		cout << "aggregation_y1 min1== " << min1 << endl;

		//cout <<"lines4.size()== " << lines4.size() << endl;
		imshow("changban_cy", changban_cy);
		//central_y=(1050+ min1-450);//这是针对三米远1米高的柱子的
		//central_y = (1050 + min1 - 245);//245是针对6米远0.5米高的柱子的,6m远1.5米高减去860
		if (min1 <= 20)
			central_y = 0;
		/********************************以上：将阈值化图先腐蚀在霍夫线变换*/

		/*******************以下是另外一种方法：将提取的图先滤波、sobel、再canny、再霍夫变换************************************/
		Mat changban_cy1 = changban.clone();
		int g_nMedianBlurValue = 20;  //中值滤波参数值
		vector<size_t> aggregation_y2(100);

		Mat grad_y, g_dstImage4, img_canny1;
		medianBlur(changban_cy1, g_dstImage4, g_nMedianBlurValue * 2 + 1);
		//imshow("g_dstImage4", g_dstImage4);
		sobel(g_dstImage4, abs_grad_x1, abs_grad_y1, dst1);
		//imshow("abs_grad_y1", abs_grad_y1);
		canny(abs_grad_y1, img_canny1, 80, 120);
		//imshow("img_canny1", img_canny1);

		Mat midImage, dstImage;//临时变量和目标图的定义
							   //  imshow("canny_again", midImage);

		cvtColor(midImage, dstImage, CV_GRAY2BGR);//转化边缘检测后的图为灰度图
		vector<Vec4i> lines6(100);//定义一个矢量结构lines用于存放得到的线段矢量集合
		HoughLinesP(img_canny1, lines6, 1, CV_PI / 180, 20, 15, 10);

		if (lines6.size() != 0)
		{
			int j2 = 0;
			for (size_t i = 0; i < lines6.size(); i++)
			{
				Vec4i l = lines6[i];
				int dy2, dx2;
				dy2 = l[3] - l[1];
				dx2 = l[2] - l[0];
				dx2 = (dx2 >= 0) ? dx2 : (-dx2);
				dy2 = (dy2 >= 0) ? dy2 : (-dy2);
				// if(dy2<=10)
				line(changban_cy1, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
				aggregation_y2[j2] = (int)(l[3] + l[1]) / 2;
				j2++;
			}
			int min2 = aggregation_y2[0];
			//寻找最小的y坐标，也即是长板的上边缘，一旦上面的步骤出问题，在地板上检测到了直线，这里就出错
			for (size_t i = 0; i < aggregation_y2.size(); i++)
			{
				if (aggregation_y2[i] < min2&&aggregation_y2[i] != 0)
				{
					min2 = aggregation_y2[i];
				}
				//cout << "aggregation_y2[i] " << aggregation_y2[i] << endl;
			}
			cout << "aggregation_y2 min2   " << min2 << endl;
			line(changban_cy1, Point(0, min2), Point(500, min2), Scalar(255, 0, 0), 3, CV_AA);
			//central_y=(1050+ min2-450);//这是针对三米远1米高的柱子的
			central_y = (1050 + min2 - 520);//245是针对6米远0.5米高的柱子的,6m远1.5米高减去860
			imshow("changban_cy1", changban_cy1);
			if (min1 <= 20)
				central_y = 0;
		}
		/*******************以shang是另外一种方法：将提取的图先sobel、再canny、再霍夫变换************************************/
	}
	else
	{
		cout << "上步骤中没有求到x坐标，故无法求取y坐标！" << "" << endl;
	}
}








