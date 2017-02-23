/*
function�� �������������������
date��2016.11.9
author:LHMD.NUC
*/

#include"opencv2/opencv.hpp"
using namespace cv;
using namespace std;

/*---------------------ȫ�ֱ�������------------------------------*/
Mat src;
Mat ROI, changban;
int low_thresd = 34;//canny��ֵ
int high_thresd = 74;//100Ҳ��ò��

					 /*!!!!!!������Ҫ��������ͷ�����ĵĲ���!!!!!!*/
					 //�����������ROI�����ڵĲ��������������ҷֱ���ROI������߶�������x��y�����dx���߶�dy
					 //���������Ҫ�������ӵ����ұ߽�
uint16_t x = 536, y = 1003, dx = 536, dy = 115;

//seek_y()�����ò���
uint16_t x_limit = 640;//��ȡ����ROI����ʱx������ж�ֵ,
uint16_t x1 = 140;
uint16_t y_cb = 1050, dx_cb = 500, dy_cb = 186;//��ⳤ���ȡROI����Ĳ���������ROI���󶥵��y���꣬ROI���Ⱥ�ROI�ĸ߶�

											   /*!!!!!!������Ҫ��������ͷ�����ĵĲ���!!!!!!*/

Mat abs_grad_x, abs_grad_y, dst;//���� abs_grad_x �� abs_grad_y ����
Mat abs_grad_x1, abs_grad_y1, dst1;//���� abs_grad_x �� abs_grad_y ����

Mat img_canny;
Mat hough_erode;
Mat midImage;
int central_x = 0;
int central_y = 0;

vector<Vec2i>mate(10);//����һ�������洢������ƥ�䵽�ġ�maxik��xi��---i��ʾ����

					  /*---------------------��������------------------------------*/
void draw_ROI(const Mat& src, Mat& ROI, int x1, int y1, int dx, int dy);
void strengthen(const Mat& ROI, Mat& result);
void sobel(const Mat& result, Mat& abs_grad_x, Mat& abs_grad_y, Mat& dst);
static void on_canny(int, void*);
void canny(Mat& img, Mat& img_canny, int low_thresd, int high_thresd);
void ThinImage(void);
void seek_y(int &central_x, Mat& changban);

/*--------------��ͼ���е��߶ν���ϸ�������߶ε���С���----------------------------*/
cv::Mat thinImage(const Mat & src, const int maxIterations = -1)
{
	assert(src.type() == CV_8UC1);
	cv::Mat dst;
	int width = src.cols;
	int height = src.rows;
	src.copyTo(dst);
	int count = 0;  //��¼��������
	while (true)
	{
		count++;
		if (maxIterations != -1 && count > maxIterations) //���ƴ������ҵ�����������
			break;
		std::vector<uchar *> mFlag; //���ڱ����Ҫɾ���ĵ�
									//�Ե���
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//��������ĸ����������б��
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
						//���
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//����ǵĵ�ɾ��
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//ֱ��û�е����㣬�㷨����
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//��mFlag���
		}

		//�Ե���
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//��������ĸ����������б��
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
						//���
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//����ǵĵ�ɾ��
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//ֱ��û�е����㣬�㷨����
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//��mFlag���
		}
	}
	return dst;
}


int main(int argc, char* argv[])
{
	int Open = 0;//open==1��������Ƶ��open==0������ͼƬ
	if (Open == 1)
	{
		int m = 0;
		VideoCapture capture("5.avi");
		if (!capture.isOpened())
		{
			cout << "\nfail to open video!\n" << endl;
			return 0;
		}
		//��ȡ����֡��
		long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
		cout << "������Ƶ��" << totalFrameNumber << "֡" << endl;
		double rate = capture.get(CV_CAP_PROP_FPS);
		cout << "֡��Ϊ:" << rate << endl;
		while (1)
		{
			if (m == totalFrameNumber)//��Ƶ��������������ѭ��
			{
				cout << "��Ƶ�Ѿ���������Ŷ����~" << endl;
				break;
			}
			capture >> src;
			m++;
			if ((src.cols < (x + dx)) || (src.rows < (y + dy)))//�����ͼƬ��С��Ҫ�󣬸��������ͼƬʱ����Ҫ��
			{
				printf("ͼƬ̫С��������ѡȡROI�����Ҫ��������ͼƬ��");
				return false;
			}
			draw_ROI(src, ROI, x, y, dx, dy);//��ȡROI����

											 //��ǿͼƬ�ĶԱȶȣ�������ʱ�������ⲽ��
											 /*Mat result;strengthen(ROI, result);*/

											 //------------��sobel���Ӻ�canny���Ӷ�ͼƬ���б�Ե���õ���������---------------
			sobel(ROI, abs_grad_x, abs_grad_y, dst);
			canny(abs_grad_x, img_canny, low_thresd, high_thresd);
			//imshow("��Ч��ͼ��Canny��Ե���", img_canny);

			//------------����任�õ����ӵ������߶�-----------------------------------------
			Mat midImage, dstImage;//��ʱ������Ŀ��ͼ�Ķ���
			Canny(img_canny, midImage, 50, 80, 3);//����һ��canny��Ե���
			cvtColor(midImage, dstImage, CV_GRAY2BGR);//ת����Ե�����ͼΪ�Ҷ�ͼ
			vector<Vec4i> lines(40);//����һ��ʸ���ṹlines���ڴ�ŵõ����߶�ʸ������
			HoughLinesP(midImage, lines, 1, CV_PI / 180, 20, 15, 10);
			printf("\n%d\n", lines.size());
			if (lines.size() == 0)//�ڵ�һ�λ���任����û�м�⵽�߶��򲻽�������ļ��
			{
				printf("\n��һ�λ���任û��ֱ�ߣ����ٽ�����һ������!\n");
			}
			else
			{
				for (size_t i = 0; i < lines.size(); i++)//����⵽���߶λ��Ƶ�ͼ��
				{
					Vec4i l = lines[i];
					line(dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 255, 255), 3, CV_AA);
				}
				Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
				Mat erode1;
				erode(dstImage, erode1, element);
				//namedWindow("erode1", 1);
				//imshow("erode1", erode1);
				imwrite("erode1.jpg", erode1);//����ʴ���ͼƬ���浽�ļ�����
				ThinImage();
				waitKey(1);//����Ӹ�С��ʱ�Ϳ�����ʾ�����ˣ�����waitkeyͼƬ�޷�������ʾ

				seek_y(central_x, changban);
				waitKey(1);
				if (central_x != x)
					line(src, Point(central_x, 0), Point(central_x, src.cols), Scalar(0, 0, 255), 3, CV_AA);
				if (central_y != 0)
					line(src, Point(0, central_y), Point(src.rows, central_y), Scalar(0, 255, 0), 3, CV_AA);
				if (central_x != x&&central_y != 0)
					circle(src, Point(central_x, central_y), 15, Scalar(255, 0, 0), -1, CV_AA);//��һ��ʵ��Բ
				namedWindow("SRC", 0);//����������ͼƬ̫��һ����ʾԭͼ������Ƶ������ͼƬ�޷���ʾ
				imshow("SRC", src);
				central_y = 0;
			}

			waitKey(1);
		}
	}
	else//open=0������ͼƬ
	{
		src = imread("24.jpg");
		if (!src.data)
		{
			printf("��ȡͼƬ����be sure that the picture is in the document.");
			return 0;
		}
		if ((src.cols < (x + dx)) || (src.rows < (y + dy)))
		{
			printf("ͼƬ̫С��������ѡȡROI�����Ҫ��������ͼƬ��");
			return 0;
		}

		draw_ROI(src, ROI, x, y, dx, dy);		//��ȡROI����
												//imshow("ROI", ROI);

												//��ǿͼƬ�ĶԱȶȣ�������ʱ�������ⲽ��
												/*Mat result;
												strengthen(ROI, result);*/

		sobel(ROI, abs_grad_x, abs_grad_y, dst);//��sobel���Ӷ�ͼƬ���б�Ե���
												//imshow("abs_grad_x", abs_grad_x);

		canny(abs_grad_x, img_canny, low_thresd, high_thresd);
		//imshow("��Ч��ͼ��Canny��Ե���", img_canny);
		///////////////////////////////////////////////////////////////////////////

		////////////////////////////////////////////////////////////////////////
		Mat midImage, dstImage;//��ʱ������Ŀ��ͼ�Ķ���
		Canny(img_canny, midImage, 50, 80, 3);//����һ��canny��Ե���
		cvtColor(midImage, dstImage, CV_GRAY2BGR);//ת����Ե�����ͼΪ�Ҷ�ͼ
												  //imshow("midImage", midImage);
		vector<Vec4i> lines(100);//ע���ڴ�����������ͬͼƬ��ʱ��Ҫ���Ƕ���һ��ʸ���ṹlines���ڴ�ŵõ����߶�ʸ������
		HoughLinesP(midImage, lines, 1, CV_PI / 180, 20, 15, 10);
		//cout << "sobel��canny֮���⵽���߶���Ŀ��" << lines.size() << endl;
		if (lines.size() == 0)
		{
			printf("��һ�λ���任û��ֱ�ߣ����ٽ�����һ������!");
		}
		else

		{
			for (size_t i = 0; i < lines.size(); i++)
			{
				Vec4i l = lines[i];
				line(dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 255, 255), 3, CV_AA);
			}
			////ֻ����ǰ���⵽��ֱ�ߵ�����²Ž��и�ʴ�ͺ�ϸ��
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
				circle(src, Point(central_x, central_y), 15, Scalar(255, 0, 0), -1, CV_AA);//��һ��ʵ��Բ

			namedWindow("SRC", 0);
			imshow("SRC", src);
		}

	}
	waitKey(0);
	return 0;

}

/*--------------------function:��ԭͼ��ѡȡ����ROI����-----------------------------------------*/
void draw_ROI(const Mat& src, Mat& ROI, int x, int y, int dx, int dy)
{
	ROI = src(Rect(x, y, dx, dy));
	//cout << "��ȡROI����ɹ���" << endl;
}

/*---------------------function:��ROIͼƬ����ϸ����ǿ-------------------------------------------*/
void strengthen(const Mat& ROI, Mat& result)
{

}

/*---------------------function:sobel���Ӷ�ͼƬ�����������--------------------------------------*/
void sobel(const Mat& result, Mat& abs_grad_x, Mat& abs_grad_y, Mat& dst)
{
	Mat grad_x, grad_y;
	Sobel(result, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);//�� X�����ݶ�
	convertScaleAbs(grad_x, abs_grad_x);//ת��Ϊ����ֵ���

	Sobel(result, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);//��Y�����ݶ�
	convertScaleAbs(grad_y, abs_grad_y);

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);//�ϲ��ݶ�(����)
	cout << "sobel���Ӽ��ɹ���" << endl;


}
//void on_canny(int, void*,int )
//{
//	canny(abs_grad_x, img_canny, low_thresd, high_thresd);
//	imshow("��Ч��ͼ��Canny��Ե���", img_canny);
//////////////////////////////////////////////////////////////////////////
//	Mat midImage, dstImage;//��ʱ������Ŀ��ͼ�Ķ���
//    Canny(img_canny, midImage, 50, 80, 3);//����һ��canny��Ե���
//	cvtColor(midImage, dstImage, CV_GRAY2BGR);//ת����Ե�����ͼΪ�Ҷ�ͼ
//	imshow("zaici canny", dstImage);
//	vector<Vec4i> lines;//����һ��ʸ���ṹlines���ڴ�ŵõ����߶�ʸ������
//	HoughLinesP(midImage, lines, 1, CV_PI / 180, 20, 15, 10);
/////////////////////////////////////////////////////////////////////////
//	//midImage = midImage1.clone();
//	//hough_erode = img_canny.clone();          //midImage.copyTo(dstImage);
//
//	//Canny(img_canny, midImage, 50, 80, 3);//����һ��canny��Ե���
//	////cvtColor(midImage, midImage, CV_GRAY2BGR);//ת����Ե�����ͼΪ�Ҷ�ͼ
//	//vector<Vec4i> lines;//����һ��ʸ���ṹlines���ڴ�ŵõ����߶�ʸ������
//	//HoughLinesP(midImage, lines, 1, CV_PI / 180, 20, 15, 10);
//	///*��4��������ͼ�л��Ƴ�ÿ���߶�*/
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

/*---------------------function:canny��Ե���-----------------------------------------------------*/
void canny(Mat& img, Mat& img_canny, int low_thresd, int high_thresd)
{
	cvtColor(img, img_canny, CV_BGR2GRAY);
	blur(img_canny, img_canny, Size(3, 3));
	Canny(img_canny, img_canny, low_thresd, high_thresd, 3);
	//imshow("��Ч��ͼ��Canny��Ե���", img_canny);
}

void houghline_erode(Mat& midImage1, Mat& hough_erode)
{
	//midImage= midImage1.clone();
	//hough_erode = midImage1.clone();          //midImage.copyTo(dstImage);
	Canny(midImage1, midImage1, 50, 80, 3);//����һ��canny��Ե���
	cvtColor(midImage1, hough_erode, CV_GRAY2BGR);//ת����Ե�����ͼΪ�Ҷ�ͼ
	vector<Vec4i> lines;//����һ��ʸ���ṹlines���ڴ�ŵõ����߶�ʸ������
	HoughLinesP(midImage1, lines, 1, CV_PI / 180, 20, 15, 10);

	/*��4��������ͼ�л��Ƴ�ÿ���߶�*/
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

Vec3i duibi(Vec2i& mate1, Vec2i& mate2);//���ڶԱ�,�������ߵ�xi֮���maxik֮�͡�cha��sumk��
int zhuzi_x();//����������ݽ���ɸѡ������������������

void ThinImage(void)
{
	Mat img = ROI.clone();//��Ҫ����һ��ROI������д���ֱ����ѡȡ��ROIͼ�ϴ����ı�ԭͼ
	Mat src1 = imread("erode1.jpg", cv::IMREAD_GRAYSCALE);
	//imshow("yuantu", src1);//��ʾԭͼ
	if (src1.empty())
	{
		cout << "��ȡ��ʴͼʧ�ܣ�" << endl;

	}

	threshold(src1, src1, 128, 1, cv::THRESH_BINARY);
	//ͼ��ϸ��
	Mat dst = thinImage(src1);
	dst = dst * 255;

	Mat midImage, dstImage;//��ʱ������Ŀ��ͼ�Ķ���
	midImage = dst.clone();
	/*���б�Ե����ת��Ϊ�Ҷ�ͼ*/
	Canny(dst, midImage, 50, 80, 3);//����һ��canny��Ե���
	cvtColor(midImage, dstImage, CV_GRAY2BGR);//ת����Ե�����ͼΪ�Ҷ�ͼ

											  //--------------����Ϊ����һ���ж������ߵķ���-----------------------------------
											  /***���ַ�����ǿ��������ROI�����ڳ���̫����ŵ����߶�����п����жϳ���***************************/
											  /***Ŀǰ���õ������ַ����������������y���������ⲽ�����ϵģ��ò���x���޷��õ�y**********************/

	vector<Vec4i> lines4(100);//
	int num[100] = { 0 };//����ͳ���߶ε�x�����
	int w = 0;
	for (size_t i = 0; i < 5; i++)
	{
		Mat roi = midImage(Rect(0, i*(dy / 5), dx, dy / 5));//��ROI������ϸ�ֳ�5������
		HoughLinesP(roi, lines4, 1, CV_PI / 180, 10, 10, 10);
		cout << "��" << i + 1 << "��ROI�����ڼ�⵽���߶�����  " << "   " << lines4.size() << endl;
		if (lines4.size() != 0)
		{
			for (size_t m = 0; m < lines4.size(); m++)
			{
				Vec4i l = lines4[m];
				int dy, dx;
				dy = l[3] - l[1];
				dx = l[2] - l[0];
				dx = (dx >= 0) ? dx : (-dx);
				if (5 >= dx)//ֻ��б��������߶βŻᱻͳ�Ƶ�������
				{
					int n = (int)((l[0] + l[2]) / 2);//n���߶������ƽ��x����
					num[w] = n;//��x�����ֵ������������У����洢100��
					w++;//��5��ROIС�����⵽�ķ���Ҫ����߶ζ���洢��ͬһ�������У���������Щ�������
				}

			}
		}
	}
	int num1[500] = { 0 };//�������Ĵ�С�����д���ȶ��̫ռ���ڴ���
	int num2[500] = { 0 };
	int num3[500] = { 0 };
	int num4[500] = { 0 };
	int num5[500] = { 0 };
	static int k1 = 0;
	static int k2 = 0;
	static int k3 = 0;
	static int k4 = 0;
	static int k5 = 0;
	//���¶Դ洢��num[]�е����ݽ��з��࣬Ŀǰ���ֻ�ܷ�����
	for (size_t i = 0; i < w; i++)
	{
		cout << "num[]�е�x����Ϊ��  " << num[i] << endl;//���洢�������е�x����һһ���

		int x = num[i];

		if (i == 0 || ((x - num1[0]) <= 10 && (x - num1[0]) >= -10))//
		{
			num1[k1] = x;
			k1++;
		}

		if (((x - num1[0]) >= 10 || (x - num1[0]) <= -10))
		{
			if (num2[0] != 0)//���������Ѿ����й���ֵ�Ժ�x��Ҫ�Ը�����ĵ�һ��ֵ���бȽ���ȷ������
			{
				if ((x - num2[0]) <= 10 && (x - num2[0]) >= -10)//ֻ�в������ڡ�-10��,10��֮�������
				{
					num2[k2] = x;
					k2++;
				}
			}
			else//��x����������1ʱ���Ҵ�ʱ����2��û�н��и�ֵʱ�ͽ�x��������2
			{
				num2[k2] = x;
				k2++;
			}

		}

		if (((x - num1[0]) >= 10 || (x - num1[0]) <= -10) && ((x - num2[0]) >= 10 || (x - num2[0]) <= -10))
		{
			if (num3[0] != 0)
			{
				if ((x - num3[0]) <= 10 && (x - num3[0]) >= -10)//ֻ�в������ڡ�-10��,10��֮�������
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
			cout << "�����޴�����~   " << x << endl;
	}
	cout << "num1   " << num1[0] << " " << num1[1] << " " << num1[2] << " " << num1[3] << " " << num1[4] << endl;
	cout << "num2   " << num2[0] << " " << num2[1] << " " << num2[2] << " " << num2[3] << " " << num2[4] << endl;
	cout << "num3   " << num3[0] << " " << num3[1] << " " << num3[2] << " " << num3[3] << " " << num3[4] << endl;
	cout << "num4   " << num4[0] << " " << num4[1] << " " << num5[2] << " " << num4[3] << " " << num4[4] << endl;
	k1 = 0, k2 = 0, k3 = 0, k4 = 0;
	//�������ѭ����Ϊ�˷ֱ��������������߶εĸ���
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
	int numk[5] = { k1,k2,k3,k4,k5 };//������߶μ���������߶���Ŀ��˳�����
	int max = 0, max1k = -5, max2k = -5, max3k = -5, max4k = -5;
	//cout << "numk[]   " << numk[0] << " " << numk[1] << " " << numk[2] << endl;
	////����si��forѭ����Ϊ���ҳ�������������߶���Ŀ��ǰsi��4����������-1
	for (int j = 0; j < 5; j++)
	{
		if (numk[j] > max)
		{
			max = numk[j];
			max1k = j;//���߶���Ŀ������Ÿ�ֵ��max1k
					  //cout <<"max1k  " << max1k <<"max "<< max << endl;
		}
	}
	numk[max1k] = 0; max = 0;
	for (int j = 0; j < 5; j++)//����ɸѡ���߶���Ŀ�ڶ����
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
	////����ֻ����߶���Ŀ����ǰ4�Ľ��бȶ�
	switch (max1k)//����maxl1k��ʾ���Ǻ��߶���Ŀ�������������-1
	{
	case 0:x1 = num1[0]; break;		case 1:x1 = num2[0]; break;		case 2:x1 = num3[0]; break;		case 3:x1 = num4[0]; break;		case 4:x1 = num5[0]; break; default:;
	}
	switch (max2k)
	{
	case 0:x2 = num1[0]; break;		case 1:x2 = num2[0]; break;		case 2:x2 = num3[0]; break;		case 3:x2 = num4[0]; break;		case 4:x2 = num5[0]; break; default:;
	}
	switch (max3k)//�������bug�����ֻ�������������߶Σ�3,4,5�������涼��0������һ��ʼ��max3k��ֵΪ0
	{            //�Ļ�������ͻ������һ��case�����Ը�ֵ����Ϊ0
	case 0:x3 = num1[0]; break;		case 1:x3 = num2[0]; break;		case 2:x3 = num3[0]; break;		case 3:x3 = num4[0]; break;		case 4:x3 = num5[0]; break; default:;
	}
	switch (max4k)
	{
	case 0:x4 = num1[0]; break;		case 1:x4 = num2[0]; break;		case 2:x4 = num3[0]; break;		case 3:x4 = num4[0]; break;		case 4:x4 = num5[0]; break; default:;
	}

	int cha = ((x1 - x2) >= 0) ? (x1 - x2) : (x2 - x1);
	cout << "����ֻ�  " << cha << endl;
	//�����Ƕ�ɸѡ�����ļ�����ͬ���߶μ����к����Է���
	//vector<Vec2i>mate(10);//����һ�������洢������ƥ�䵽�ġ�maxik��xi��---i��ʾ����
	mate[0][0] = max1k; mate[0][1] = x1;
	mate[1][0] = max2k; mate[1][1] = x2;
	mate[2][0] = max3k; mate[2][1] = x3;
	mate[3][0] = max4k; mate[3][1] = x4;
	cout << "mate[0]������ֵ" << mate[0][0] << " " << mate[0][1] << endl;
	cout << "mate[1]������ֵ" << mate[1][0] << " " << mate[1][1] << endl;
	cout << "mate[2]������ֵ" << mate[2][0] << " " << mate[2][1] << endl;
	cout << "mate[3]������ֵ" << mate[3][0] << " " << mate[3][1] << endl;

	central_x = (zhuzi_x() + x);//����������ݽ���ɸѡ������������������

								//---------------------����Ϊ����һ�ַ���--------------------------------/

								/*���л����߱任*/
	vector<Vec4i> lines2(100);//����һ��ʸ���ṹlines���ڴ�ŵõ����߶�ʸ������
							  //vector<Vec4i> lines3;//���ڴ�ž�б���ų�����߶�
	HoughLinesP(midImage, lines2, 1, CV_PI / 180, 20, 15, 10);
	cout << "ϸ��ͼ����任�е��߶���Ŀ��" << lines2.size() << endl;
	//int m = 0;//����ͳ������б�ʵ��߶���Ŀ
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
			if (6 >= dx)//�߶������˵��dx�����ж�Ϊ������������5ʱ�Żử����
			{          //�ⲽ�жϿ��Է��ڵ�һ�λ���任��						
					   //lines3.push_back(m);
				line(img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, CV_AA);
				//m++;
			}
			else;
		}
		//��ROIͼ�л���������
		line(img, Point(central_x - x, 0), Point(central_x - x, 115), Scalar(0, 255, 0), 1, CV_AA);

	}
	else
		cout << "ϸ��ͼ��û����ȡ��ֱ��" << endl;

	namedWindow("ROIϸ��ͼ", 1);
	imshow("ROIϸ��ͼ", img);

}

//------function�����ڶԱ�,�������ߵ�xi֮�maxik֮�͡�xi֮�͵�һ�롾cha��sumk��central��-----
Vec3i duibi(Vec2i& mate1, Vec2i& mate2)
{
	Vec3i mate3;
	mate3[0] = 0;
	mate3[1] = 0;
	mate3[2] = 0;
	if (mate1[1] == 0 || mate2[1] == 0)//ֻҪ��һ��ƥ����xiΪ0�Ͳ�������Щ�Ƚ��ˣ���Ϊû�б�Ҫ��
	{
		return mate3;
	}
	int cha = ((mate1[1] - mate2[1])>0) ? (mate1[1] - mate2[1]) : (mate2[1] - mate1[1]);//ȡ���߲�ľ���ֵ
	int sumk = mate1[0] + mate2[0];
	int central = (int)((mate1[1] + mate2[1]) / 2);
	if ((143 <= cha&&cha <= 163) || (235 <= cha&&cha <= 255))//����Զ���������߶μ��245������Զ��Ϊ153
	{
		mate3[0] = cha;
		mate3[1] = sumk;
		mate3[2] = central;

	}
	return mate3;
}
//---------------function�������ݽ���ɸѡ������������������-----------------
int zhuzi_x()
{
	vector<Vec3i>mate1(10);//����һ�������洢��Щ�ԱȽ��
	vector<Vec3i>mate2;//����һ�������洢����Ҫ��ĶԱȽ��
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
			cout << "��ֵ��sumk/central��  " << mate1[i][0] << "  " << mate1[i][1] << "  " << mate1[i][2] << endl;
			mate2.push_back(mate1[i]);
			//num_pipei++;
		}
	}
	cout << "mate2.size() :  " << mate2.size() << endl;
	if (mate2.size() == 1)//ֻ��һ��ƥ��������ֱ�ӷ���
	{
		return (mate2[0][2]);
	}
	else  if (mate2.size() > 1)//���ƥ��������Ҫ�󣬽���ɸѡ
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
	else//û������Ҫ���ƥ��㣬����0
	{
		cout << "û������Ҫ���ƥ�������뿼���Ƿ���bug��ͼ���Ƿ�������Ҫ��" << "  " << endl;
		return 0;
	}
}

//----------------function�������������y����-----------------------------------------------------
void seek_y(int &central_x, Mat& changban)
{
	if (central_x > x)//ֻ����xihua������������������x����Ž�����y����Ĳ���
	{
		Mat g_grayImage1;
		cout << "central_x  " << central_x << endl;
		//������ڷ��գ�������ӹ��ڿ��ߣ����������ȥroi��������ⳤ��Ļ��ͻ���籣��
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
		/***************************���£�����ֵ��ͼ�ȸ�ʴ�ڻ����߱任***********************/
		Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
		Mat erode1;
		erode(g_grayImage1, erode1, element);
		//imshow("erode1", erode1);
		Canny(erode1, erode1, 50, 80, 3);//����һ��canny��Ե���
		vector<Vec4i> lines5(100);//����һ��ʸ���ṹlines���ڴ�ŵõ����߶�ʸ������
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
				if (dy1 <= 10)//��֤ƽ�жȣ������������������gg��
				{
					aggregation_y1[j1] = (int)(l[3] + l[1]) / 2;
					j1++;
					//����ȡ����ֱ�߻�����
					line(changban_cy, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 2, CV_AA);
				}
			}
		}
		int min1 = aggregation_y1[0];
		//Ѱ����С��y���꣬Ҳ���ǳ�����ϱ�Ե��һ������Ĳ�������⣬�ڵذ��ϼ�⵽��ֱ�ߣ�����ͳ���
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
		//central_y=(1050+ min1-450);//�����������Զ1�׸ߵ����ӵ�
		//central_y = (1050 + min1 - 245);//245�����6��Զ0.5�׸ߵ����ӵ�,6mԶ1.5�׸߼�ȥ860
		if (min1 <= 20)
			central_y = 0;
		/********************************���ϣ�����ֵ��ͼ�ȸ�ʴ�ڻ����߱任*/

		/*******************����������һ�ַ���������ȡ��ͼ���˲���sobel����canny���ٻ���任************************************/
		Mat changban_cy1 = changban.clone();
		int g_nMedianBlurValue = 20;  //��ֵ�˲�����ֵ
		vector<size_t> aggregation_y2(100);

		Mat grad_y, g_dstImage4, img_canny1;
		medianBlur(changban_cy1, g_dstImage4, g_nMedianBlurValue * 2 + 1);
		//imshow("g_dstImage4", g_dstImage4);
		sobel(g_dstImage4, abs_grad_x1, abs_grad_y1, dst1);
		//imshow("abs_grad_y1", abs_grad_y1);
		canny(abs_grad_y1, img_canny1, 80, 120);
		//imshow("img_canny1", img_canny1);

		Mat midImage, dstImage;//��ʱ������Ŀ��ͼ�Ķ���
							   //  imshow("canny_again", midImage);

		cvtColor(midImage, dstImage, CV_GRAY2BGR);//ת����Ե�����ͼΪ�Ҷ�ͼ
		vector<Vec4i> lines6(100);//����һ��ʸ���ṹlines���ڴ�ŵõ����߶�ʸ������
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
			//Ѱ����С��y���꣬Ҳ���ǳ�����ϱ�Ե��һ������Ĳ�������⣬�ڵذ��ϼ�⵽��ֱ�ߣ�����ͳ���
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
			//central_y=(1050+ min2-450);//�����������Զ1�׸ߵ����ӵ�
			central_y = (1050 + min2 - 520);//245�����6��Զ0.5�׸ߵ����ӵ�,6mԶ1.5�׸߼�ȥ860
			imshow("changban_cy1", changban_cy1);
			if (min1 <= 20)
				central_y = 0;
		}
		/*******************��shang������һ�ַ���������ȡ��ͼ��sobel����canny���ٻ���任************************************/
	}
	else
	{
		cout << "�ϲ�����û����x���꣬���޷���ȡy���꣡" << "" << endl;
	}
}








