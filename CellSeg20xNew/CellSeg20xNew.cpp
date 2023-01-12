#include "pch.h"
#include "CellSeg20xNew.h"

#include "common.h"
#include <windows.h>
#include <direct.h>
#include <io.h>

#include "rt_nonfinite.h"
#include "watershed_matlab.h"
#include "watershed_matlab_terminate.h"
#include "watershed_matlab_emxAPI.h"
#include "watershed_matlab_initialize.h"

#include <opencv.hpp>
#include <opencv2\imgproc\types_c.h>



#define PI 3.1415926535
#define Lambda 0.532
#define Pixelsize 4.4
#define MAG 10

int AreaLimit_EGT;  //EGT最小细胞
int HoloLimit;  //EGT最大孔洞
float EGT_Finetune;
int AreaLimit_DT;  //DT最小细胞
float H_Index;

int count;
cv::Mat centroids, centroidsout, cell, mattest;
std::vector<std::vector<cv::Point>> contours;

int hist(float* dat, int* hist_num, int dat_len, int num, float max_val, float min_val);
void sort1(int* a, int length, int* b);
void RemoveSmallRegion(cv::Mat& Src, cv::Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode);
void morphReconstruct(cv::Mat& marker, cv::Mat& mask, cv::Mat& dst);
double graythresh(cv::Mat& _src);
cv::Mat EGT_Segmentation(cv::Mat& Src);
cv::Mat DT(cv::Mat& Src, cv::Mat& mask);
cv::Mat imreconstruct(cv::Mat marker, cv::Mat mask);
cv::Mat imimposemin(cv::Mat marker, cv::Mat mask);
static emxArray_real32_T* c_argInit_UnboundedxUnbounded_r(cv::Mat& im);

void PhaseImgSegmentation(cv::Mat& cell_img, int Width, int Height, int* count, cv::Mat& centroids, std::vector<std::vector<cv::Point>>* contours);

int test_int_array(void* array_int)
{
	int* p = (int*)array_int;
	int size = GetArraySize(p);
	AreaLimit_EGT = *p;
	HoloLimit = *(p + 1);
	AreaLimit_DT = *(p + 2);
	return 0;
}

int test_float_array(void* array_float)
{
	float* k = (float*)array_float;
	int size = GetArraySize(k);
	EGT_Finetune = *k;
	H_Index = *(k + 1);

	return 0;
}

int ImgNew(cv::Mat& phase_norm, cv::Mat& phase_new, int* histmax, int* histmin)
{
	cv::Mat temp;
	int nH = phase_norm.rows, nW = phase_norm.cols;
	phase_norm.copyTo(temp);

	if (*histmax == 0)
		*histmax = 1;
	for (int j = 0; j < nH; j++)
	{
		uchar* data = temp.ptr<uchar>(j);
		for (int i = 0; i < nW; i++)
		{
			if (data[i] < *histmin)
				data[i] = 0;
			else if (data[i] > *histmax)
				data[i] = 0;
			else
				data[i] = 1;
		}
	}
	temp.copyTo(phase_new);

	return 0;
}

void DrawCenter(cv::Mat& cell_img)
{
	for (int i = 0; i < centroids.rows; i++)
	{
		//找到连通域的质心
		int center_x = cvRound(centroids.at<float>(i, 0));
		int center_y = cvRound(centroids.at<float>(i, 1));

		//绘制中心点
		circle(cell_img, cv::Point(center_x, center_y), 3, cv::Scalar(50, 205, 50), 3, 8, 0);
	}

	cv::Mat centroidserror = centroidsout.clone();
	if (centroidserror.total() > 2)
	{
		for (int i = 0; i < centroidsout.rows; i++)
		{
			//找到连通域的质心
			int center_x = cvRound(centroidsout.at<float>(i, 0));
			int center_y = cvRound(centroidsout.at<float>(i, 1));

			//绘制中心点
			circle(cell_img, cv::Point(center_x, center_y), 3, cv::Scalar(50, 50, 205), 3, 8, 0);
		}
	}
}

void DrawContour(cv::Mat& cell_img)
{
	drawContours(cell_img, contours, -1, cv::Scalar(25, 234, 252), 1, 8);
}

void DrawNumber(cv::Mat& cell_img)
{
	//for (int i = 0; i < (contours).size(); i++)
	//{
	//	cv::Point p = cv::Point((contours)[i][0].x, (contours)[i][0].y);
	//	putText(cell_img, std::to_string(i + 1), p, 1, 2, cv::Scalar(255, 200, 200), 2);
	//}

	cv::Mat centroidsall;
	if (centroidsout.total() > 2)
	{
		cv::vconcat(centroids, centroidsout, centroidsall);
	}
	else
	{
		centroidsall = centroids.clone();
	}
	for (int i = 0; i < centroidsall.rows; i++)
	{
		cv::Point p = cv::Point(centroidsall.at<float>(i, 0), centroidsall.at<float>(i, 1));
		putText(cell_img, std::to_string(i + 1), p, 1, 2, cv::Scalar(255, 200, 200), 2, 8, false);
	}

	mattest = centroidsall.clone();
}

int DrawCell20xNew(cv::Mat& cell_img, int* center, int* contour, int* number)
{
	int channel = cell_img.channels();
	if (channel == 1)
	{
		cvtColor(cell_img, cell_img, CV_GRAY2BGR);
	}

	Logger::Log1(Severity::INFO, "Invoke DrawCell20xNew");
	Logger::Log2(Severity::INFO, L" DrawCell event: [%d, %d, %d]", *center, *contour, *number);

	if (*center == 1)
	{
		DrawCenter(cell_img);
		Logger::Log1(Severity::INFO, "Invoke DrawCenter");
	}
	if (*contour == 1)
	{
		DrawContour(cell_img);
		Logger::Log1(Severity::INFO, "Invoke DrawContour");
	}
	if (*number == 1)
	{
		DrawNumber(cell_img);
		Logger::Log1(Severity::INFO, "Invoke DrawNumber");
	}

	return 0;
}

int CellAnalysis(cv::Mat& cell, std::vector<std::vector<cv::Point>>* contours)
{
	cv::Mat area((*contours).size(), 1, CV_32FC1);
	for (size_t i = 0; i < (*contours).size(); i++)
	{
		double areatemp = cv::contourArea((*contours)[i]);
		area.at<float>(i, 0) = (float)areatemp;
	}
	//area = area * (Pixelsize / MAG) * (Pixelsize / MAG);

	cv::Mat perimeter((*contours).size(), 1, CV_32FC1);
	for (int i = 0; i < (*contours).size(); i++)
	{
		float output = arcLength((*contours)[i], true);
		perimeter.at<float>(i, 0) = output;
	}
	//perimeter = perimeter * (Pixelsize / MAG);

	cv::hconcat(area, perimeter, cell);

	cv::Mat roundness((*contours).size(), 1, CV_32FC1);
	roundness = 4 * PI * area / (perimeter.mul(perimeter));

	return 0;
}

int Phase_Seg20xNew(cv::Mat& cell_img, void* array_int, void* array_float, cv::Mat& cell)
{
	test_int_array(array_int);
	test_float_array(array_float);

	//mask = cv::Mat::zeros(cell_img.size(), CV_32FC1);
	//ImgNew(cell_img, mask, &histmax, &histmin);

	//cv::Mat cell_img_new = cell_img.mul(mask);

	cv::Mat cell_img_small;
	int Width = cell_img.cols, Height = cell_img.rows;
	cv::resize(cell_img, cell_img_small, cv::Size(Width / 2, Height / 2), 0, 0, cv::INTER_NEAREST);
	PhaseImgSegmentation(cell_img_small, Width, Height, &count, centroids, &contours);

	//int channel = cell_img.channels();
	//if (channel == 1)
	//{
	//	cvtColor(cell_img, cell_img, CV_GRAY2BGR);
	//}
	//DrawCenter(cell_img);
	//DrawContour(cell_img);
	//DrawNumber(cell_img);

	Logger::Log1(Severity::INFO, "Invoke Phase Seg20xNew");

	CellAnalysis(cell, &contours);
	if (cell.rows == centroids.rows)
	{
		cv::hconcat(cell, centroids, cell);
		cv::Mat cellout;
		copyMakeBorder(centroidsout, cellout, 0, 0, 2, 0, 0, cv::Scalar(0));
		cv::vconcat(cell, cellout, cell);
	}
	else
	{
		Logger::Log1(Severity::INFO, "Invoke Phase Seg20xNew error");
	}

	//cell_img.convertTo(cell_img, CV_8UC1, 255, 0);

	//Event::Trigger("Solution_img", &cell_img);

	return 0;
}

int test(cv::Mat& centroidstest)
{
	centroidstest = mattest.clone();

	return 0;
}

void PhaseImgSegmentation(cv::Mat& cell_img, int Width, int Height, int* count, cv::Mat& centroids, std::vector<std::vector<cv::Point>>* contours)
{
	cv::Mat I, I_norm;
	//cell_img.convertTo(I, CV_32FC1);
	//normalize(I, I_norm, 0, 255, cv::NORM_MINMAX);
	cell_img.convertTo(I_norm, CV_32FC1, 1.0 / 255.0);

	//开始分割----EGT
	cv::Mat fg_mask = EGT_Segmentation(I_norm);

	//DT
	cv::Mat detection = cv::Mat::zeros(I_norm.size(), CV_32FC1);
	cv::Mat imageBW = DT(I_norm, fg_mask);
	//I_norm.release();

	//时间作为随机种子防止重复
	cv::RNG rng(time(NULL));
	cv::Mat labels, stats, centroids_temp;
	int counts = connectedComponentsWithStats(imageBW, labels, stats, centroids_temp, 8, CV_16U);
	*count = counts - 1;

	imageBW.release();
	labels.release();
	stats.release();
	centroids_temp.convertTo(centroids_temp, CV_32FC1);

	cv::Mat centroids_tem(counts - 1, 2, CV_32FC1);
	int width = cell_img.cols, height = cell_img.rows;
	for (int i = 1; i < counts; i++)
	{
		int center_x = cvRound(centroids_temp.at<float>(i, 0));
		int center_y = cvRound(centroids_temp.at<float>(i, 1));

		centroids_tem.at<float>(i - 1, 0) = cvRound(float(center_x) * Width / width);
		centroids_tem.at<float>(i - 1, 1) = cvRound(float(center_y) * Width / width);

		detection.at<float>(center_y, center_x) = 1;
	}
	//centroids_tem.copyTo(centroids);

	//watershed
	cv::Mat imp = imimposemin(detection, -I_norm);
	watershed_matlab_initialize();

	emxArray_real32_T* output;
	emxArray_real32_T* input;
	emxInitArray_real32_T(&output, 2);
	input = c_argInit_UnboundedxUnbounded_r(imp);
	watershed_matlab(input, output);

	cv::Mat watershed(height, width, CV_32FC1);
	for (int x = 0; x < height; x++)
	{
		float* watershedptr = watershed.ptr<float>(x);
		for (int y = 0; y < width; y++)
		{
			watershedptr[y] = output->data[x + output->size[0] * y];
		}
	}
	emxDestroyArray_real32_T(output);
	emxDestroyArray_real32_T(input);
	watershed_matlab_terminate();
	imp.release();

	cv::Mat final_seg;
	final_seg = -watershed.mul(fg_mask);
	fg_mask.release();
	watershed.release();
	final_seg = -imimposemin(detection, final_seg);
	detection.release();

	cv::Mat watershed_result;
	final_seg.convertTo(watershed_result, CV_8UC1);
	final_seg.release();

	cv::resize(watershed_result, watershed_result, cv::Size(Width, Height), 0, 0, cv::INTER_NEAREST);
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(watershed_result, *contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());

	//counts = connectedComponentsWithStats(watershed_result, labels, stats, centroids1_result, 8, CV_16U);

	std::vector<std::vector<cv::Point>> temp;
	for (int i = 0; i < centroids_tem.rows; i++)
	{
		for (size_t j = 0; j < (*contours).size(); j++)
		{
			if (pointPolygonTest((*contours)[j], cv::Point2f(centroids_tem.at<float>(i, 0), centroids_tem.at<float>(i, 1)), false) >= 0)
			{
				temp.push_back((*contours)[j]);
				continue;
			}
		}
	}
	*contours = temp;

	cv::Mat centroidstemp, tem;
	copyMakeBorder(centroids_tem, tem, 0, 0, 0, 1, 0, cv::Scalar(0));
	for (int i = 0; i < centroids_tem.rows; i++)
	{
		for (size_t j = 0; j < (*contours).size(); j++)
		{
			if (pointPolygonTest((*contours)[j], cv::Point2f(centroids_tem.at<float>(i, 0), centroids_tem.at<float>(i, 1)), false) >= 0)
			{
				centroidstemp.push_back(centroids_tem.row(i));
				tem.at<float>(i, 2) = 1;
				break;
			}
		}
	}
	centroidsout = cv::Mat::zeros(cv::Size(2, 1), CV_32FC1);
	for (int i = 0; i < centroids_tem.rows; i++)
	{
		if (tem.at<float>(i, 2) == 0)
		{
			centroidsout.push_back(centroids_tem.row(i));
		}
	}
	centroidsout = centroidsout.rowRange(1, centroidsout.rows);
	centroids = centroidstemp.clone();
	centroids_tem.release();
	centroidstemp.release();
	tem.release();
}

int hist(float* dat, int* hist_num, int dat_len, int num, float max_val, float min_val)
{
	float len = max_val - min_val;
	float jiange = len / num;
	int a, j = 0;
	float  i;
	int hist_len;
	float* hist_mean;
	hist_mean = (float*)malloc(sizeof(float));
	for (i = min_val; i < max_val; i = i + jiange, j++)
	{
		*(hist_num + j) = 0;//初始化
		for (a = 0; a < dat_len; a++)//记录在某范围的数据个数 
		{
			if (j == 0) // [ ]
			{
				if ((i <= *(dat + a)) && (*(dat + a) <= i + jiange))
					*(hist_num + j) = *(hist_num + j) + 1;
			}
			else // ( ] ( ].......( ]
			{
				if ((i < *(dat + a)) && (*(dat + a) <= i + jiange))
					*(hist_num + j) = *(hist_num + j) + 1;
			}
		}
	}
	return (hist_len = j);
}

void sort1(int* a, int length, int* b)//降序
{
	int i, j, t1, t;
	for (j = 0; j < length; j++)
		for (i = 0; i < length - 1 - j; i++)
			if (*(a + i) < *(a + i + 1))
			{
				t = *(a + i);
				*(a + i) = *(a + i + 1);
				*(a + i + 1) = t;

				t1 = *(b + i);
				*(b + i) = *(b + i + 1);
				*(b + i + 1) = t1;
			}
}

void RemoveSmallRegion(cv::Mat& Src, cv::Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode)
{
	int RemoveCount = 0;       //记录除去的个数  
	//记录每个像素点检验状态的标签，0代表未检查，1代表正在检查,2代表检查不合格（需要反转颜色），3代表检查合格或不需检查  
	cv::Mat Pointlabel = cv::Mat::zeros(Src.size(), CV_32FC1);

	if (CheckMode == 1)
	{
		for (int i = 0; i < Src.rows; ++i)
		{
			float* iData = Src.ptr<float>(i);
			float* iLabel = Pointlabel.ptr<float>(i);
			for (int j = 0; j < Src.cols; ++j)
			{
				if (iData[j] < 10)
				{
					iLabel[j] = 3;
				}
			}
		}
	}
	else
	{
		for (int i = 0; i < Src.rows; ++i)
		{
			float* iData = Src.ptr<float>(i);
			float* iLabel = Pointlabel.ptr<float>(i);
			for (int j = 0; j < Src.cols; ++j)
			{
				if (iData[j] > 10)
				{
					iLabel[j] = 3;
				}
			}
		}
	}

	std::vector<cv::Point2i> NeihborPos;  //记录邻域点位置  
	NeihborPos.push_back(cv::Point2i(-1, 0));
	NeihborPos.push_back(cv::Point2i(1, 0));
	NeihborPos.push_back(cv::Point2i(0, -1));
	NeihborPos.push_back(cv::Point2i(0, 1));
	if (NeihborMode == 1)
	{
		NeihborPos.push_back(cv::Point2i(-1, -1));
		NeihborPos.push_back(cv::Point2i(-1, 1));
		NeihborPos.push_back(cv::Point2i(1, -1));
		NeihborPos.push_back(cv::Point2i(1, 1));
	}
	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//开始检测  
	for (int i = 0; i < Src.rows; ++i)
	{
		float* iLabel = Pointlabel.ptr<float>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 0)
			{
				//********开始该点处的检查**********  
				std::vector<cv::Point2i> GrowBuffer;  //堆栈，用于存储生长点  
				GrowBuffer.push_back(cv::Point2i(j, i));
				Pointlabel.at<float>(i, j) = 1;
				int CheckResult = 0;  //用于判断结果（是否超出大小），0为未超出，1为超出  

				for (int z = 0; z < GrowBuffer.size(); z++)
				{
					for (int q = 0; q < NeihborCount; q++)  //检查四个邻域点  
					{
						CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
						CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
						if (CurrX >= 0 && CurrX < Src.cols && CurrY >= 0 && CurrY < Src.rows)  //防止越界  
						{
							if (Pointlabel.at<float>(CurrY, CurrX) == 0)
							{
								GrowBuffer.push_back(cv::Point2i(CurrX, CurrY));  //邻域点加入buffer  
								Pointlabel.at<float>(CurrY, CurrX) = 1;  //更新邻域点的检查标签，避免重复检查  
							}
						}
					}
				}
				if (GrowBuffer.size() > AreaLimit) CheckResult = 2;  //判断结果（是否超出限定的大小），1为未超出，2为超出  
				else { CheckResult = 1;   RemoveCount++; }
				for (int z = 0; z < GrowBuffer.size(); z++)  //更新Label记录  
				{
					CurrX = GrowBuffer.at(z).x;
					CurrY = GrowBuffer.at(z).y;
					Pointlabel.at<float>(CurrY, CurrX) += CheckResult;
				}
				GrowBuffer.clear();
			}
		}
	}

	CheckMode = 255 * (1 - CheckMode);
	//开始反转面积过小的区域  
	for (int i = 0; i < Src.rows; ++i)
	{
		float* iData = Src.ptr<float>(i);
		float* iDstData = Dst.ptr<float>(i);
		float* iLabel = Pointlabel.ptr<float>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 2)
			{
				iDstData[j] = CheckMode;
			}
			else if (iLabel[j] == 3)
			{
				float tem1 = iData[j];
				float tem2 = iDstData[j];
				iDstData[j] = iData[j];
			}
		}
	}

	NeihborPos.clear();
}

void morphReconstruct(cv::Mat& marker, cv::Mat& mask, cv::Mat& dst)
{
	(cv::min)(marker, mask, dst);
	dilate(dst, dst, cv::Mat());
	(cv::min)(dst, mask, dst);
	cv::Mat temp1 = cv::Mat(marker.size(), CV_8UC1);
	cv::Mat temp2 = cv::Mat(marker.size(), CV_8UC1);
	do
	{
		dst.copyTo(temp1);
		dilate(dst, dst, cv::Mat());
		(cv::min)(dst, mask, dst);
		compare(temp1, dst, temp2, cv::CMP_NE);
	} while (sum(temp2).val[0] != 0);

}

cv::Mat EGT_Segmentation(cv::Mat& Src)
{
	// 求梯度
	cv::Mat xdst, ydst;
	Sobel(Src, xdst, Src.depth(), 1, 0, 1, 1, 0, cv::BORDER_DEFAULT);  //dst.depth()
	Sobel(Src, ydst, Src.depth(), 0, 1, 1, 1, 0, cv::BORDER_DEFAULT);

	cv::Mat dst(Src.size(), CV_32FC1);
	for (int row = 0; row < Src.rows; row++)
	{
		float* ptr = dst.ptr<float>(row);
		float* xptr = xdst.ptr<float>(row);
		float* yptr = ydst.ptr<float>(row);
		for (int col = 0; col < Src.cols; col++)
		{
			ptr[col] = sqrt(xptr[col] * xptr[col] + yptr[col] * yptr[col]);
		}
	}
	xdst.release();
	ydst.release();

	// 计算ratio
	double minv, maxv;
	cv::Point pt_min, pt_max;
	minMaxLoc(dst, &minv, &maxv, &pt_min, &pt_max);
	// hist函数
	int h = dst.rows, w = dst.cols;
	float* array1 = new float[w * h];
	for (int j = 0; j < h; j++)
	{
		float* ptr = dst.ptr<float>(j);
		for (int i = 0; i < w; i++)
		{
			array1[j * h + i] = ptr[i];
		}
	}

	int array_len = w * h;
	int* hist_num = new int[1000];
	int len = hist(array1, hist_num, array_len, 1000, float(maxv), float(minv));
	delete[] array1;
	// sort排序
	int* loc = new int[1000];
	for (int tm = 0; tm < 1000; tm++)
	{
		*(loc + tm) = tm + 1;
	}
	int* hist_num1 = new int[1000];
	for (int tp = 0; tp < 1000; tp++)
	{
		*(hist_num1 + tp) = *(hist_num + tp);
	}
	sort1(hist_num1, 1000, loc);

	int hist_mode_loc = round((*loc + *(loc + 1) + *(loc + 2)) / 3);
	//归一化hist_data
	float* temp_hist = new float[1000];
	int sum_hist_num = 0;
	for (int x = 0; x < 1000; x++)
	{
		sum_hist_num = sum_hist_num + *(hist_num + x);
	}
	for (int x1 = 0; x1 < 1000; x1++)
	{
		float A = *(hist_num + x1);
		float B = A / sum_hist_num * 100;
		*(temp_hist + x1) = B;
	}

	int lower_bound = 3 * hist_mode_loc;
	//归一化temp_hist
	float* norm_hist = new float[1000];
	float max_temp_hist = *(temp_hist + 0);
	for (int y = 0; y < 1000; y++)
	{
		if (max_temp_hist >= *(temp_hist + y + 1))
		{
			max_temp_hist = max_temp_hist;
		}
		else
		{
			max_temp_hist = *(temp_hist + y + 1);
		}
	}
	for (int y1 = 0; y1 < 1000; y1++)
	{
		float A1 = *(temp_hist + y1);
		float B1 = A1 / max_temp_hist;
		*(norm_hist + y1) = B1;
	}
	int idx = 0;
	for (int y2 = 4; y2 < 1000; y2++)
	{
		if (*(norm_hist + y2) > 0.05)
		{
			idx++;
		}
		else
		{
			idx = idx;
		}
	}
	idx = idx + hist_mode_loc - 1;
	int upper_bound;
	if (idx > 18 * hist_mode_loc)
	{
		upper_bound = idx;
	}
	else
	{
		upper_bound = 18 * hist_mode_loc;
	}
	float density_metric;
	density_metric = *(temp_hist + lower_bound - 1);
	for (int y3 = lower_bound; y3 < upper_bound; y3++)
	{
		density_metric = density_metric + *(temp_hist + y3);
	}
	float saturation1 = 3;
	float saturation2 = 42;
	float a = (95 - 40) / (saturation1 - saturation2);
	float b = 95 - a * saturation1;
	float prct_value = round(a * density_metric + b);
	if (prct_value > 98)
	{
		prct_value = 98;
	}
	if (prct_value < 25)
	{
		prct_value = 25;
	}
	float greedy_step = 1;
	prct_value = prct_value - greedy_step * EGT_Finetune;
	if (prct_value > 100)
	{
		prct_value = 100;
	}
	if (prct_value < 1)
	{
		prct_value = 1;
	}
	prct_value = prct_value / 100;
	// percentile_computation部分
	cv::Mat Ou;
	int indx = round(prct_value * (w * h) + 1);
	if (indx < 1)
	{
		indx = 1;
	}
	if (indx > w * h)
	{
		indx = Ou.rows;
	}
	div_t res = div(indx, h);
	float T = 0;
	int index1 = res.quot + 1;
	int index2 = res.rem;
	cv::Mat flat;
	dst.reshape(1, 1).copyTo(flat);
	cv::sort(flat, flat, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);

	flat.reshape(1, h).copyTo(Ou);
	T = Ou.at<float>(index1, index2);
	for (int i = 0; i < h; i++)
	{
		float* dstptr = dst.ptr<float>(i);
		for (int j = 0; j < w; j++)
		{
			if (dstptr[j] >= T)
			{
				dstptr[j] = 1;
			}
			else
			{
				dstptr[j] = 0;
			}
		}
	}
	Ou.release();
	flat.release();

	// fill_holo部分
	multiply(dst, 255, dst);
	cv::Mat Dst = cv::Mat::zeros(dst.size(), CV_32FC1);
	RemoveSmallRegion(dst, Dst, HoloLimit, 0, 1);
	dst.release();

	// imrode部分
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::Mat dimg;
	erode(Dst, dimg, element);  //和matlab中的imerode相对应,腐蚀函数的调用过程
	// bwareaopen部分
	cv::Mat fg_mask = cv::Mat::zeros(Dst.size(), CV_32FC1);
	Dst.release();

	RemoveSmallRegion(dimg, fg_mask, AreaLimit_EGT, 1, 1);
	element.release();
	dimg.release();

	medianBlur(fg_mask, fg_mask, 5);
	cv::Mat element20 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	erode(fg_mask, fg_mask, element20);  //和matlab中的imerode相对应,腐蚀函数的调用过程
	medianBlur(fg_mask, fg_mask, 5);

	return fg_mask;
}

cv::Mat DT(cv::Mat& Src, cv::Mat& mask)
{
	cv::Mat m;
	double DT_Treshold = graythresh(Src);
	DT_Treshold = DT_Treshold * 0.8;
	threshold(Src, m, DT_Treshold, 1, 0);
	normalize(m, m, 0, 255, cv::NORM_MINMAX);

	cv::Mat tem = cv::Mat::zeros(m.size(), CV_32FC1);
	cv::Mat output = cv::Mat::zeros(m.size(), CV_32FC1);
	//cv::Mat element20 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 1));
	//dilate(m, m, element20);  //和matlab中的imerode相对应,腐蚀函数的调用过程
	RemoveSmallRegion(m, tem, HoloLimit, 0, 1);
	RemoveSmallRegion(tem, output, AreaLimit_DT, 1, 1);
	output.convertTo(output, CV_8UC1, 255);
	m.release();
	tem.release();

	cv::Mat distanceImg(output.size(), CV_32FC1);
	distanceTransform(output, distanceImg, cv::DIST_L2, cv::DIST_MASK_PRECISE);

	cv::Mat imhmaxImg = imreconstruct(distanceImg - H_Index, distanceImg);
	distanceImg.release();

	cv::Mat m1_one(imhmaxImg.size(), CV_32FC1, cv::Scalar(1));
	cv::Mat m1_diff, result, imageBW;
	absdiff(imhmaxImg, m1_one, m1_diff);
	morphReconstruct(m1_diff, imhmaxImg, output);
	absdiff(imhmaxImg, output, result);
	m1_one.release();
	m1_diff.release();
	imhmaxImg.release();
	output.release();

	normalize(result, result, 0, 255, cv::NORM_MINMAX);
	threshold(result, imageBW, 254, 255, cv::THRESH_BINARY);//二值化阈值调整
	imageBW = imageBW.mul(mask);
	imageBW.convertTo(imageBW, CV_8UC1);
	result.release();

	return imageBW;
}

double graythresh(cv::Mat& _src)
{
	cv::Size size = _src.size();
	if (_src.isContinuous())
	{
		size.width *= size.height;
		size.height = 1;
	}
	const int N = 256;
	int i, j, h[N] = { 0 };
	for (i = 0; i < size.height; i++)
	{
		const uchar* src = _src.data + _src.step * i;
		for (j = 0; j <= size.width - 4; j += 4)
		{
			int v0 = src[j], v1 = src[j + 1];
			h[v0]++; h[v1]++;
			v0 = src[j + 2]; v1 = src[j + 3];
			h[v0]++; h[v1]++;
		}
		for (; j < size.width; j++)
			h[src[j]]++;
	}

	double mu = 0, scale = 1. / (size.width * size.height);
	for (i = 0; i < N; i++)
		mu += i * h[i];

	mu *= scale;
	double mu1 = 0, q1 = 0;
	double max_sigma = 0, max_val = 0;

	for (i = 0; i < N; i++)
	{
		double p_i, q2, mu2, sigma;

		p_i = h[i] * scale;
		mu1 *= q1;
		q1 += p_i;
		q2 = 1. - q1;

		if (std::min(q1, q2) < FLT_EPSILON || std::max(q1, q2) > 1. - FLT_EPSILON)
			continue;

		mu1 = (mu1 + i * p_i) / q1;
		mu2 = (mu - q1 * mu1) / q2;
		sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
		if (sigma > max_sigma)
		{
			max_sigma = sigma;
			max_val = i;
		}
	}

	max_val = max_val / 255;

	return max_val;
}

cv::Mat imreconstruct(cv::Mat marker, cv::Mat mask)
{
	cv::Mat curr_marker;
	marker.copyTo(curr_marker);
	cv::Mat kernel = cv::Mat::ones(3, 3, CV_32F);
	cv::Mat next_marker;
	while (true)
	{
		dilate(curr_marker, next_marker, kernel);
		min(next_marker, mask, next_marker);
		if (sum(curr_marker != next_marker) == cv::Scalar(0, 0, 0, 0))
		{
			return curr_marker;
		}
		next_marker.copyTo(curr_marker);
	}
}

cv::Mat imimposemin(cv::Mat marker, cv::Mat mask)
{
	cv::Mat fm;
	mask.copyTo(fm);
	for (int i = 0; i < fm.rows; i++)
	{
		float* ptr = fm.ptr<float>(i);
		float* ptr1 = marker.ptr<float>(i);
		for (int j = 0; j < fm.cols; j++)
		{
			if (ptr1[j] == 0)
			{
				ptr[j] = std::numeric_limits<float>::infinity();
			}
			else {
				ptr[j] = -std::numeric_limits<float>::infinity();
			}
		}
	}

	float h = 0;
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;

	minMaxLoc(mask, &minVal, &maxVal, &minLoc, &maxLoc);
	float range = float(maxVal - minVal);
	if (range == 0)
	{
		h = 0.1;
	}
	else
	{
		h = range * 0.001;
	}
	cv::Mat g;
	min(mask + h, fm, g);
	fm = 1 - fm;
	g = 1 - g;
	cv::Mat J = imreconstruct(fm, g);
	return 1 - J;
}

static emxArray_real32_T* c_argInit_UnboundedxUnbounded_r(cv::Mat& im)
{
	emxArray_real32_T* result;
	int idx0;
	int idx1;
	result = emxCreate_real32_T(im.rows, im.cols);
	for (idx0 = 0; idx0 < result->size[0U]; idx0++)
	{
		float* ptr = im.ptr<float>(idx0);
		for (idx1 = 0; idx1 < result->size[1U]; idx1++)
		{
			result->data[idx0 + result->size[0] * idx1] = ptr[idx1];
		}
	}
	return result;
}