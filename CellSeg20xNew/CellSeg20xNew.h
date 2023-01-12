#pragma once
#include <opencv2/core.hpp>

#ifdef CellSeg_EXPORTS
#define CellSeg20xNew_API __declspec(dllexport)
#else
#define CellSeg20xNew_API __declspec(dllimport)
#endif


/// <summary>
/// 相位图细胞计数 20X物镜
/// </summary>
/// <param name="cell_img"></param>
/// <param name="array_int"></param>
/// <param name="array_float"></param>
/// <param name="cell"></param>
/// <returns></returns>
extern "C" CellSeg20xNew_API int Phase_Seg20xNew(cv::Mat & cell_img, void* array_int, void* array_float, cv::Mat & cell);

/// <summary>
/// 绘制带标志图像
/// </summary>
/// <param name="cell_img"></param>
/// <param name="center"></param>
/// <param name="contour"></param>
/// <param name="number"></param>
/// <returns></returns>
extern "C" CellSeg20xNew_API int DrawCell20xNew(cv::Mat & cell_img, int* center, int* contour, int* number);

extern "C" CellSeg20xNew_API int test(cv::Mat & centroidstest);