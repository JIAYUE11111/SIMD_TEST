#include <arm_neon.h>
#include<assert.h>
#include <stdio.h>
#include <sys/time.h>
#include<cmath>
#include <iostream>
#include<fstream>
#include<sstream>
#include<string>
#define INTERVAL 1000
using namespace std;
typedef long long ll;
const int ROWS = 14999;
const int COLS = 8;
const int dim = 8;
const int trainNum = 12000;
const int testNum = 2999;
float train[trainNum][dim];
float test[testNum][dim];
float dist[testNum][trainNum];
void plain()
{
	for (int i = 0; i < testNum; i++)
	{
		for (int j = 0; j < trainNum; j++)
		{
			float sum = 0.0;
			for (int k = 0; k < dim; k++)
			{
				float temp = test[i][k] - train[j][k];
				temp *= temp;
				sum += temp;
			}
			dist[i][j] = sqrtf(sum);
		}
	}
}

void one_cycle_unwrapped()
{
	for (int i = 0; i < testNum; i++)
	{
		for (int j = 0; j < trainNum; j++)
		{
			float32x4_t sum = vmovq_n_f32(0);
			for (int k = 0; k < dim - 3; k += 4)//处理4的倍数
			{
				float32x4_t temp_test = vld1q_f32(&test[i][k]);
				float32x4_t temp_train = vld1q_f32(&train[j][k]);
				temp_test = vsubq_f32(temp_test, temp_train);
				//temp_test = vmulq_f32(temp_test, temp_test);
				//sum = vaddq_f32(sum, temp_test);
				sum = vmlaq_f32(sum, temp_test, temp_test);
			}
			float32x2_t sumlow = vget_low_f32(sum);
			float32x2_t sumhigh = vget_high_f32(sum);
			sumlow = vpadd_f32(sumlow, sumhigh);
			float32_t sumlh = vpadds_f32(sumlow);
			for (int k = dim - dim % 4; k < dim; k++)//串行处理尾部
			{
				float temp = test[i][k] - train[j][k];
				temp *= temp;
				sumlh += temp;
			}
			dist[i][j] = sqrtf(sumlh);
		}
	}
}

void sqrt_unwrapped()
{
	for (int i = 0; i < testNum; i++)
	{
		for (int j = 0; j < trainNum; j++)
		{
			float32x4_t sum = vmovq_n_f32(0);
			for (int k = 0; k < dim - 3; k += 4)//处理4的倍数
			{
				float32x4_t temp_test = vld1q_f32(&test[i][k]);
				float32x4_t temp_train = vld1q_f32(&train[j][k]);
				temp_test = vsubq_f32(temp_test, temp_train);
				//temp_test = vmulq_f32(temp_test, temp_test);
				//sum = vaddq_f32(sum, temp_test);
				sum = vmlaq_f32(sum, temp_test, temp_test);
			}
			float32x2_t sumlow = vget_low_f32(sum);
			float32x2_t sumhigh = vget_high_f32(sum);
			sumlow = vpadd_f32(sumlow, sumhigh);
			float32_t sumlh = vpadds_f32(sumlow);
			for (int k = dim - dim % 4; k < dim; k++)//串行处理尾部
			{
				float temp = test[i][k] - train[j][k];
				temp *= temp;
				sumlh += temp;
			}
			dist[i][j] = sumlh;
		}
	}
	for (int j = 0; j < trainNum * testNum - 3; j += 4)//处理4的倍数
	{
		float32x4_t temp_dist = vld1q_f32(&dist[0][0] + j);
		temp_dist = vsqrtq_f32(temp_dist);
		vst1q_f32(&dist[0][0] + j, temp_dist);
	}
	for (int j = trainNum * testNum - (trainNum * testNum) % 4; j < trainNum * testNum; j++)//串行处理尾部
		*(dist[0] + j) = sqrtf(*(dist[0] + j));
}

void sqrt_unwrapped_cached()
{
	for (int testZone = 0; testZone < 4; testZone++)
		for (int trainZone = 0; trainZone < 16; trainZone++)
		{
			for (int i = testZone * testNum / 4; i < (testZone + 1) * testNum / 4; i++)
			{
				for (int j = trainZone * trainNum / 16; j < (trainZone + 1) * trainNum / 16; j++)
				{
					float32x4_t sum = vmovq_n_f32(0);
					for (int k = 0; k < dim - 3; k += 4)//处理4的倍数
					{
						float32x4_t temp_test = vld1q_f32(&test[i][k]);
						float32x4_t temp_train = vld1q_f32(&train[j][k]);
						temp_test = vsubq_f32(temp_test, temp_train);
						//temp_test = vmulq_f32(temp_test, temp_test);
						//sum = vaddq_f32(sum, temp_test);
						sum = vmlaq_f32(sum, temp_test, temp_test);
					}
					float32x2_t sumlow = vget_low_f32(sum);
					float32x2_t sumhigh = vget_high_f32(sum);
					sumlow = vpadd_f32(sumlow, sumhigh);
					float32_t sumlh = vpadds_f32(sumlow);
					for (int k = dim - dim % 4; k < dim; k++)//串行处理尾部
					{
						float temp = test[i][k] - train[j][k];
						temp *= temp;
						sumlh += temp;
					}
					dist[i][j] = sumlh;
				}
			}
		}
	for (int j = 0; j < trainNum * testNum - 3; j += 4)//处理4的倍数
	{
		float32x4_t temp_dist = vld1q_f32(&dist[0][0] + j);
		temp_dist = vsqrtq_f32(temp_dist);
		vst1q_f32(&dist[0][0] + j, temp_dist);
	}
	for (int j = trainNum * testNum - (trainNum * testNum) % 4; j < trainNum * testNum; j++)//串行处理尾部
		*(dist[0] + j) = sqrtf(*(dist[0] + j));
}

void vertical_SIMD()
{
	for (int i = 0; i < testNum; i++)
	{
		for (int j = 0; j < trainNum - 3; j += 4)//并行处理4的倍数部分
		{
			float32x4_t sum = vmovq_n_f32(0);
			for (int k = 0; k < dim; k++)
			{
				float32x4_t temp_train, temp_test;
				temp_train = vld1q_lane_f32(&train[j][k], sum, 0);
				temp_train = vld1q_lane_f32(&train[j + 1][k], sum, 1);
				temp_train = vld1q_lane_f32(&train[j + 2][k], sum, 2);
				temp_train = vld1q_lane_f32(&train[j + 3][k], sum, 3);
				temp_test = vld1q_dup_f32(&test[i][k]);
				temp_test = vsubq_f32(temp_test, temp_train);
				temp_test = vaddq_f32(temp_test, temp_test);
				sum = vaddq_f32(temp_test, sum);
			}
			vst1q_f32(&dist[i][j], sum);
		}
		for (int j = trainNum - trainNum % 4; j < trainNum; j++)//串行处理剩余部分
		{
			float sum = 0.0;
			for (int k = 0; k < dim; k++)
			{
				float temp = test[i][k] - train[j][k];
				temp *= temp;
				sum += temp;
			}
			dist[i][j] = sqrtf(sum);
		}
	}
}

void vertical_SIMD_cached()
{
	float train_conv[dim][trainNum];
	for (int i = 0; i < trainNum; i++)
		for (int j = 0; j < dim; j++)
			train_conv[j][i] = train[i][j];
	for (int i = 0; i < testNum; i++)
	{
		for (int j = 0; j < trainNum - 3; j += 4)//并行处理4的倍数部分
		{
			float32x4_t sum = vmovq_n_f32(0);
			for (int k = 0; k < dim; k++)
			{
				float32x4_t temp_train, temp_test;
				temp_train = vld1q_f32(&train_conv[k][j]);
				temp_test = vld1q_dup_f32(&test[i][k]);
				temp_test = vsubq_f32(temp_test, temp_train);
				temp_test = vaddq_f32(temp_test, temp_test);
				sum = vaddq_f32(temp_test, sum);
			}
			vst1q_f32(&dist[i][j], sum);
		}
		for (int j = trainNum - trainNum % 4; j < trainNum; j++)//串行处理剩余部分
		{
			float sum = 0.0;
			for (int k = 0; k < dim; k++)
			{
				float temp = test[i][k] - train[j][k];
				temp *= temp;
				sum += temp;
			}
			dist[i][j] = sqrtf(sum);
		}
	}
}

void square_unwrapped()
{
	float temp_test[testNum];
	float temp_train[trainNum];
	for (int i = 0; i < testNum; i++)
	{
		float sum = 0.0;
		for (int j = 0; j < dim; j++)
			sum += test[i][j] * test[i][j];
		temp_test[i] = sum;
	}
	for (int i = 0; i < trainNum; i++)
	{
		float sum = 0.0;
		for (int j = 0; j < dim; j++)
			sum += train[i][j] * train[i][j];
		temp_train[i] = sum;
	}
	for (int i = 0; i < testNum; i++)
		for (int j = 0; j < trainNum; j++)
		{
			float sum = 0;
			for (int k = 0; k < dim; k++)
				sum += test[i][k] * train[j][k];
			dist[i][j] = sqrtf(temp_test[i] + temp_train[j] - 2 * sum);
		}
}

void square_unwrapped_NEON()
{
	float temp_test[testNum];
	float temp_train[trainNum];
	for (int i = 0; i < testNum; i++)
	{
		float32x4_t sum = vmovq_n_f32(0);
		for (int j = 0; j < dim - 3; j += 4)
		{
			float32x4_t square = vld1q_f32(&test[i][j]);
			sum = vmlaq_f32(sum, square, square);
		}
		float32x2_t sumlow = vget_low_f32(sum);
		float32x2_t sumhigh = vget_high_f32(sum);
		sumlow = vpadd_f32(sumlow, sumhigh);
		float32_t sumlh = vpadds_f32(sumlow);
		for (int k = dim - dim % 4; k < dim; k++)//串行处理尾部
			sumlh += test[i][k] * test[i][k];
		temp_test[i] = sumlh;
	}
	for (int i = 0; i < trainNum; i++)
	{
		float32x4_t sum = vmovq_n_f32(0);
		for (int j = 0; j < dim - 3; j += 4)
		{
			float32x4_t square = vld1q_f32(&train[i][j]);
			sum = vmlaq_f32(sum, square, square);
		}
		float32x2_t sumlow = vget_low_f32(sum);
		float32x2_t sumhigh = vget_high_f32(sum);
		sumlow = vpadd_f32(sumlow, sumhigh);
		float32_t sumlh = vpadds_f32(sumlow);
		for (int k = dim - dim % 4; k < dim; k++)//串行处理尾部
			sumlh += train[i][k] * train[i][k];
		temp_train[i] = sumlh;
	}
	for (int i = 0; i < testNum; i++)
	{
		for (int j = 0; j < trainNum; j++)
		{
			float32x4_t sum = vmovq_n_f32(0);
			for (int k = 0; k < dim - 3; k += 4)
			{
				float32x4_t _train = vld1q_f32(&train[j][k]);
				float32x4_t _test = vld1q_f32(&test[i][k]);
				_train = vmulq_f32(_train, _test);
				sum = vaddq_f32(_train, sum);
			}
			float32x2_t sumlow = vget_low_f32(sum);
			float32x2_t sumhigh = vget_high_f32(sum);
			sumlow = vpadd_f32(sumlow, sumhigh);
			float32_t sumlh = vpadds_f32(sumlow);
			for (int k = dim - dim % 4; k < dim; k++)//串行处理尾部
				sumlh += train[j][k] * test[i][k];
			dist[i][j] = sumlh;
		}
		//dist[i][j] = sqrtf(temp_test[i] + temp_train[j] - 2 * sum);
		float32x4_t _test = vld1q_dup_f32(&temp_test[i]);
		for (int j = 0; j < trainNum - 3; j += 4)
		{
			float32x4_t _train = vld1q_f32(&temp_train[j]);
			float32x4_t res = vld1q_f32(&dist[i][j]);
			res = vmulq_n_f32(res, -2);
			res = vaddq_f32(_train, res);
			res = vaddq_f32(_test, res);
			res = vsqrtq_f32(res);
			vst1q_f32(&dist[i][j], res);
		}
		for (int j = trainNum - trainNum % 4; j < trainNum; j++)//串行处理尾部
			dist[i][j] = sqrtf(temp_test[i] + temp_train[j] - 2 * dist[i][j]);
	}
}

void timing(void(*func)())
{
	timeval tv_begin, tv_end;
	int counter(0);
	double time = 0;
	gettimeofday(&tv_begin, 0);
	while (INTERVAL > time)
	{
		func();
		gettimeofday(&tv_end, 0);
		counter++;
		time = ((ll)tv_end.tv_sec - (ll)tv_begin.tv_sec)*1000.0 + ((ll)tv_end.tv_usec - (ll)tv_begin.tv_usec) / 1000.0;
	}
	cout << time / counter << "," << counter << '\n';
}

void init()
{
	for (int i = 0; i < testNum; i++)
		for (int k = 0; k < dim; k++)
			test[i][k] = rand() / double(RAND_MAX) * 1000;//0-100间随机浮点数
	for (int i = 0; i < trainNum; i++)
		for (int k = 0; k < dim; k++)
			train[i][k] = rand() / double(RAND_MAX) * 1000;//0-100间随机浮点数
}
void init1()
{
	float myArray[ROWS][COLS]; // 定义二维数组

	   // 读取CSV文件
	std::ifstream file("D:\\data.csv");
	if (file.is_open()) {
		std::string line;
		int row = 0;
		while (std::getline(file, line) && row < ROWS) {
			std::istringstream ss(line);
			std::string cell;
			int col = 0;
			while (std::getline(ss, cell, ',') && col < COLS) {
				// 将从CSV文件读取的字符串转换为整数，并存入二维数组
				double value = std::stod(cell);
				myArray[row][col] = value;
				col++;
			}
			row++;
		}
		file.close();
	}
	else {
		std::cout << "Failed to open file." << std::endl;

	}
	for (int i = 0; i < testNum; i++)
		for (int k = 0; k < dim; k++)
			test[i][k] = myArray[i][k];

	for (int i = 0; i < trainNum; i++)
		for (int k = 0; k < dim; k++)
			train[i][k] = myArray[i + testNum][k];
}
int main()
{
	float distComp[testNum][trainNum];
	init1();
	//printf("%s", "朴素算法耗时：");
	timing(plain);
	float error = 0;
	for (int i = 0; i < testNum; i++)
		for (int j = 0; j < trainNum; j++)
			distComp[i][j] = dist[i][j];
	
	//printf("%s", "SIMD算法耗时：");
	timing(one_cycle_unwrapped);
	for (int i = 0; i < testNum; i++)
		for (int j = 0; j < trainNum; j++)
			error += (distComp[i][j] - dist[i][j]) * (distComp[i][j] - dist[i][j]);
	//printf("误差%f\n", error);
	cout << error << endl;
	error = 0;
	//printf("%s", "开方SIMD算法耗时：");
	timing(sqrt_unwrapped);
	for (int i = 0; i < testNum; i++)
		for (int j = 0; j < trainNum; j++)
			error += (distComp[i][j] - dist[i][j]) * (distComp[i][j] - dist[i][j]);
	//printf("误差%f\n", error);
	cout << error << endl;
	error = 0;
	//printf("%s", "cache优化开方SIMD算法耗时：");
	timing(sqrt_unwrapped_cached);
	for (int i = 0; i < testNum; i++)
		for (int j = 0; j < trainNum; j++)
			error += (distComp[i][j] - dist[i][j]) * (distComp[i][j] - dist[i][j]);
	//printf("误差%f\n", error);
	cout << error << endl;
	error = 0;
	//printf("%s", "纵向SIMD算法耗时：");
	timing(vertical_SIMD);
	for (int i = 0; i < testNum; i++)
		for (int j = 0; j < trainNum; j++)
			error += (distComp[i][j] - dist[i][j]) * (distComp[i][j] - dist[i][j]);
	//printf("误差%f\n", error);
	cout << error << endl;
	error = 0;
	//printf("%s", "纵向SIMD+cache算法耗时：");
	timing(vertical_SIMD_cached);
	for (int i = 0; i < testNum; i++)
		for (int j = 0; j < trainNum; j++)
			error += (distComp[i][j] - dist[i][j]) * (distComp[i][j] - dist[i][j]);
	//printf("误差%f\n", error);
	cout << error << endl;
	//printf("%s", "串行平方展开算法耗时：");
	timing(square_unwrapped);
	error = 0;
	//printf("%s", "SIMD平方展开算法耗时：");
	timing(square_unwrapped_NEON);
	for (int i = 0; i < testNum; i++)
		for (int j = 0; j < trainNum; j++)
			error += (distComp[i][j] - dist[i][j]) * (distComp[i][j] - dist[i][j]);
	//printf("误差%f\n", error);
	cout << error << endl;
}