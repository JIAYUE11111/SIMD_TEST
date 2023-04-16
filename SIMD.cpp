#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2
#define INTERVAL 1000
using namespace std;
#include<assert.h>
#include<windows.h>
#include <stdio.h>
#include<iostream>
#include <time.h>
#include<cmath>
#include<fstream>
#include<sstream>
#include<string>
typedef long long ll;
const int ROWS = 14999;
const int COLS = 8;
const int dim = 8;
const int trainNum =12000 ;
const int testNum = 2999;
float test[testNum][dim];

float train[trainNum][dim];

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

void AVX_512()
{
	for (int i = 0; i < testNum; i++)
	{
		for (int j = 0; j < trainNum; j++)
		{
			assert(dim % 16 == 0);//
			__m512 sum = _mm512_setzero_ps();
			for (int k = 0; k < dim; k += 16)
			{
				__m512 temp_test = _mm512_load_ps(&test[i][k]);
				__m512 temp_train = _mm512_load_ps(&train[j][k]);
				temp_test = _mm512_sub_ps(temp_test, temp_train);
				temp_test = _mm512_mul_ps(temp_test, temp_test);
				sum = _mm512_add_ps(sum, temp_test);
			}
			__m128 sum1 = _mm512_extractf32x4_ps(sum, 0);
			__m128 sum2 = _mm512_extractf32x4_ps(sum, 1);
			__m128 sum3 = _mm512_extractf32x4_ps(sum, 2);
			__m128 sum4 = _mm512_extractf32x4_ps(sum, 3);
			sum1 = _mm_add_ps(sum1, sum3);
			sum2 = _mm_add_ps(sum4, sum2);
			sum1 = _mm_add_ps(sum1, sum2);
			sum1 = _mm_hadd_ps(sum1, sum1);
			sum1 = _mm_hadd_ps(sum1, sum1);
			_mm_store_ss(&dist[i][j], sum1);
		}
		for (int j = 0; j < trainNum; j += 16)
		{
			__m512 temp_dist = _mm512_load_ps(&dist[i][j]);
			temp_dist = _mm512_sqrt_ps(temp_dist);
			_mm512_store_ps(&dist[i][j], temp_dist);
		}
	}
}

void SSE()
{
	for (int i = 0; i < testNum; i++)
	{
		for (int j = 0; j < trainNum; j++)
		{
			assert(dim % 4 == 0);//首先假定维度为4的倍数
			__m128 sum = _mm_setzero_ps();
			for (int k = 0; k < dim; k += 4)
			{
				__m128 temp_test = _mm_load_ps(&test[i][k]);
				__m128 temp_train = _mm_load_ps(&train[j][k]);
				temp_test = _mm_sub_ps(temp_test, temp_train);
				temp_test = _mm_mul_ps(temp_test, temp_test);
				sum = _mm_add_ps(sum, temp_test);
			}
			sum = _mm_hadd_ps(sum, sum);
			sum = _mm_hadd_ps(sum, sum);
			_mm_store_ss(dist[i] + j, sum);
		}
		for (int j = 0; j < trainNum; j += 4)
		{
			__m128 temp_dist = _mm_load_ps(&dist[i][j]);
			temp_dist = _mm_sqrt_ps(temp_dist);
			_mm_store_ps(&dist[i][j], temp_dist);
		}
	}
}

void AVX()
{
	for (int i = 0; i < testNum; i++)
	{
		for (int j = 0; j < trainNum; j++)
		{
			assert(dim % 8 == 0);//首先假定维度为8的倍数
			__m256 sum = _mm256_setzero_ps();
			for (int k = 0; k < dim; k += 8)
			{
				__m256 temp_test = _mm256_load_ps(&test[i][k]);
				__m256 temp_train = _mm256_load_ps(&train[j][k]);
				temp_test = _mm256_sub_ps(temp_test, temp_train);
				temp_test = _mm256_mul_ps(temp_test, temp_test);
				sum = _mm256_add_ps(sum, temp_test);
			}
			__m256 hi = _mm256_permute2f128_ps(sum, sum, 1);
			sum = _mm256_add_ps(sum, hi);
			sum = _mm256_hadd_ps(sum, sum);
			sum = _mm256_hadd_ps(sum, sum);
			float tempArray[8];
			_mm256_store_ps(tempArray, sum);
			dist[i][j] = tempArray[0];
		}
		for (int j = 0; j < trainNum; j += 8)
		{
			__m256 temp_dist = _mm256_load_ps(&dist[i][j]);
			temp_dist = _mm256_sqrt_ps(temp_dist);
			_mm256_store_ps(&dist[i][j], temp_dist);
		}
	}
}

void unaligned()
{
	for (int i = 1; i <= testNum; i++)
	{
		for (int j = 1; j <= trainNum; j++)
		{
			assert(dim % 4 == 0);//首先假定维度为4的倍数
			__m128 sum = _mm_setzero_ps();
			for (int k = 1; k <= dim; k += 4)
			{
				__m128 temp_test = _mm_loadu_ps(&test[i][k]);
				__m128 temp_train = _mm_loadu_ps(&train[j][k]);
				temp_test = _mm_sub_ps(temp_test, temp_train);
				temp_test = _mm_mul_ps(temp_test, temp_test);
				sum = _mm_add_ps(sum, temp_test);
			}
			sum = _mm_hadd_ps(sum, sum);
			sum = _mm_hadd_ps(sum, sum);
			_mm_store_ss(dist[i] + j, sum);
		}
		for (int j = 1; j <= trainNum; j += 4)
		{
			__m128 temp_dist = _mm_loadu_ps(&dist[i][j]);
			temp_dist = _mm_sqrt_ps(temp_dist);
			_mm_storeu_ps(&dist[i][j], temp_dist);
		}
	}
}

void aligned()
{
	for (int i = 1; i <= testNum; i++)
	{
		for (int j = 1; j <= trainNum; j++)
		{
			assert(dim % 4 == 0);//首先假定维度为4的倍数
			__m128 sum = _mm_setzero_ps();
			float serial_sum = 0;
			for (int k = 1; k <= 3; k++)//处理1-3（头部）
			{
				float temp = test[i][k] - train[j][k];
				temp *= temp;
				serial_sum += temp;
			}
			//处理dim（尾部）
			float temp = test[i][dim] - train[j][dim];
			temp *= temp;
			serial_sum += temp;
			for (int k = 4; k < dim; k += 4)//4~dim-1是对齐的
			{
				__m128 temp_test = _mm_load_ps(&test[i][k]);
				__m128 temp_train = _mm_load_ps(&train[j][k]);
				temp_test = _mm_sub_ps(temp_test, temp_train);
				temp_test = _mm_mul_ps(temp_test, temp_test);
				sum = _mm_add_ps(sum, temp_test);
			}
			sum = _mm_hadd_ps(sum, sum);
			sum = _mm_hadd_ps(sum, sum);
			_mm_store_ss(dist[i] + j, sum);
			dist[i][j] += serial_sum;//串行与并行结果合并
		}
		for (int j = 1; j <= 3; j++)//处理1-3（头部）
			dist[i][j] = sqrtf(dist[i][j]);
		dist[i][trainNum] = sqrtf(dist[i][trainNum]);//处理trainNum（尾部）
		for (int j = 4; j < trainNum; j += 4)
		{
			__m128 temp_dist = _mm_load_ps(&dist[i][j]);
			temp_dist = _mm_sqrt_ps(temp_dist);
			_mm_store_ps(&dist[i][j], temp_dist);
		}
	}
}

void timing(void(*func)())
{
	ll head, tail, freq;
	double time = 0;
	int counter = 0;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	while (INTERVAL > time)
	{
		func();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		counter++;
		time = (tail - head) * 1000.0 / freq;
	}
	std::cout << time / counter << '\n';
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
	std::cout << myArray[14998][7 - 1] << endl;
	for (int i = 0; i < testNum; i++)
		for (int k = 0; k < dim; k++)
			test[i][k] = myArray[i][k];

	for (int i = 0; i < trainNum; i++)
		for (int k = 0; k < dim; k++)
			train[i][k] = myArray[i+testNum][k];
	std::cout << train[testNum- 1][dim - 1 - 1] << endl;
}
int main()
{
	init1();
	printf("%s%p\n", "train首地址", train);
	printf("%s%p\n", "test首地址", test);
	printf("%s%p\n", "dist首地址", dist);
	
	timing(plain);

	timing(SSE);

	timing(AVX);
	
	//timing(AVX_512);

	timing(unaligned);

	timing(aligned);
	return 0;
}