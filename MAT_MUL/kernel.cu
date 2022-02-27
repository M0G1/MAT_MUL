#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;



//функция ядра
__global__ void matrixMult(const double* A, const double* B, double* C, int n)
{
	int ai = n * (blockDim.y * blockIdx.y + threadIdx.y);	// индекс начала строки матрицы A
	int bj = blockDim.x * blockIdx.x + threadIdx.x;			// индекс начала строки матрицы B
	double sum = 0;											// промежуточная переменная для вычиселний
	for (int k = 0; k < n; k++)
		sum += A[ai + k] * B[k * n + bj];					// вычисление произведения
	int index = n * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x; // индекс вычисляемого элемента матрицы C 
	C[index] = sum;											// заполнение массива результатми
}

// генерация матриц
double* generateRandMatrix(int n, size_t sizeMatrix) {
	double* matrix = (double*)malloc(sizeMatrix);			// выделение памяти под массив
	for (int i = 0; i < n * n; i++) {
		matrix[i] = (double)rand() / (double)RAND_MAX;		// заполнение массива случайными числами
	}
	return matrix;											// возврат заполненной матрицы
}

// вывод матрицы в консоль
void printMatrix(double* matrix, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%4.1lf ", matrix[i * n + j]);
		}
		printf("\n");
	}
}

// функция для последовательного варианта умножения матриц
void matrixMultCPU(double* A, double* B, double* C, int n) {
	// реализация математического алгоритма умножения матриц
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			for (int k = 0; k < n; k++) {
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
}

// проверка результатов умножения
bool checkMult(double* C1, double* C2, int n) {
	double accuracy = 1.e-6;						//точность с которой будем производить проверку
	for (int i = 0; i < n * n; i++) {				// в цикле идем по всем ячейкам и 
		if (abs(C1[i] - C2[i]) >= accuracy)			// и проверяем если модуль разницы между значением полученным 
													// на ЦП и ГП больше либо равна нуля то тогда матрица посчитана неверно
			return false;
	}
	return true;									// иначе все пучком и матрица посчитана четко
}

int main(int argc, char* argv[])
{
	int BLOCK_SIZE = 32;
	setlocale(LC_ALL, "RUS");						// функция для НОРМАЛЬНОГО отображени кириллицы в консоли.
	for (int ii = 11; ii < 12; ++ii)
	{	
		printf("%d\n",ii);
		double N = 1 << ii;										//размерность массива
		switch (ii) // определим для каждой размерности данных свой размер блока(нет)
		{
			case 7:
				BLOCK_SIZE = 16;
				break;
			case 8:
				BLOCK_SIZE = 16;
				break;
			case 9:
				BLOCK_SIZE = 16;
				break;
			case 10:
				BLOCK_SIZE = 16;
				break;
			default:
				BLOCK_SIZE = 16;
				break;
		}


		// события начала и окончания времени
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		srand(time(NULL));
		size_t sizeMatrix = sizeof(double) * N * N;			// расширенная размерность массива на всякий случай

		// генерация массивов для работы на центральном процессоре
		double* A_CPU = generateRandMatrix(N, sizeMatrix);
		double* B_CPU = generateRandMatrix(N, sizeMatrix);
		double* C_CPU = (double*)malloc(sizeMatrix);			// матрица для копирования данных из памяти графического процессора
		double* C_seq_CPU = (double*)malloc(sizeMatrix);
		for (int i = 0; i < N * N; i++) {
			C_seq_CPU[i] = 0;
		}

		// high_resolution_clock - улучшенная версия функции clock
		high_resolution_clock::time_point t1 = high_resolution_clock::now();		// начало отсчета времени 
		matrixMultCPU(A_CPU, B_CPU, C_seq_CPU, N);									// расчет матричного произведения
		high_resolution_clock::time_point t2 = high_resolution_clock::now();		// окончание отсчета времени 
		duration<double, std::milli> time_span = t2 - t1;							// расчет затраченного времени						
		double cpu_time = time_span.count();
		printf("%f The time:  milliseconds\n", cpu_time);

		// выделение памяти на графическом процессоре
		double* A_GPU;
		cudaMalloc((void**)&A_GPU, sizeMatrix);
		double* B_GPU;
		cudaMalloc((void**)&B_GPU, sizeMatrix);
		double* C_GPU;
		cudaMalloc((void**)&C_GPU, sizeMatrix);

		// копирование данных в память графического процессора
		cudaMemcpy(A_GPU, A_CPU, sizeMatrix, cudaMemcpyHostToDevice);
		cudaMemcpy(B_GPU, B_CPU, sizeMatrix, cudaMemcpyHostToDevice);

		// определим число нитей и блоков для работы функции ядра
		dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
		dim3 blocksPerGrid = dim3(N / BLOCK_SIZE, N / BLOCK_SIZE);

		cudaEventRecord(start, 0);														// начало отсчета времени
		matrixMult << <blocksPerGrid, threadsPerBlock >> > (A_GPU, B_GPU, C_GPU, N);	// работа функции ядра
		cudaEventRecord(stop, 0);														// окончание отсчета времени
		cudaEventSynchronize(stop);														// синхронизация

		// подсчет времени работы функции на графическом процессоре
		float KernelTime;
		cudaEventElapsedTime(&KernelTime, start, stop);
		printf("%f KernelTime:  milliseconds\n", KernelTime);

		// расчет ускорения времени
		double S = cpu_time / KernelTime;
		printf("%f Acceleration: \n", S);

		// копирование результирующего массива из памяти графического процессора для последующей проверки
		cudaMemcpy(C_CPU, C_GPU, sizeMatrix, cudaMemcpyDeviceToHost);


		// проверка корректности вычисления
		if (checkMult(C_CPU, C_seq_CPU, N))
			printf("The multiplication results are correct.\n");
		else
			printf("Multiplication results are NOT correct.\n");

		// высвобождение памяти
		cudaFree(A_GPU);
		cudaFree(B_GPU);
		cudaFree(C_GPU);
		free(A_CPU);
		free(B_CPU);
		free(C_CPU);
		free(C_seq_CPU);
	}
	return 0;
}