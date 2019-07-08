/*
@author : Nilanka Manoj
@compile : nvcc meanfilter.cu -o build/meanfilter lib/EasyBMP.cpp
@run : ./build/meanfilter <<imgSize>> <<kernal>>
*/
#include <iostream>
#include <fstream>
#include "lib/EasyBMP.h"
#include <string>
#include <cuda.h>

using namespace std;

int *matrix, *cpuOut, *gpuOut;
int *matrix_d, *out_d;

__global__ void filter_gpu_ker3(int *matrix_in, int *matrix_out, int imgSize)
{

   int i = max(imgSize / blockDim.x + 1, blockIdx.x) * blockDim.x + threadIdx.x;
   matrix_out[i] = (matrix_in[i] + matrix_in[i + 1] + matrix_in[i - 1] +
                    matrix_in[i - imgSize] + matrix_in[i - imgSize + 1] + matrix_in[i - imgSize - 1] +
                    matrix_in[i + imgSize] + matrix_in[i + imgSize + 1] + matrix_in[i + imgSize - 1]) /
                   9;
}

__global__ void filter_gpu_ker5(int *matrix_in, int *matrix_out, int imgSize)
{
   int i = max((imgSize / blockDim.x) * 2 + 1, blockIdx.x) * blockDim.x + threadIdx.x;
   matrix_out[i] = (matrix_in[i] + matrix_in[i + 1] + matrix_in[i - 1] + matrix_in[i + 2] + matrix_in[i - 2] +
                    matrix_in[i - imgSize] + matrix_in[i - imgSize + 1] + matrix_in[i - imgSize - 1] + +matrix_in[i - imgSize + 2] + matrix_in[i - imgSize - 2] +
                    matrix_in[i + imgSize] + matrix_in[i + imgSize + 1] + matrix_in[i + imgSize - 1] + matrix_in[i + imgSize + 2] + matrix_in[i + imgSize - 2] +
                    matrix_in[i - imgSize * 2] + matrix_in[i - imgSize * 2 + 1] + matrix_in[i - imgSize * 2 - 1] + +matrix_in[i - imgSize * 2 + 2] + matrix_in[i - imgSize * 2 - 2] +
                    matrix_in[i + imgSize * 2] + matrix_in[i + imgSize * 2 + 1] + matrix_in[i + imgSize * 2 - 1] + matrix_in[i + imgSize * 2 + 2] + matrix_in[i + imgSize * 2 - 2]) /
                   25;
}

void filter_cpu_ker3(int *matrix_in, int *matrix_out, int imgSize)
{
   for (int i = imgSize; i < imgSize * (imgSize - 1); i++)
   {
      int x = i % imgSize;
      if (x != 0 && x != imgSize - 1)
      {
         matrix_out[i] = (matrix_in[i] + matrix_in[i + 1] + matrix_in[i - 1] +
                          matrix_in[i - imgSize] + matrix_in[i - imgSize + 1] + matrix_in[i - imgSize - 1] +
                          matrix_in[i + imgSize] + matrix_in[i + imgSize + 1] + matrix_in[i + imgSize - 1]) /
                         9;
      }
   }
}
void filter_cpu_ker5(int *matrix_in, int *matrix_out, int imgSize)
{
   for (int i = imgSize * 2; i < imgSize * (imgSize - 2); i++)
   {
      int x = i % imgSize;
      if (x != 0 && x != imgSize - 1 && x != 1 && x != imgSize - 2)
      {
         matrix_out[i] = (matrix_in[i] + matrix_in[i + 1] + matrix_in[i - 1] + matrix_in[i + 2] + matrix_in[i - 2] +
                          matrix_in[i - imgSize] + matrix_in[i - imgSize + 1] + matrix_in[i - imgSize - 1] + +matrix_in[i - imgSize + 2] + matrix_in[i - imgSize - 2] +
                          matrix_in[i + imgSize] + matrix_in[i + imgSize + 1] + matrix_in[i + imgSize - 1] + matrix_in[i + imgSize + 2] + matrix_in[i + imgSize - 2] +
                          matrix_in[i - imgSize * 2] + matrix_in[i - imgSize * 2 + 1] + matrix_in[i - imgSize * 2 - 1] + +matrix_in[i - imgSize * 2 + 2] + matrix_in[i - imgSize * 2 - 2] +
                          matrix_in[i + imgSize * 2] + matrix_in[i + imgSize * 2 + 1] + matrix_in[i + imgSize * 2 - 1] + matrix_in[i + imgSize * 2 + 2] + matrix_in[i + imgSize * 2 - 2]) /
                         25;
      }
   }
}

void img_to_matrix(char *file, int imgSize, int *matrix)
{
   BMP Image;
   Image.ReadFromFile(file);

   for (int i = 0; i < imgSize; i++)
   {
      for (int j = 0; j < imgSize; j++)
      {
         int b = Image(i, j)->Blue;
         matrix[i * imgSize + j] = b;
      }
   }
   cout << "matrix is extracted from :";
   cout << file << endl;
}

void matrix_to_img(char *file, int imgSize, int *matrix)
{
   BMP Image;
   Image.SetSize(imgSize, imgSize);

   for (int i = 0; i < imgSize; i++)
   {
      for (int j = 0; j < imgSize; j++)
      {
         int b = matrix[i * imgSize + j];
         Image(i, j)->Blue = b;
         Image(i, j)->Green = b;
         Image(i, j)->Red = b;
      }
   }
   Image.SetBitDepth(8);
   CreateGrayscaleColorTable(Image);
   Image.WriteToFile(file);
   cout << "matrix is saved to :";
   cout << file << endl;
}

void runRound(int imgSize, char *file, int kernal, char *gpu_file, char *cpu_file)
{
   cout << "======================round starting====================================\n";
   matrix = (int *)malloc((imgSize) * (imgSize) * sizeof(int));
   cpuOut = (int *)malloc((imgSize) * (imgSize) * sizeof(int));
   gpuOut = (int *)malloc((imgSize) * (imgSize) * sizeof(int));

   img_to_matrix(file, imgSize, matrix);

   cudaMalloc((void **)&matrix_d, (imgSize) * (imgSize) * sizeof(int));
   cudaMalloc((void **)&out_d, (imgSize) * (imgSize) * sizeof(int));
   cudaMemcpy(matrix_d, matrix, imgSize * imgSize * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(out_d, matrix, imgSize * imgSize * sizeof(int), cudaMemcpyHostToDevice);
   cpuOut = matrix;

   printf("Doing GPU Filtering\n");
   clock_t start_d = clock();
   if (kernal == 3)
   {
      filter_gpu_ker3<<<imgSize * imgSize / 64, 64>>>(matrix_d, out_d, imgSize);
   }
   else if (kernal == 5)
   {
      filter_gpu_ker5<<<imgSize * imgSize / 64, 64>>>(matrix_d, out_d, imgSize);
   }

   cudaThreadSynchronize();
   clock_t end_d = clock();

   printf("Doing CPU filtering\n");
   clock_t start_h = clock();
   if (kernal == 3)
   {
      filter_cpu_ker3(matrix, cpuOut, imgSize);
   }
   if (kernal == 5)
   {
      filter_cpu_ker5(matrix, cpuOut, imgSize);
   }
   clock_t end_h = clock();

   double time_d = (double)(end_d - start_d) / CLOCKS_PER_SEC;
   double time_h = (double)(end_h - start_h) / CLOCKS_PER_SEC;

   cudaMemcpy(gpuOut, out_d, imgSize * imgSize * sizeof(int), cudaMemcpyDeviceToHost);
   cudaFree(out_d);
   cudaFree(matrix_d);

   matrix_to_img(cpu_file, imgSize, cpuOut);
   matrix_to_img(gpu_file, imgSize, gpuOut);
   printf("image size: %d kernal: %d GPU Time: %f CPU Time: %f\n", imgSize, kernal, time_d, time_h);
}

int main(int argc, char *argv[])
{
   if (argc == 3)
   {
      int imgSize = atoi(argv[1]);
      int kernal = atoi(argv[2]);
      if(imgSize==1280){
         if(kernal==3){
            runRound(1280, "input/img1280.bmp", 3, "output/gpu_1280_3.bmp", "output/cpu_1280_3.bmp");
         }
         else if(kernal==5){
            runRound(1280, "input/img1280.bmp", 5, "output/gpu_1280_5.bmp", "output/cpu_1280_5.bmp");
         }
         else{
            cout<<"invalid kernal size\n";
         }
      }
      else if(imgSize==640){
         if(kernal==3){
            runRound(640, "input/img640.bmp", 3, "output/gpu_640_3.bmp", "output/cpu_640_3.bmp");
         }
         else if(kernal==5){
            runRound(640, "input/img640.bmp", 5, "output/gpu_640_5.bmp", "output/cpu_640_5.bmp");
         }
         else{
            cout<<"invalid kernal size\n";
         }
      }
      else{
         cout<<"invalid image size\n";
      }
         
   }
   else{
      cout<<"invalid number of arguments\n";
   }   
   
   return 0;
}
