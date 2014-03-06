#include "..\common\book.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define  SIZE (100*1024*1024)

float cpumalloc(int size,bool up)
{
  cudaEvent_t start,stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  int *a,*dev_a;
  a=(int *)malloc(SIZE*sizeof(*a));
  HANDLE_NULL(a);
  HANDLE_ERROR(cudaMalloc((void **)&dev_a,size*sizeof(*dev_a)));
  HANDLE_ERROR(cudaEventRecord(start,0));
  for (int i=0;i!=100;++i)
  {
    if (up)
    {
      HANDLE_ERROR(cudaMemcpy(a,dev_a,size*sizeof(*a),cudaMemcpyDeviceToHost));
    } 
    else
    {
      HANDLE_ERROR(cudaMemcpy(dev_a,a,size*sizeof(*a),cudaMemcpyHostToDevice));
    }
  }
  HANDLE_ERROR(cudaEventRecord(stop,0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  float elaspedtime;
  HANDLE_ERROR(cudaEventElapsedTime(&elaspedtime,start,stop));
  HANDLE_ERROR(cudaFree(dev_a));
  free(a);
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
  return elaspedtime;
}

float cudahostmalloc(int size,bool up)
{
  cudaEvent_t start,stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  int *a,*dev_a;
  HANDLE_ERROR(cudaMalloc((void **)&dev_a,size*sizeof(*dev_a)));
  HANDLE_ERROR(cudaHostAlloc((void **)&a,size*sizeof(*a),cudaHostAllocDefault));
  HANDLE_ERROR(cudaEventRecord(start,0));
  for (int i=0;i!=100;++i)
  {
    if (up)
    {
      HANDLE_ERROR(cudaMemcpy(a,dev_a,size*sizeof(*a),cudaMemcpyDeviceToHost));
    } 
    else
    {
      HANDLE_ERROR(cudaMemcpy(dev_a,a,size*sizeof(*a),cudaMemcpyHostToDevice));
    }
  }
  HANDLE_ERROR(cudaEventRecord(stop,0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  float elaspedtime;
  HANDLE_ERROR(cudaEventElapsedTime(&elaspedtime,start,stop));
  HANDLE_ERROR(cudaFreeHost(a));
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
  return elaspedtime;
}

int main(void)
{
  float elaspedtime;
  float MB=(float)100*sizeof(int)*SIZE/1024./1024.;
  bool up=true;
  elaspedtime=cpumalloc(SIZE,up);
  printf("device to host: \nrun time using malloc:%f ms\n",elaspedtime);
  printf("%f MB per second.\n",MB/elaspedtime/1000.);
  elaspedtime=cudahostmalloc(SIZE,up);
  printf("run time using hostmalloc:%f ms\n",elaspedtime);
  printf("%f MB per second.\n",MB/elaspedtime/1000.);
  up=false;
  elaspedtime=cpumalloc(SIZE,up);
  printf("\n host to device: \nrun time using malloc:%f ms\n",elaspedtime);
  printf("%f MB per second.\n",MB/elaspedtime/1000.);
  elaspedtime=cudahostmalloc(SIZE,up);
  printf("run time using hostmalloc:%f ms\n",elaspedtime);
  printf("%f MB per second.\n",MB/elaspedtime/1000.);
  getchar();
  return 0;
}