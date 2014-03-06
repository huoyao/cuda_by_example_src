#include "..\common\book.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define  N 1024*1024
#define FULL_SIZE (N*20)

__global__ void kernel(int *a,int *b,int *c)
{
  int i=threadIdx.x+blockIdx.x*blockDim.x;
  if (i<N)
  {
    int id1=(i+1)%256;
    int id2=(i-1)%256;
    a[i]=(a[id1]+a[id2]+a[i])/3.0f;
    b[i]=(b[id1]+b[id2]+b[i])/3.0f;
    c[i]=(a[i]+b[i])/2.0f;
  }
}

int main(void)
{
  cudaDeviceProp prop;
  int dev;
  HANDLE_ERROR(cudaGetDevice(&dev));
  HANDLE_ERROR(cudaGetDeviceProperties(&prop,dev));
  if (!prop.deviceOverlap)
  {
    printf("device can`t overflap\n");
    return 0;
  }
  cudaEvent_t start,stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start,0));
  cudaStream_t stream;
  HANDLE_ERROR(cudaStreamCreate(&stream));
  int *dev_a,*dev_b,*dev_c;
  int *a,*b,*c;
  HANDLE_ERROR(cudaMalloc((void **)&dev_a,N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b,N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_c,N*sizeof(int)));
  HANDLE_ERROR(cudaHostAlloc((void **)&a,FULL_SIZE*sizeof(int),cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void **)&b,FULL_SIZE*sizeof(int),cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void **)&c,FULL_SIZE*sizeof(int),cudaHostAllocDefault));
  for (int i=0;i!=FULL_SIZE;++i)
  {
    a[i]=rand();
    b[i]=rand();
  }
  for (int i=0;i!=FULL_SIZE;i+=N)
  {
    HANDLE_ERROR(cudaMemcpyAsync(dev_a,a+i,N*sizeof(int),cudaMemcpyHostToDevice,stream));
    HANDLE_ERROR(cudaMemcpyAsync(dev_b,b+i,N*sizeof(int),cudaMemcpyHostToDevice,stream));
    kernel<<<N/256,256,0,stream>>>(dev_a,dev_b,dev_c);
    HANDLE_ERROR(cudaMemcpyAsync(c+i,dev_c,N*sizeof(int),cudaMemcpyDeviceToHost,stream));
  }
  HANDLE_ERROR(cudaStreamSynchronize(stream));
  float elaspedtime;
  HANDLE_ERROR(cudaEventRecord(stop,0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elaspedtime,start,stop));
  printf("total time:%.3f ms\n",elaspedtime);
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));
  HANDLE_ERROR(cudaFreeHost(a));
  HANDLE_ERROR(cudaFreeHost(b));
  HANDLE_ERROR(cudaFreeHost(c));
  HANDLE_ERROR(cudaStreamDestroy(stream));
  getchar();
  return 0;
}