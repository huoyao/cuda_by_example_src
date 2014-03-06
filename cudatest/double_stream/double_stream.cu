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
  cudaStream_t stream,stream1;
  HANDLE_ERROR(cudaStreamCreate(&stream));
  HANDLE_ERROR(cudaStreamCreate(&stream1));
  int *dev_a,*dev_b,*dev_c;
  int *dev_a1,*dev_b1,*dev_c1;
  int *a,*b,*c;
  HANDLE_ERROR(cudaMalloc((void **)&dev_a,N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b,N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_c,N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_a1,N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b1,N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_c1,N*sizeof(int)));
  HANDLE_ERROR(cudaHostAlloc((void **)&a,FULL_SIZE*sizeof(int),cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void **)&b,FULL_SIZE*sizeof(int),cudaHostAllocDefault));
  HANDLE_ERROR(cudaHostAlloc((void **)&c,FULL_SIZE*sizeof(int),cudaHostAllocDefault));
  for (int i=0;i!=FULL_SIZE;++i)
  {
    a[i]=rand();
    b[i]=rand();
  }
  //first time
  HANDLE_ERROR(cudaEventRecord(start,0));
  for (int i=0;i!=FULL_SIZE;i+=2*N)
  {
    HANDLE_ERROR(cudaMemcpyAsync(dev_a,a+i,N*sizeof(int),cudaMemcpyHostToDevice,stream));
    HANDLE_ERROR(cudaMemcpyAsync(dev_b,b+i,N*sizeof(int),cudaMemcpyHostToDevice,stream));
    kernel<<<N/256,256,0,stream>>>(dev_a,dev_b,dev_c);
    HANDLE_ERROR(cudaMemcpyAsync(c+i,dev_c,N*sizeof(int),cudaMemcpyDeviceToHost,stream));

    HANDLE_ERROR(cudaMemcpyAsync(dev_a1,a+i+N,N*sizeof(int),cudaMemcpyHostToDevice,stream1));
    HANDLE_ERROR(cudaMemcpyAsync(dev_b1,b+i+N,N*sizeof(int),cudaMemcpyHostToDevice,stream1));
    kernel<<<N/256,256,0,stream1>>>(dev_a1,dev_b1,dev_c1);
    HANDLE_ERROR(cudaMemcpyAsync(c+i+N,dev_c1,N*sizeof(int),cudaMemcpyDeviceToHost,stream1));
  }
  float elaspedtime;
  HANDLE_ERROR(cudaEventRecord(stop,0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elaspedtime,start,stop));
  printf("first total time:%.3f ms\n",elaspedtime);
  //second time
  HANDLE_ERROR(cudaEventRecord(start,0));
  for (int i=0;i!=FULL_SIZE;i+=2*N)
  {
    HANDLE_ERROR(cudaMemcpyAsync(dev_a,a+i,N*sizeof(int),cudaMemcpyHostToDevice,stream));
    HANDLE_ERROR(cudaMemcpyAsync(dev_a1,a+i+N,N*sizeof(int),cudaMemcpyHostToDevice,stream1));
    HANDLE_ERROR(cudaMemcpyAsync(dev_b,b+i,N*sizeof(int),cudaMemcpyHostToDevice,stream));
    HANDLE_ERROR(cudaMemcpyAsync(dev_b1,b+i+N,N*sizeof(int),cudaMemcpyHostToDevice,stream1));
    kernel<<<N/256,256,0,stream>>>(dev_a,dev_b,dev_c);
    kernel<<<N/256,256,0,stream1>>>(dev_a1,dev_b1,dev_c1);
    HANDLE_ERROR(cudaMemcpyAsync(c+i,dev_c,N*sizeof(int),cudaMemcpyDeviceToHost,stream));
    HANDLE_ERROR(cudaMemcpyAsync(c+i+N,dev_c1,N*sizeof(int),cudaMemcpyDeviceToHost,stream1));
  }
  HANDLE_ERROR(cudaEventRecord(stop,0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elaspedtime,start,stop));
  printf("second total time:%.3f ms\n",elaspedtime);
  HANDLE_ERROR(cudaStreamSynchronize(stream));
  HANDLE_ERROR(cudaStreamSynchronize(stream1));
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));
  HANDLE_ERROR(cudaFree(dev_a1));
  HANDLE_ERROR(cudaFree(dev_b1));
  HANDLE_ERROR(cudaFree(dev_c1));
  HANDLE_ERROR(cudaFreeHost(a));
  HANDLE_ERROR(cudaFreeHost(b));
  HANDLE_ERROR(cudaFreeHost(c));
  HANDLE_ERROR(cudaStreamDestroy(stream));
  HANDLE_ERROR(cudaStreamDestroy(stream1));
  getchar();
  return 0;
}