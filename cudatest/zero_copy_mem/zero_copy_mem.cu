#include "cuda.h"
#include "..\common\book.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#define  N 200*1024
#define imin(a,b) (a<b?a:b)

const int threadPerBlock=256;
const int blockPerGrid=imin(32,(N+threadPerBlock-1)/threadPerBlock);

__global__ void dot(float *a,float *b,float *c)
{
  __shared__ float cahe[threadPerBlock];
  int tid=threadIdx.x+blockDim.x*blockIdx.x;
  int caheidex=threadIdx.x;
  float temp=0;
  while(tid<N)
  {
    temp+=a[tid]*b[tid];
    tid+=blockDim.x*gridDim.x;
  }
  cahe[caheidex]=temp;
  __syncthreads();
  int i=blockDim.x/2;
  while(i!=0)
  {
    if (caheidex<i)
    {
      cahe[caheidex]+=cahe[caheidex+i];
    }
    __syncthreads();
    i/=2;
  }
  if (caheidex==0)
  {
    c[blockIdx.x]=cahe[0];
  }
}

void host_malloc()
{
  float *a,*b,*c;
  double sum=0.;
  float *dev_a,*dev_b,*dev_c;
  cudaEvent_t start,stop;
  float elaspedtime;
  HANDLE_ERROR(cudaEventCreate(&start,0));
  HANDLE_ERROR(cudaEventCreate(&stop,0));
  a=(float *)malloc(N*sizeof(float));
  b=(float *)malloc(N*sizeof(float));
  c=(float *)malloc(blockPerGrid*sizeof(float));
  HANDLE_ERROR(cudaMalloc((void **)&dev_a,N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b,N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_c,blockPerGrid*sizeof(float)));
  for (int i=0;i!=N;++i)
  {
    a[i]=(float)i;
    b[i]=(float)i*2;
  }
  HANDLE_ERROR(cudaEventRecord(start,0));
  HANDLE_ERROR(cudaMemcpy(dev_a,a,N*sizeof(float),cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b,b,N*sizeof(float),cudaMemcpyHostToDevice));
  dot<<<blockPerGrid,threadPerBlock>>>(dev_a,dev_b,dev_c);
  HANDLE_ERROR(cudaMemcpy(c,dev_c,blockPerGrid*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaEventRecord(stop,0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elaspedtime,start,stop));
  printf("host malloc total time:%.3f ms\n",elaspedtime);
  for (int i=0;i!=blockPerGrid;++i)
  {
    sum+=c[i];
  }
#define sumaryofx(x) (x*(x+1)*(2*x+1)/6)
  printf("does %.3lf=%.3lf\n",sum,2*sumaryofx((double)(N-1)));
  double x=2*1024.;
  double  xx=2*x*(x+1)*(2*x+1)/6;
  printf("%.3lf\n",xx);
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));
  free(a);
  free(b);
  free(c);
}

void zero_copy()
{
  float *a,*b,*c;
  double sum=0.;
  float *dev_a,*dev_b,*dev_c;
  cudaEvent_t start,stop;
  float elaspedtime;
  HANDLE_ERROR(cudaEventCreate(&start,0));
  HANDLE_ERROR(cudaEventCreate(&stop,0));
  HANDLE_ERROR(cudaHostAlloc(&a,N*sizeof(float),cudaHostAllocMapped|cudaHostAllocWriteCombined));
  HANDLE_ERROR(cudaHostAlloc(&b,N*sizeof(float),cudaHostAllocMapped|cudaHostAllocWriteCombined));
  HANDLE_ERROR(cudaHostAlloc(&c,blockPerGrid*sizeof(float),cudaHostAllocMapped|cudaHostAllocWriteCombined));
  for (int i=0;i!=N;++i)
  {
    a[i]=(float)i;
    b[i]=(float)i*2;
  }
  HANDLE_ERROR(cudaHostGetDevicePointer(&dev_a,a,0));
  HANDLE_ERROR(cudaHostGetDevicePointer(&dev_b,b,0));
  HANDLE_ERROR(cudaHostGetDevicePointer(&dev_c,c,0));
  HANDLE_ERROR(cudaEventRecord(start,0));
  dot<<<blockPerGrid,threadPerBlock>>>(dev_a,dev_b,dev_c);
  HANDLE_ERROR(cudaThreadSynchronize());
  HANDLE_ERROR(cudaEventRecord(stop,0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elaspedtime,start,stop));
  printf("\nzero copy total time:%.3f ms\n",elaspedtime);
  for (int i=0;i!=blockPerGrid;++i)
  {
    sum+=c[i];
  }
#define sumaryofx(x) (x*(x+1)*(2*x+1)/6)
  printf("does %.3lf=%.3lf\n",sum,2*sumaryofx((double)(N-1)));
  double x=2*1024.;
  double  xx=2*x*(x+1)*(2*x+1)/6;
  printf("%.3lf\n",xx);
  HANDLE_ERROR(cudaFreeHost(a));
  HANDLE_ERROR(cudaFreeHost(b));
  HANDLE_ERROR(cudaFreeHost(c));
}

int main(void)
{
  cudaDeviceProp prop;
  int dev;
  HANDLE_ERROR(cudaGetDevice(&dev));
  HANDLE_ERROR(cudaGetDeviceProperties(&prop,dev));
  if (prop.canMapHostMemory!=1)
  {
    printf("device can`t map memory\n");
    getchar();
    return 0;
  }
  host_malloc();
  zero_copy();
  HANDLE_ERROR(cudaThreadExit());
  if (getchar()==27)
  {
    exit(0);
  }
  return 0;
}