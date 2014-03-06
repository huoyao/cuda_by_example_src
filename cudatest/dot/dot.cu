#include "cuda.h"
#include "book.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#define  N 2*1024
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

int main(void)
{
  float *a,*b,*c;
  double sum=0.;
  float *dev_a,*dev_b,*dev_c;
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
  HANDLE_ERROR(cudaMemcpy(dev_a,a,N*sizeof(float),cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b,b,N*sizeof(float),cudaMemcpyHostToDevice));
  dot<<<blockPerGrid,threadPerBlock>>>(dev_a,dev_b,dev_c);
  HANDLE_ERROR(cudaMemcpy(c,dev_c,blockPerGrid*sizeof(float),cudaMemcpyDeviceToHost));
  for (int i=0;i!=blockPerGrid;++i)
  {
    sum+=c[i];
  }
#define sumaryofx(x) (x*(x+1)*(2*x+1)/6)
  printf("does %lf=%lf\n",sum,2*sumaryofx((double)(N-1)));
  double x=2*1024.;
  double  xx=2*x*(x+1)*(2*x+1)/6;
  printf("%lf\n",xx);
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));
  free(a);
  free(b);
  free(c);
  if (getchar()==27)
  {
    exit(0);
  }
  return 0;
}