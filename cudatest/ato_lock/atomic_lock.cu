#include "cuda.h"
#include "..\common\book.h"
#include "..\common\lock.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#define  N 2*1024
#define imin(a,b) (a<b?a:b)

const int threadPerBlock=256;
const int blockPerGrid=imin(32,(N+threadPerBlock-1)/threadPerBlock);

//struct lock
//{
//  int *mutex;
//  lock(void)
//  {
//    int state=0;
//    HANDLE_ERROR(cudaMalloc((void **)&mutex,sizeof(int)));
//    HANDLE_ERROR(cudaMemcpy(mutex,&state,sizeof(int),cudaMemcpyHostToDevice));
//  }
//  ~lock(){cudaFree(mutex)}
//  __device__ void locked(void)
//  {
//    while(atomicCAS(mutex,0,1)!=1);
//  }
//  __device__ void unlock(void)
//  {
//    atomicExch(mutex,0);
//  }
//};

__global__ void dot(Lock lock,float *a,float *b,float *c)
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
    lock.lock();
    *c+=cahe[0];
    lock.unlock();
  }
}

int main(void)
{
  float *a,*b,c=0;
  float *dev_a,*dev_b,*dev_c;
  a=(float *)malloc(N*sizeof(float));
  b=(float *)malloc(N*sizeof(float));
  HANDLE_ERROR(cudaMalloc((void **)&dev_a,N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b,N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_c,sizeof(float)));
  for (int i=0;i!=N;++i)
  {
    a[i]=(float)i;
    b[i]=(float)i*2;
  }
  HANDLE_ERROR(cudaMemcpy(dev_a,a,N*sizeof(float),cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b,b,N*sizeof(float),cudaMemcpyHostToDevice));
  Lock lock;
  dot<<<blockPerGrid,threadPerBlock>>>(lock,dev_a,dev_b,dev_c);
  HANDLE_ERROR(cudaMemcpy(&c,dev_c,sizeof(float),cudaMemcpyDeviceToHost));
#define sumaryofx(x) (x*(x+1)*(2*x+1)/6)
  printf("does %lf=%lf\n",c,2*sumaryofx((double)(N-1)));
  double x=2*1024.;
  double  xx=2*x*(x+1)*(2*x+1)/6;
  printf("%lf\n",xx);
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));
  free(a);
  free(b);
  if (getchar()==27)
  {
    exit(0);
  }
  return 0;
}