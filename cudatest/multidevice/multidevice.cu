#include "cuda.h"
#include "..\common\book.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#define  N 2*1024
#define imin(a,b) (a<b?a:b)

const int threadPerBlock=256;
const int blockPerGrid=imin(32,(N+threadPerBlock-1)/threadPerBlock);

struct datastruct
{
  int deviceid;
  int size;
  float *a;
  float *b;
  float returnvalue;
};
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

void *rutine(void *pvoiddata)
{
  datastruct *data=(datastruct *)pvoiddata;
  HANDLE_ERROR(cudaSetDevice(data->deviceid));
  int size=data->size;
  float *a,*b,*c;
  double sum=0.;
  float *dev_a,*dev_b,*dev_c;
  cudaEvent_t start,stop;
  float elaspedtime;
  HANDLE_ERROR(cudaEventCreate(&start,0));
  HANDLE_ERROR(cudaEventCreate(&stop,0));
  a=data->a;
  b=data->b;
  c=(float *)malloc(blockPerGrid*sizeof(float));
  HANDLE_ERROR(cudaMalloc((void **)&dev_a,size*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b,size*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_c,blockPerGrid*sizeof(float)));
  HANDLE_ERROR(cudaEventRecord(start,0));
  HANDLE_ERROR(cudaMemcpy(dev_a,a,size*sizeof(float),cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b,b,size*sizeof(float),cudaMemcpyHostToDevice));
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
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));
  free(a);
  free(b);
  free(c);
  data->returnvalue=sum;
  return 0;
}

int main(void)
{
  int dev_count;
  HANDLE_ERROR(cudaGetDeviceCount(&dev_count));
  if (dev_count<2)
  {
    printf("no more gpu,gpu num£º%d\n",dev_count);
    getchar();
    return 0;
  }
  float *a=(float *)malloc(N*sizeof(float));
  HANDLE_NULL(a);
  float *b=(float *)malloc(N*sizeof(float));
  HANDLE_NULL(b);
  for (int i=0;i!=N;++i)
  {
    a[i]=(float)i;
    b[i]=(float)i*2;
  }
  datastruct data[2];
  data[0].deviceid=0;
  data[0].a=a;
  data[0].b=b;
  data[0].size=N/2;
  data[1].deviceid=1;
  data[1].a=a+N/2;
  data[1].b=b+N/2;
  data[1].size=N/2;
  CUTThread threadx=start_thread((CUT_THREADROUTINE)rutine,&data[0]);
  rutine(&data[1]);
  end_thread(threadx);
  free(a);
  free(b);
  printf("value:%f\n",data[0].returnvalue+data[1].returnvalue);
  cudaThreadExit();
  if (getchar()==27)
  {
    exit(0);
  }
  return 0;
}