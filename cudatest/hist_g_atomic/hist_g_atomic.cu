#include "cuda.h"
#include "..\common\book.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define  SIZE (100*1024*1024)

__global__ void hist_kernel(unsigned char *dev_buf,long size_,unsigned int *hist_count)
{
  __shared__ unsigned int temp[256];
  temp[threadIdx.x]=0;
  long i=threadIdx.x+blockIdx.x*blockDim.x;
  int offerset=blockDim.x*gridDim.x;
  __syncthreads();
  while (i<size_)
  {
    atomicAdd(&temp[dev_buf[i]],1);
    i+=offerset;
  }
  __syncthreads();
  atomicAdd(&hist_count[threadIdx.x],temp[threadIdx.x]);
}

int main(void)
{
  unsigned char *buffer=(unsigned char *)big_random_block(SIZE);
  cudaEvent_t start,stop;
  HANDLE_ERROR(cudaEventCreate(&start,0));
  HANDLE_ERROR(cudaEventCreate(&stop,0));
  HANDLE_ERROR(cudaEventRecord(start,0));
  unsigned char *dev_buf;
  unsigned int *dev_hist_count;
  HANDLE_ERROR(cudaMalloc((void **)&dev_buf,SIZE));
  HANDLE_ERROR(cudaMemcpy(dev_buf,buffer,SIZE,cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void **)&dev_hist_count,256*sizeof(int)));
  HANDLE_ERROR(cudaMemset(dev_hist_count,0,256*sizeof(int)));
  cudaDeviceProp prop;
  HANDLE_ERROR(cudaGetDeviceProperties(&prop,0));
  int blocks=prop.multiProcessorCount;
  //blocks*2 may make the gpu run in best status 
  hist_kernel<<<blocks*2,256>>>(dev_buf,SIZE,dev_hist_count);
  unsigned int h_hist_count[256];
  HANDLE_ERROR(cudaMemcpy(h_hist_count,dev_hist_count,256*sizeof(int),cudaMemcpyDeviceToHost));
  float elaspedtime;
  HANDLE_ERROR(cudaEventRecord(stop,0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elaspedtime,start,stop));
  printf("total time:%.3f ms\n",elaspedtime);

  long histcount=0;
  for (int i=0;i!=256;++i)
  {
    histcount+=h_hist_count[i];
    //printf("%d ",h_hist_count[i]);
  }
  printf("sum of hist is:%ld \n",histcount);
  for (int i=0;i!=SIZE;++i)
  {
    h_hist_count[buffer[i]]--;
  }
  for (int i=0;i!=256;++i)
  {
    if (h_hist_count[i]!=0)
    {
      printf("failed\n");
      break;
    }
  }
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
  HANDLE_ERROR(cudaFree(dev_buf));
  HANDLE_ERROR(cudaFree(dev_hist_count));
  free(buffer);
  getchar();
  return 0;
}