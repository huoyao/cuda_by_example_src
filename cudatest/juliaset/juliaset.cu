#include <stdio.h>
#include <stdlib.h>
#include "book.h"
#include "cpu_bitmap.h"
#include <cuda_runtime.h>
#include"device_launch_parameters.h"

#define DIM 1000
struct cuComplex 
{
  float r;
  float i;
  __device__ cuComplex(float a,float b):r(a),i(b){}
  __device__ float magnitude2(void){return r*r+i*i;}
  __device__ cuComplex operator*(const cuComplex &a){return cuComplex(r*a.r-i*a.i,i*a.r+r*a.i);}
  __device__ cuComplex operator+(const cuComplex &a){return cuComplex(r+a.r,i+a.i);}
};

__device__ int julia(int x,int y)
{
  const float scale_=1.5;
  float jx=scale_*(float)(DIM/2-x)/(DIM/2);
  float jy=scale_*(float)(DIM/2-y)/(DIM/2);
  cuComplex c(-0.8,0.156);
  cuComplex a(jx,jy);
  for (int i=0;i!=200;++i)
  {
    a=a*a+c;
    if (a.magnitude2()>1000)
    {
      return 0;
    }
  }
  return 1;
}

__global__ void kernel(unsigned char *ptr)
{
  int x=blockIdx.x;
  int y=blockIdx.y;
  int offSet=x+y*gridDim.x;
  int juliavale=julia(x,y);
  ptr[offSet*4+0]=255*juliavale;
  ptr[offSet*4+1]=0;
  ptr[offSet*4+2]=0;
  ptr[offSet*4+3]=255;
}

struct DataBlock {
  unsigned char *device_pt;
};

int main(void)
{
  //DataBlock data;
  //CPUBitmap bitmap(DIM,DIM,&data);
  CPUBitmap bitmap(DIM,DIM);
  unsigned char *dev_bitmap;
  HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap,bitmap.image_size()));
  //data.device_pt=dev_bitmap;
  dim3 grid(DIM,DIM);
  kernel<<<grid,1>>>(dev_bitmap);
  HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(),dev_bitmap,bitmap.image_size(),cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(dev_bitmap));
  bitmap.display_and_exit();
  cudaThreadExit();
}