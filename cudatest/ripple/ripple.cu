#include "book.h"
#include "cpu_anim.h"
#include <cuda_runtime.h>
#include"device_launch_parameters.h"

#define DIM 1024

struct DataBlock
{
  unsigned char *dev_animap;
  CPUAnimBitmap *animap;
};

__global__ void kernal(unsigned char *d,int ticks)
{
  int x=threadIdx.x+blockDim.x*blockIdx.x;
  int y=threadIdx.y+blockIdx.y*blockDim.y;
  int offerset=x+y*gridDim.x*blockDim.x;
  float fx=x-DIM/2;
  float fy=y-DIM/2;
  float dst=sqrtf(fx*fx+fy*fy);
  unsigned char grey=(unsigned char)(128+127.0f*cos(dst/10.0f-ticks/7.0f)/(dst/10.0f+1.0f));
  d[offerset*4+0]=grey;
  d[offerset*4+1]=grey;
  d[offerset*4+2]=grey;
  d[offerset*4+3]=255;
}

void cleanup(DataBlock *d)
{
  HANDLE_ERROR(cudaFree(d->dev_animap));
}

void generate_frame(DataBlock *d,int ticks)
{
  dim3 blocks(DIM/16,DIM/16);
  dim3 thredads(16,16);
  kernal<<<blocks,thredads>>>(d->dev_animap,ticks);
  HANDLE_ERROR(cudaMemcpy(d->animap->get_ptr(),d->dev_animap,d->animap->image_size(),cudaMemcpyDeviceToHost));
}

int main(void)
{
  DataBlock data;
  CPUAnimBitmap ani_map(DIM,DIM,&data);
  HANDLE_ERROR(cudaMalloc((void**)&data.dev_animap,ani_map.image_size()));
  data.animap=&ani_map;
  ani_map.anim_and_exit((void(*)(void*,int))generate_frame,(void(*)(void*))cleanup);
  getchar();
  return 0;
}