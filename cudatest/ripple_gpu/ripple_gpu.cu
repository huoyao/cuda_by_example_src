#include "..\common\book.h"
#include "..\common\gpu_anim.h"
#include <cuda_runtime.h>
#include"device_launch_parameters.h"

#define DIM 1024

__global__ void kernal(uchar4 *d,int ticks)
{
  int x=threadIdx.x+blockDim.x*blockIdx.x;
  int y=threadIdx.y+blockIdx.y*blockDim.y;
  int offerset=x+y*gridDim.x*blockDim.x;
  float fx=x-DIM/2;
  float fy=y-DIM/2;
  float dst=sqrtf(fx*fx+fy*fy);
  unsigned char grey=(unsigned char)(128+127.0f*cos(dst/10.0f-ticks/7.0f)/(dst/10.0f+1.0f));
  d[offerset].x=grey;
  d[offerset].y=grey;
  d[offerset].z=grey;
  d[offerset].w=255;
}

void generate_frame(uchar4 *d,void *,int ticks)
{
  dim3 blocks(DIM/16,DIM/16);
  dim3 thredads(16,16);
  kernal<<<blocks,thredads>>>(d,ticks);
}

int main(void)
{
  GPUAnimBitmap ani_map(DIM,DIM,NULL);
  ani_map.anim_and_exit((void(*)(uchar4 *,void *,int))generate_frame,NULL);
}