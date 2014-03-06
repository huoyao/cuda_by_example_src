#include "..\common\book.h"
#include "..\common\gpu_anim.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define DIM 1024
#define  MIN_TEMP 0.0001F
#define MAX_TEMP 1.0f
#define  SPEED 0.25F

struct Datablock
{
  unsigned char *dev_bitmap;
  float *dev_insrc;
  float *dev_outsrc;
  float *dev_constsrc;
  GPUAnimBitmap *bitmap;
  cudaEvent_t start,stop;
  float totaltime;
  float frams;
};

__global__ void copy_kernel(float *in,const float *const_in)
{
  int x=threadIdx.x+blockDim.x*blockIdx.x;
  int y=threadIdx.y+blockIdx.y*blockDim.y;
  int offerset=x+y*gridDim.x*blockDim.x;
  if (const_in[offerset]!=0)
  {
    in[offerset]=const_in[offerset];
  }
}

__global__ void blend_kernel(const float *in,float *out)
{
  int x=threadIdx.x+blockDim.x*blockIdx.x;
  int y=threadIdx.y+blockIdx.y*blockDim.y;
  int offerset=x+y*gridDim.x*blockDim.x;
  int l=offerset-1;
  int r=offerset+1;
  if (x==0)
  {
    ++l;
  }
  if (x==DIM)
  {
    --r;
  }
  int t=offerset-DIM;
  int b=offerset+DIM;
  if(y==0)
  {
    t+=DIM;
  }
  if (y==DIM-1)
  {
    b-=DIM;
  }
  out[offerset]=in[offerset]+SPEED*(in[l]+in[r]+in[t]+in[b]-4*in[offerset]);
}

void ani_gpu(uchar4 *out,Datablock *d,int ticks)
{
  HANDLE_ERROR(cudaEventRecord(d->start,0));
  GPUAnimBitmap *bitmap=d->bitmap;
  dim3 blocks(DIM/16,DIM/16);
  dim3 threads(16,16);
  for (int i=0;i!=90;++i)
  {
    copy_kernel<<<blocks,threads>>>(d->dev_insrc,d->dev_constsrc);
    blend_kernel<<<blocks,threads>>>(d->dev_insrc,d->dev_outsrc);
    swap(d->dev_insrc,d->dev_outsrc);
  }
  float_to_color<<<blocks,threads>>>(out,d->dev_insrc);
  HANDLE_ERROR(cudaEventRecord(d->stop,0));
  HANDLE_ERROR(cudaEventSynchronize(d->stop));
  float elaspedtime;
  HANDLE_ERROR(cudaEventElapsedTime(&elaspedtime,d->start,d->stop));
  d->totaltime+=elaspedtime;
  ++d->frams;
  printf("averrage time per fram:%f ms\n",d->totaltime/d->frams);
}

void ani_exit(Datablock *d)
{
  HANDLE_ERROR(cudaFree(d->dev_constsrc));
  HANDLE_ERROR(cudaFree(d->dev_insrc));
  HANDLE_ERROR(cudaFree(d->dev_outsrc));
  HANDLE_ERROR(cudaEventDestroy(d->start));
  HANDLE_ERROR(cudaEventDestroy(d->stop));
}

int main(void)
{
  Datablock data;
  GPUAnimBitmap bitmap(DIM,DIM,&data);
  data.bitmap=&bitmap;
  data.totaltime=0;
  data.frams=0;
  HANDLE_ERROR(cudaEventCreate(&data.start,0));
  HANDLE_ERROR(cudaEventCreate(&data.stop,0));
  HANDLE_ERROR(cudaMalloc((void **)&data.dev_bitmap,bitmap.image_size()));
  HANDLE_ERROR(cudaMalloc((void **)&data.dev_insrc,bitmap.image_size()));
  HANDLE_ERROR(cudaMalloc((void **)&data.dev_outsrc,bitmap.image_size()));
  HANDLE_ERROR(cudaMalloc((void **)&data.dev_constsrc,bitmap.image_size()));
  float *temp=(float *)malloc(bitmap.image_size());
  for (int i=0;i!=DIM*DIM;++i)
  {
    temp[i]=0;
    int x=i%DIM;
    int y=i/DIM;
    if (300<x && x<600 && 310<y && y<601)
    {
      temp[i]=MAX_TEMP;
    }
    //some data changes may happen
  }
  temp[DIM*100+100]=MIN_TEMP;
  temp[DIM*700+100]=MIN_TEMP;
  temp[DIM*300+300]=MIN_TEMP;
  temp[DIM*200+700]=MIN_TEMP;
  for (int x=800;x!=900;++x)
  {
    for (int y=400;y!=500;++y)
    {
      temp[x+y*DIM]=MIN_TEMP;
    }
  }
  HANDLE_ERROR(cudaMemcpy(data.dev_constsrc,temp,bitmap.image_size(),cudaMemcpyHostToDevice));
  for (int y=800;y!=DIM;++y)
  {
    for (int x=0;x!=200;++x)
    {
      temp[x+y*DIM]=MIN_TEMP;
    }
  }
  HANDLE_ERROR(cudaMemcpy(data.dev_insrc,temp,bitmap.image_size(),cudaMemcpyHostToDevice));
  free(temp);
  bitmap.anim_and_exit((void (*)(uchar4 *,void *,int))ani_gpu,(void (*)(void *))ani_exit);
  getchar();
}
