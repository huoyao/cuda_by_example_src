#include "cuda.h"
#include "..\common\book.h"
#include "..\common\cpu_anim.h"
#include <cuda_runtime.h>
#include"device_launch_parameters.h"

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
  CPUAnimBitmap *bitmap;
  cudaEvent_t start,stop;
  float totaltime;
  float frams;
};

texture<float,2> texconst;
texture<float,2> texin;
texture<float,2> texout;

__global__ void copy_kernel(float *in)
{
  int x=threadIdx.x+blockDim.x*blockIdx.x;
  int y=threadIdx.y+blockIdx.y*blockDim.y;
  int offerset=x+y*gridDim.x*blockDim.x;
  float c=tex2D(texconst,x,y);
  if (c!=0)
  {
    in[offerset]=c;
  }
}

__global__ void blend_kernel(float *out,bool dst)
{
  int x=threadIdx.x+blockDim.x*blockIdx.x;
  int y=threadIdx.y+blockIdx.y*blockDim.y;
  int offerset=x+y*gridDim.x*blockDim.x;
  float a,b_,c,d,e;
  if (dst)
  {
    a=tex2D(texin,x-1,y);
    b_=tex2D(texin,x+1,y);
    c=tex2D(texin,x,y-1);
    d=tex2D(texin,y,y+1);
    e=tex2D(texin,x,y);
  }else{
    a=tex2D(texout,x-1,y);
    b_=tex2D(texout,x+1,y);
    c=tex2D(texout,x,y-1);
    d=tex2D(texout,x,y+1);
    e=tex2D(texout,x,y);
  }
  out[offerset]=e+SPEED*(a+b_+c+d-4*e);
}

void ani_gpu(Datablock *d,int ticks)
{
  HANDLE_ERROR(cudaEventRecord(d->start,0));
  CPUAnimBitmap *bitmap=d->bitmap;
  dim3 blocks(DIM/16,DIM/16);
  dim3 threads(16,16);
  volatile bool dst=true;
  for (int i=0;i!=90;++i)
  {
    //float *in,*out;
    if (dst)
    {
      /*in=d->dev_insrc;
      out=d->dev_outsrc;*/
      copy_kernel<<<blocks,threads>>>(d->dev_insrc);
      blend_kernel<<<blocks,threads>>>(d->dev_outsrc,dst);
    }else{
      /*in=d->dev_outsrc;
      out=d->dev_insrc;*/
      copy_kernel<<<blocks,threads>>>(d->dev_outsrc);
      blend_kernel<<<blocks,threads>>>(d->dev_insrc,dst);
    }
    
    dst=!dst;
  }
  float_to_color<<<blocks,threads>>>(d->dev_bitmap,d->dev_insrc);
  HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(),d->dev_bitmap,bitmap->image_size(),cudaMemcpyDeviceToHost));
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
  HANDLE_ERROR(cudaUnbindTexture(texin));
  HANDLE_ERROR(cudaUnbindTexture(texout));
  HANDLE_ERROR(cudaUnbindTexture(texconst));
  HANDLE_ERROR(cudaFree(d->dev_constsrc));
  HANDLE_ERROR(cudaFree(d->dev_insrc));
  HANDLE_ERROR(cudaFree(d->dev_outsrc));
  HANDLE_ERROR(cudaFree(d->dev_bitmap));
  HANDLE_ERROR(cudaEventDestroy(d->start));
  HANDLE_ERROR(cudaEventDestroy(d->stop));
}

int main(void)
{
  Datablock data;
  CPUAnimBitmap bitmap(DIM,DIM,&data);
  data.bitmap=&bitmap;
  data.totaltime=0;
  data.frams=0;
  HANDLE_ERROR(cudaEventCreate(&data.start,0));
  HANDLE_ERROR(cudaEventCreate(&data.stop,0));
  HANDLE_ERROR(cudaMalloc((void **)&data.dev_bitmap,bitmap.image_size()));
  HANDLE_ERROR(cudaMalloc((void **)&data.dev_insrc,bitmap.image_size()));
  HANDLE_ERROR(cudaMalloc((void **)&data.dev_outsrc,bitmap.image_size()));
  HANDLE_ERROR(cudaMalloc((void **)&data.dev_constsrc,bitmap.image_size()));
  cudaChannelFormatDesc des=cudaCreateChannelDesc<float>();
  HANDLE_ERROR(cudaBindTexture2D(NULL,texout,data.dev_outsrc,des,DIM,DIM,DIM*sizeof(float)));
  HANDLE_ERROR(cudaBindTexture2D(NULL,texin,data.dev_insrc,des,DIM,DIM,DIM*sizeof(float)));
  HANDLE_ERROR(cudaBindTexture2D(NULL,texconst,data.dev_constsrc,des,DIM,DIM,DIM*sizeof(float)));
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
  bitmap.anim_and_exit((void (*)(void *,int))ani_gpu,(void (*)(void *))ani_exit);
  getchar();
}
