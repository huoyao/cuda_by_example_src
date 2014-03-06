#include "cuda.h"
#include "book.h"
#include "cpu_bitmap.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define  SPHERES 20
#define  DIM 1024
#define  INF 2e10f
#define  ran(x) (x*rand()/RAND_MAX)

struct sphere
{
  float x,y,z;
  float r,g,b;
  float radius;
  __device__ float hit(float ox,float oy,float *n)
  {
    float dx=ox-x;
    float dy=oy-y;
    if (dx*dx+dy*dy<radius*radius)
    {
      float dz=sqrtf(radius*radius-dx*dx-dy*dy);
      *n=dz/sqrtf(radius*radius);
      return dz+z;
    }
    return -INF;
  }
};


__constant__ sphere const_s[SPHERES];

__global__ void kernal(sphere *s,unsigned char *bitmap_ptr)
{
  int x=threadIdx.x+blockDim.x*blockIdx.x;
  int y=threadIdx.y+blockIdx.y*blockDim.y;
  int offerset=x+y*gridDim.x*blockDim.x;
  float fx=x-DIM/2;
  float fy=y-DIM/2;
  float maxn=-INF;
  float r=0,g=0,b=0;
  for (int i=0;i!=SPHERES;++i)
  {
    float scal;
    float t=s[i].hit(fx,fy,&scal);
    if (t>maxn)
    {
      r=s[i].r*scal;
      g=s[i].g*scal;
      b=s[i].b*scal;
      maxn=t;
    }
  }
  bitmap_ptr[offerset*4+0]=(int)(r*255);
  bitmap_ptr[offerset*4+1]=(int)(g*255);
  bitmap_ptr[offerset*4+2]=(int)(b*255);
  bitmap_ptr[offerset*4+3]=255;
}

__global__ void kernal_const(unsigned char *bitmap_ptr)
{
  int x=threadIdx.x+blockDim.x*blockIdx.x;
  int y=threadIdx.y+blockIdx.y*blockDim.y;
  int offerset=x+y*gridDim.x*blockDim.x;
  float fx=x-DIM/2;
  float fy=y-DIM/2;
  float maxn=-INF;
  float r=0,g=0,b=0;
  for (int i=0;i!=SPHERES;++i)
  {
    float scal;
    float t=const_s[i].hit(fx,fy,&scal);
    if (t>maxn)
    {
      r=const_s[i].r*scal;
      g=const_s[i].g*scal;
      b=const_s[i].b*scal;
      maxn=t;
    }
  }
  bitmap_ptr[offerset*4+0]=(int)(r*255);
  bitmap_ptr[offerset*4+1]=(int)(g*255);
  bitmap_ptr[offerset*4+2]=(int)(b*255);
  bitmap_ptr[offerset*4+3]=255;
}

//struct DataBlock {
//  unsigned char   *dev_bitmap;
//};

void const_raytrace()
{
  cudaEvent_t start,stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start,0));
  CPUBitmap bitmap(DIM,DIM);
  unsigned char *dev_bitmap;

  HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap,bitmap.image_size()));
  sphere *temp_s=(sphere *)malloc(SPHERES*sizeof(sphere));
  for (int i=0;i!=SPHERES;++i)
  {
    temp_s[i].r=ran(1.0f);
    temp_s[i].g=ran(1.0f);
    temp_s[i].b=ran(1.0f);
    temp_s[i].x=ran(1000.0f)-500.0f;
    temp_s[i].y=ran(1000.0f)-500.0f;
    temp_s[i].z=ran(1000.0f)-500.0f;
    temp_s[i].radius=ran(100.0f)+20.0f;
  }
  HANDLE_ERROR(cudaMemcpyToSymbol(const_s,temp_s,SPHERES*sizeof(sphere)));
  dim3 grids(DIM/16,DIM/16);
  dim3 blocks(16,16);
  kernal_const<<<grids,blocks>>>(dev_bitmap);
  free(temp_s);
  HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(),dev_bitmap,bitmap.image_size(),cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaEventRecord(stop,0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  float elaspedtime;
  HANDLE_ERROR(cudaEventElapsedTime(&elaspedtime,start,stop));
  printf("raytrace time using constant memory:%f ms\n",elaspedtime);
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
  HANDLE_ERROR(cudaFree(dev_bitmap));
  //HANDLE_ERROR(cudaFree(const_s));
  bitmap.display_and_exit();
}

void raytrace()
{
  cudaEvent_t start,stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start,0));

  CPUBitmap bitmap(DIM,DIM);
  unsigned char *dev_bitmap;
  sphere *s;
  HANDLE_ERROR(cudaMalloc((void **)&s,SPHERES*sizeof(sphere)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap,bitmap.image_size()));
  sphere *temp_s=(sphere *)malloc(SPHERES*sizeof(sphere));
  for (int i=0;i!=SPHERES;++i)
  {
    temp_s[i].r=ran(1.0f);
    temp_s[i].b=ran(1.0f);
    temp_s[i].g=ran(1.0f);
    temp_s[i].x=ran(1000.0f)-500.0f;
    temp_s[i].y=ran(1000.0f)-500.0f;
    temp_s[i].z=ran(1000.0f)-500.0f;
    temp_s[i].radius=ran(100.0f)+20.0f;
  }
  HANDLE_ERROR(cudaMemcpy(s,temp_s,SPHERES*sizeof(sphere),cudaMemcpyHostToDevice));
  dim3 grids(DIM/16,DIM/16);
  dim3 blocks(16,16);
  kernal<<<grids,blocks>>>(s,dev_bitmap);
  free(temp_s);
  HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(),dev_bitmap,bitmap.image_size(),cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaEventRecord(stop,0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  float elaspedtime;
  HANDLE_ERROR(cudaEventElapsedTime(&elaspedtime,start,stop));
  printf("raytrace time using no constant memory:%f ms\n",elaspedtime);
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
  HANDLE_ERROR(cudaFree(dev_bitmap));
  HANDLE_ERROR(cudaFree(s));
  bitmap.display_and_exit();
}

int main(void)
{
  raytrace();
  const_raytrace();
  return 0;
}