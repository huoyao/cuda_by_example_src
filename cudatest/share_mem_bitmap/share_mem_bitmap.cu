#include "book.h"
#include "cpu_bitmap.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#define DIM 1024
#define  PI 3.1415926535897932f
__global__ void kernale(unsigned char *pt)
{
  int x=threadIdx.x+blockIdx.x*blockDim.x;
  int y=threadIdx.y+blockIdx.y*blockDim.y;
  int offerset=x+y*gridDim.x*blockDim.x;
  __shared__ float colormem[16][16];
  const float peri=128.;
  colormem[threadIdx.x][threadIdx.y]=255*(sinf(x*2*PI/peri)+1.0f)*(sinf(y*2*PI/peri)+1.0f)*4.0f;
  __syncthreads();
  pt[offerset*4+0]=0;
  pt[offerset*4+1]=colormem[15-threadIdx.x][15-threadIdx.y];
  pt[offerset*4+2]=0;
  pt[offerset*4+3]=255;
}

int main(void)
{
  CPUBitmap bitmap(DIM,DIM);
  unsigned char *dec_bitmap;
  dim3 grids(DIM/16,DIM/16);
  dim3 threads(16,16);
  HANDLE_ERROR(cudaMalloc((void **)&dec_bitmap,bitmap.image_size()));
  kernale<<<grids,threads>>>(dec_bitmap);
  HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(),dec_bitmap,bitmap.image_size(),cudaMemcpyDeviceToHost));
  bitmap.display_and_exit();
  HANDLE_ERROR(cudaFree(dec_bitmap));
  if (getchar()==27)
  {
    exit(0);
  }
  return 0;
}