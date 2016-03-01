#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include"device_launch_parameters.h"

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
#define getLastCudaError(msg)  __getLastCudaError (msg, __FILE__, __LINE__)
#define N (55*1024)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#undef  assert
#define assert(arg)
#endif

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions
// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
  if(cudaSuccess != err)
  {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
    return ;
  }
}
// This will output the proper error string when calling cudaGetLastError
inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
      file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
    return ;
  }
}
// end of CUDA Helper Functions

__global__ void VecAdd(float *a,float *b,float *c)
{
  long long i=threadIdx.x+blockIdx.x*blockDim.x;
  while(i<N)
  {
    a[i]=b[i]+c[i];
    //a[i]/=2.0;
    //a[i]/=3.0;
    i+=blockDim.x+gridDim.x;
  }
  //printf("blockdim:%d\n",blockDim.x);
  assert(blockDim.x);
}

__global__ void add(int a,int b ,int *c)
{
  *c=a+b;
}

int main(){

  cudaSetDevice(0);
  cudaDeviceSynchronize();
  cudaThreadSynchronize();
  float A[N],B[N],C[N];
  for(long long i=0;i!=N;++i)
  {
      B[i]=i;
      C[i]=i;
  }
  float *dec_a,*dec_b,*dec_c;
  checkCudaErrors( cudaMalloc((void**) &dec_a, sizeof(int)*N));
  checkCudaErrors( cudaMalloc((void**) &dec_b, sizeof(int)*N));
  checkCudaErrors( cudaMalloc((void**) &dec_c, sizeof(int)*N));
  checkCudaErrors( cudaMemcpy(dec_b,B,sizeof(int)*N,cudaMemcpyHostToDevice));
  checkCudaErrors( cudaMemcpy(dec_c,C,sizeof(int)*N,cudaMemcpyHostToDevice));
  VecAdd<<<256,128>>>(dec_a,dec_b,dec_c);
  checkCudaErrors( cudaMemcpy(A,dec_a,sizeof(int)*N,cudaMemcpyDeviceToHost));
  cudaFree(dec_a);
  cudaFree(dec_b);
  cudaFree(dec_c);
  bool suc=true;
  for (long long i=0;i!=N;++i)
  {
    if (A[i]!=B[i]+C[i])
    {
      suc=false;
    }
  }
  if (suc)
  {
    printf("we did it\n");
  } 
  else
  {
    printf("we fail\n");
  }
  //matAdd<<<1,dimBlock>>>(A,B,C);
  int c;
  int *resultc;
  checkCudaErrors( cudaMalloc((void**) &resultc, sizeof(int)));
  add<<<1,1>>>(2,7,resultc);
  checkCudaErrors( cudaMemcpy(&c,resultc,sizeof(int),cudaMemcpyDeviceToHost));
  printf("%d\n",c);
  cudaFree(resultc);
  getchar();
  cudaThreadExit();
  return 0;
}