#include "cuda.h"
#include "GL\glut.h"
#include "cuda_gl_interop.h"
#include "..\common\book.h"
#include "..\common\cpu_bitmap.h"
#include <cuda_runtime.h>
#include"device_launch_parameters.h"

#define  DIM 512
//declare functions
PFNGLBINDBUFFERARBPROC    glBindBuffer     = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers  = NULL;
PFNGLGENBUFFERSARBPROC    glGenBuffers     = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData     = NULL;

GLuint bufferobj;
cudaGraphicsResource *resource;

__global__ void kernel(uchar4 * d)
{
  int x=threadIdx.x+blockDim.x*blockIdx.x;
  int y=threadIdx.y+blockIdx.y*blockDim.y;
  int offerset=x+y*gridDim.x*blockDim.x;
  float fx=x/(float)DIM-0.5f;
  float fy=y/(float)DIM-0.5f;
  float dst=sqrtf(fx*fx+fy*fy);
  unsigned char grey=128+127.0f*sin(abs(fx*100)-abs(fy*100));
  d[offerset].x=0;
  d[offerset].y=grey;
  d[offerset].z=0;
  d[offerset].w=255;
}

static void draw_func()
{
  glDrawPixels(DIM,DIM,GL_BGRA,GL_UNSIGNED_BYTE,0);
  glutSwapBuffers();
}

static void key_func(unsigned char t,int x,int y)
{
  switch(t)
  {
  case 27:
    HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB,0);
    glDeleteBuffers(1,&bufferobj);
    exit(0);
  }
}

int main(int argc,char **argv)
{
  cudaDeviceProp prop;
  memset(&prop,0,sizeof(cudaDeviceProp));
  prop.major=1.0;
  prop.minor=0.;
  int dev;
  HANDLE_ERROR(cudaChooseDevice(&dev,&prop));
  HANDLE_ERROR(cudaGLSetGLDevice(dev));
  //called before any other opengl functions
  glutInit(&argc,argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowSize(DIM,DIM);
  glutCreateWindow("bitmap");
  //get the address of the functions
  glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
  glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
  glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
  glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");
  //the functions is the extension of opengl that put the data into video memory directly
  //VBO 
  glGenBuffers(1,&bufferobj);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB,bufferobj);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,DIM*DIM*4,NULL,GL_DYNAMIC_DRAW_ARB);

  HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource,bufferobj,cudaGraphicsMapFlagsNone));
  uchar4 *dev_ptr;
  size_t size;
  HANDLE_ERROR(cudaGraphicsMapResources(1,&resource,NULL));
  HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void **)&dev_ptr,&size,resource));
  dim3 grids(DIM/16,DIM/16);
  dim3 blocks(16,16);
  kernel<<<grids,blocks>>>(dev_ptr);
  HANDLE_ERROR(cudaGraphicsUnmapResources(1,&resource,NULL));
  glutKeyboardFunc(key_func);
  glutDisplayFunc(draw_func);
  glutMainLoop();
}