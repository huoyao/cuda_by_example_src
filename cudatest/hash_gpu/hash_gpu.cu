#include "cuda.h"
#include "..\common\book.h"
#include "..\common\lock.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#define  SISE (100*1024*1024)
#define ELEMNTS (SISE/sizeof(unsigned int))
#define ENTRIES 1024

struct Entry
{
  unsigned int key;
  void *value;
  Entry *next;
};

struct Table
{
  size_t count;
  Entry  **entries;
  Entry *pool;
};

void init_table(Table &table,size_t entr,size_t elem)
{
  table.count=entr;
  //notes: callo(size_t nelem, size_t elsize) alocate nelem blocks memory which size is elsize,make the initial
  //value to be 0
  HANDLE_ERROR(cudaMalloc((void **)&table.entries,entr*sizeof(Entry*)));
  HANDLE_ERROR(cudaMemset(table.entries,0,entr*sizeof(Entry *)));
  HANDLE_ERROR(cudaMalloc((void **)&table.pool,elem*sizeof(Entry)));
}

__device__ __host__ size_t hash(unsigned int key,size_t count)
{
  return key%count;
}

__global__ void add_elem(Table table,unsigned int *key,void **value,Lock *lock)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  while (tid<ELEMNTS)
  {
    unsigned int key_=key[tid];
    size_t hashvalue=hash(key_,table.count);
    for (int i=0;i!=32;++i)
    {
      if (tid%32==i)
      {
        Entry *newelem=&(table.pool[tid]);
        newelem->key=key_;
        newelem->value=value[tid];
        lock[hashvalue].lock();
        newelem->next=table.entries[hashvalue];
        table.entries[hashvalue]=newelem;
        lock[hashvalue].unlock();
      }
    }
    tid+=stride;
  }
}

void freettable(Table &table)
{
  HANDLE_ERROR(cudaFree(table.entries));
  HANDLE_ERROR(cudaFree(table.pool));
}

void copytohost_table(const Table &dev_t,Table &host_t)
{
  host_t.count=dev_t.count;
  host_t.entries=(Entry **)calloc(dev_t.count,sizeof(Entry*));
  host_t.pool=(Entry *)malloc(ELEMNTS*sizeof(Entry));
  HANDLE_ERROR( cudaMemcpy( host_t.entries, dev_t.entries,dev_t.count * sizeof(Entry*),cudaMemcpyDeviceToHost ) );
  HANDLE_ERROR( cudaMemcpy( host_t.pool, dev_t.pool,ELEMNTS * sizeof( Entry ),cudaMemcpyDeviceToHost ) );
  for (int i=0;i<dev_t.count;++i)
  {
    if (host_t.entries[i]!=NULL)
    {
      host_t.entries[i]=(Entry *)((size_t)host_t.pool+(size_t)host_t.entries[i]-(size_t)dev_t.pool);
    }
  }
  for (int i=0;i!=ELEMNTS;++i)
  {
    if (host_t.pool[i].next!=NULL)
    {
      host_t.pool[i].next=(Entry *)((size_t)host_t.pool+(size_t)host_t.pool[i].next-(size_t)dev_t.pool);
    }
  }
}

void verifytable(const Table &dev_table)
{
  Table table;
  copytohost_table(dev_table,table);
  int cnt=0;
  for (size_t i=0;i!=table.count;++i)
  {
    Entry *elm=table.entries[i];
    while (elm!=NULL)
    {
      ++cnt;
      if (hash(elm->key,table.count)!=i)
      {
        printf("%d hash to %ld,but located in  %ld\n",elm->key,hash(elm->key,table.count),i);
      }
      elm=elm->next;
    }
  }
  if (cnt!=ELEMNTS)
  {
    printf("%d was found ,but real num is %d\n",cnt,ELEMNTS);
  }else{
    printf("%d elemnts was all found.sucucess!\n",cnt);
  }
  free(table.entries);
  free(table.pool);
}

int main(void)
{
  unsigned int *buff=(unsigned int*)big_random_block(SISE);
  unsigned int *dev_key;
  void **dev_value;
  HANDLE_ERROR(cudaMalloc((void **)&dev_key,SISE));
  HANDLE_ERROR(cudaMalloc((void **)&dev_value,SISE));
  HANDLE_ERROR(cudaMemcpy(dev_key,buff,SISE,cudaMemcpyHostToDevice));
  Lock lock[ENTRIES];
  Lock *dev_lock;
  HANDLE_ERROR(cudaMalloc((void **)&dev_lock,ENTRIES*sizeof(Lock)));
  HANDLE_ERROR(cudaMemcpy(dev_lock,lock,ENTRIES*sizeof(Lock),cudaMemcpyHostToDevice));
  Table table;
  init_table(table,ENTRIES,ELEMNTS);
  cudaEvent_t start,stop;
  HANDLE_ERROR(cudaEventCreate(&start,0));
  HANDLE_ERROR(cudaEventCreate(&stop,0));
  HANDLE_ERROR(cudaEventRecord(start,0));
  add_elem<<<60,256>>>(table,dev_key,dev_value,dev_lock);
  HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
  HANDLE_ERROR( cudaEventSynchronize( stop ) );
  float   elapsedTime;
  HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
    start, stop ) );
  printf( "Time to hash:  %3.1f ms\n", elapsedTime );
  verifytable(table);
  freettable(table);
  HANDLE_ERROR( cudaEventDestroy( start ) );
  HANDLE_ERROR( cudaEventDestroy( stop ) );
  HANDLE_ERROR( cudaFree( dev_lock ) );
  HANDLE_ERROR( cudaFree( dev_key ) );
  HANDLE_ERROR( cudaFree( dev_value ) );
  free(buff);
  getchar();
  return 0;
}
