#include "cuda.h"
#include "..\common\book.h"
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
  Entry *firstfree;
};

void init_table(Table &table,size_t entr,size_t elem)
{
  table.count=entr;
  //notes: callo(size_t nelem, size_t elsize) alocate nelem blocks memory which size is elsize,make the initial
  //value to be 0
  table.entries=(Entry **)calloc(entr,sizeof(Entry*));
  table.pool=(Entry *)malloc(elem*sizeof(Entry));
  table.firstfree=table.pool;
}

size_t hash(unsigned int key,size_t count)
{
  return key%count;
}

void add_elem(Table &table,unsigned int key,void *value)
{
  size_t hashvalue=hash(key,table.count);
  Entry *newelem=table.firstfree++;
  newelem->key=key;
  newelem->value=value;
  newelem->next=table.entries[hashvalue];
  table.entries[hashvalue]=newelem;
}

void freettable(Table &table)
{
  free(table.entries);
  free(table.pool);
}

void verifytable(const Table &table)
{
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
}

int main(void)
{
  unsigned int *buff=(unsigned int*)big_random_block(SISE);
  Table table;
  init_table(table,ENTRIES,ELEMNTS);
  clock_t start,stop;
  start=clock();
  for (size_t i=0;i!=ELEMNTS;++i)
  {
    add_elem(table,buff[i],(void*)NULL);
  }
  stop=clock();
  float elsptime=(float)(stop-start)/(float)CLOCKS_PER_SEC*1000.f;
  printf("total time:%f ms\n",elsptime);
  verifytable(table);
  freettable(table);
  free(buff);
  getchar();
  return 0;
}
