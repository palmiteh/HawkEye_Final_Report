#ifndef HAWK_PIPELINE
#define HAWK_PIPELINE

typedef unsigned char Pixel;
#include <pthread.h>
#include <cuda_runtime.h>
#include <unistd.h>

#define MAX_PIPELINES 5
#define BUFFER_SIZE 10
#define MAX_CAMERAS 3
#define START_DELAY 1000

void *start_pipeline(void *);

class HawkEye_Pipeline{
protected:
  Pixel *buffers[BUFFER_SIZE];
  volatile bool running;
  volatile int nextBuffer;
  volatile int latestOutput;
  int lastOutput;
  pthread_mutex_t buffer_mutex;
  pthread_mutex_t output_mutex;
  pthread_mutex_t input_mutex;
  pthread_mutex_t ready_mutex;
  int num_pipelines;
  volatile int current_buffers[MAX_PIPELINES];
  pthread_t threads[MAX_PIPELINES];
  cudaStream_t streams[MAX_PIPELINES];
public:
  int add_pipeline(){
    if(num_pipelines==MAX_PIPELINES) return -1;
    pthread_create(&(threads[num_pipelines]),0,start_pipeline,(void*)this);
    usleep(START_DELAY);
    num_pipelines++;
  }
  
  void setup(){
    
    add_pipeline();
  };  
  
  HawkEye_Pipeline(){
    num_pipelines=0;
    running=true;
    nextBuffer=0;
    latestOutput=-1;
    lastOutput=-1;
    buffer_mutex=PTHREAD_MUTEX_INITIALIZER;
    output_mutex=PTHREAD_MUTEX_INITIALIZER;
    input_mutex=PTHREAD_MUTEX_INITIALIZER;
    ready_mutex=PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&ready_mutex);
    for(int i=0;i<BUFFER_SIZE;i++){
      cudaMalloc((void**)&(buffers[i]),1920*1080*sizeof(char)*3);
    }
    for(int i=0;i<MAX_PIPELINES;i++){
      current_buffers[i]=-1;
      cudaStreamCreate(&streams[MAX_PIPELINES]);
    }
  }  
    
    
  int getBuffer(){
    pthread_mutex_lock(&buffer_mutex);
    int op = nextBuffer;
    
    
    
    nextBuffer = (++nextBuffer)%BUFFER_SIZE;
    bool ok=true;
    do{
      if(nextBuffer==lastOutput) ok=false;
      for(int i=0;i<num_pipelines;i++)
	if(nextBuffer==current_buffers[i]) ok=false;
      if(!ok){
	nextBuffer = (++nextBuffer)%BUFFER_SIZE;
      }
    }while(!ok);
    pthread_mutex_unlock(&buffer_mutex);
    return op;
  }
    
  void setLatestOutput(int n){
    pthread_mutex_lock(&output_mutex);
    latestOutput=n;
    pthread_mutex_unlock(&output_mutex);
    pthread_mutex_unlock(&ready_mutex);
  }
  
  virtual void pipeline(Pixel *output,int pipeline_no){
    printf("FUCK");;
  }
  
  void _pipeline(){
    int pipeline_no = num_pipelines;
    while(running){
      current_buffers[pipeline_no]=-1;
      int buf=getBuffer();
      current_buffers[pipeline_no]=buf;
      /// do stuff
      this->pipeline(buffers[buf],pipeline_no);
      cudaStreamSynchronize(streams[pipeline_no]);
      // output to buffers
      setLatestOutput(buf);
      
    }
    pthread_exit(0);
  }
  
  Pixel *getLatestOutput(){
    pthread_mutex_lock(&ready_mutex);
    int buf = latestOutput;
    lastOutput=buf;
    return buffers[buf];
  }
};


void *start_pipeline(void *pip){
  ((HawkEye_Pipeline*)pip)->_pipeline();
}



#endif