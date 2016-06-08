#include <cuda_runtime.h>
#include "base_module_kernels.h"
#include "/home/ubuntu/HawkEyedVideo/include/FlyCapture2.h"
#include <iostream>
#include <unistd.h>
#include <pthread.h>
#include "frame.h"
#include "camera_2.0.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

using namespace FlyCapture2;
using namespace std;

#ifndef BUFFER_SIZE
#define BUFFER_SIZE 10
#endif


#define FPS_MAX 60

/// keeps Flycapture from releasing the buffer until the image has been processed


void *capture(void*);

class PGCamera{
private:
  int current_frame;
  int current_output;
  Camera camera;
  Frame frames[BUFFER_SIZE];
  unsigned char *input_buffer;
  unsigned char *input_buffer_d;
  Image rawImages[BUFFER_SIZE];
  PGRGuid guid;
  bool firstrun;
  bool monoOutput,rgbOutput;
  CameraType camType;
  char *name;
  
  pthread_mutex_t mutex;
  pthread_t thread;
  
  void fail(){
     cout << "Camera " << name << " failed!" << endl;
     delete this;
     raise(SIGINT);
  }
  
  
  bool running;
  
public:
  static int camera_no;
  static BusManager busMgr;
  volatile int framesready;
      
  void setup(){
      framesready=0;
      current_frame=0;
      current_output=0;
      mutex=PTHREAD_MUTEX_INITIALIZER;
    // set up user buffer to recieve data
    cudaHostAlloc((void**)&input_buffer, 1920*1080*sizeof(char)*BUFFER_SIZE, cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&input_buffer_d,(void*)input_buffer,0);
    
    // tell frames not to destroy memory when done
    // create CUDA stream for each frame
    for(int i=0;i<BUFFER_SIZE;i++) {
       frames[i].disableFree();
    }
    // activate camera
    
    
    Error error = camera.Connect(&guid);
    if(error != PGRERROR_OK){
      cout << "Failed to connect to camera " << name << "!" << endl;
      fail();
    }
    
    //reset camera
  //  camera.WriteRegister(0x610,0);
  //  camera.WriteRegister(0x610,1);
    
    
    //set up user buffers
    if((camera.SetUserBuffers(input_buffer,1920*1080*sizeof(char),BUFFER_SIZE))!=PGRERROR_OK){
        fprintf(stderr,"Failed to setup buffers\n");
	fail();
    }
    
    
    // get camera info and print it out
    CameraInfo camInfo;
    camera.GetCameraInfo(&camInfo);
    cout << camInfo.vendorName << " " << camInfo.modelName << " "
              << camInfo.serialNumber << endl;
    
    // set up camera output
    Mode k_fmt7Mode=MODE_0;
    //if(camType==MONO) k_fmt7Mode= MODE_0;
    //else k_fmt7Mode=MODE_7;
    const PixelFormat k_fmt7PixFmt = PIXEL_FORMAT_RAW8;
    Format7Info fmt7Info;
    bool supported;
    fmt7Info.mode = k_fmt7Mode;
    error = camera.GetFormat7Info(&fmt7Info, &supported);
    Format7ImageSettings fmt7ImageSettings;
    fmt7ImageSettings.mode = k_fmt7Mode;
    fmt7ImageSettings.offsetX = (fmt7Info.maxWidth - 1920) / 2;
    fmt7ImageSettings.offsetY = (fmt7Info.maxHeight - 1080) / 2;
    fmt7ImageSettings.width = 1920;
    fmt7ImageSettings.height = 1080;
    fmt7ImageSettings.pixelFormat = k_fmt7PixFmt;
    bool valid;
    Format7PacketInfo fmt7PacketInfo;
    
    // Validate the settings to make sure that they are valid
    error = camera.ValidateFormat7Settings(&fmt7ImageSettings, &valid,
                                           &fmt7PacketInfo);
    if (error != PGRERROR_OK) {
      cout << "Invalid camera settings" << endl;
      fail();
    }

    error = camera.SetFormat7Configuration(
        &fmt7ImageSettings, fmt7PacketInfo.recommendedBytesPerPacket);
    if(error != PGRERROR_OK){
      cout << "Failed to set format!" << endl;
      fail();
    }
    
    //camera.WriteRegister(0x968,60,1);
    /*Property myFrameRate;
    myFrameRate.type = FRAME_RATE;
    myFrameRate.absValue = 60.0;
    camera.SetProperty(&myFrameRate);
    */
    error = camera.StartCapture();
    if(error!=PGRERROR_OK){
      cout << "Failed to start image capture!" << endl;
      fail();
    }
    
    Property frmRate;
    frmRate.type = FRAME_RATE;
    error = camera.GetProperty(&frmRate);
    if (error != PGRERROR_OK) {
      // PrintError( error );
      exit(-1);
    }
    cout << "Maximum frame rate is " << fixed << frmRate.absValue << " fps" << endl;
    running=true;
    usleep(1000);
    pthread_create(&thread,0,capture,(void*)this);    
  }  
  

  bool isRunning(){
    bool var;
    pthread_mutex_lock(&mutex);
    var=running;
    pthread_mutex_unlock(&mutex);
    return var;
  }
  
  ~PGCamera(){
    for(int i=0;i<BUFFER_SIZE;i++){
      frames[i].enableFree();
    }
    pthread_mutex_lock(&mutex);
    running=false;
    pthread_mutex_unlock(&mutex);
    int x,*xx=&x;
    pthread_join(thread,(void**)&xx);
    cudaFreeHost(input_buffer);
    camera.StopCapture();
    
    camera.Disconnect();
  } 
public:
  
  PGCamera(const char *newname){
    camType=RGB;
    unsigned int cameras_connected;
    busMgr.GetNumOfCameras(&cameras_connected);
    if(camera_no + 1 > cameras_connected) {
      if(camera_no==100) printf("Must use GUIDs for ALL connected cameras or NONE\n");
      else printf("No camera %d\n", camera_no);
      raise(SIGINT);
    }
    busMgr.GetCameraFromIndex(camera_no,&guid);
    camera_no++;
    name = (char*)malloc(sizeof(char)*strlen(newname)+1);
    strcpy(name,newname);
    setup();
  }
  
  PGCamera(const char *newname,const char *filename){
    camType=RGB;
    int fd = open(filename,O_RDONLY);
    if(fd==-1) {
      printf("No file %s\n",filename);
      raise(SIGINT);
    }
    read(fd,(void*)&guid,16);
    close(fd);
    
    name = (char*)malloc(sizeof(char)*strlen(newname)+1);
    strcpy(name,newname);
    camera_no = 100;
    setup();
  }
  
  PGCamera(const char *newname,const char *filename,CameraType ct){
    int fd = open(filename,O_RDONLY);
    read(fd,(void*)&guid,16);
    close(fd);
    
    name = (char*)malloc(sizeof(char)*strlen(newname)+1);
    strcpy(name,newname);
    camera_no = 100;
    camType=ct;
    setup();
    
  }
  
  
  bool readyToOutput(){
    bool ret;
    pthread_mutex_lock(&mutex);
    if(framesready>0) ret=true;
    else ret=false;
    pthread_mutex_unlock(&mutex);
    return ret;
  }
  
  Frame *grabFrame(){
    while(!readyToOutput())usleep(1000);
    int output_frame = current_output;
    current_output = ++current_output % BUFFER_SIZE;
    pthread_mutex_lock(&mutex);
    framesready--;
    pthread_mutex_unlock(&mutex);
    return &(frames[output_frame]);
  }
  
  bool readyToCapture(){
    bool ret;
    pthread_mutex_lock(&mutex);
    if(framesready+1<BUFFER_SIZE) ret=true;
    else ret=false;
    pthread_mutex_unlock(&mutex);
    return ret;
  }
  
  
  void capture_loop(){
     clock_t clocks_per_frame = CLOCKS_PER_SEC/FPS_MAX;
     Error error;
     
     clock_t start_time,stop_time;
     while(isRunning()){
        
       while(framesready+1>=BUFFER_SIZE) usleep(1000);
	
       
      start_time=clock();
      frames[current_frame].synchronize();
      frames[current_frame].start();
      Image rawImage;
      error = camera.RetrieveBuffer(&rawImage);
      if(error!=PGRERROR_OK) cout << "error";
      // get offset and apply to device buffer
      void *offset_d =(void*)(input_buffer_d+(rawImage.GetData()-input_buffer));
      frames[current_frame].start();
      frames[current_frame].getFromRAW(offset_d,1920,1080,camType);
      
      current_frame = ++(current_frame) % BUFFER_SIZE;
      pthread_mutex_lock(&mutex); 
      framesready++;
      pthread_mutex_unlock(&mutex);
      
      stop_time = clock()-start_time;
      //cout<<stop_time << "us" << endl;
      if(stop_time<clocks_per_frame) {
	//cout << "need to sleep " << clocks_per_frame-stop_time << " cycles" << endl;
	//usleep(clocks_per_frame-stop_time);    
       
    
      }
    }
  }
  
};

void *capture(void *th){
    PGCamera *thiss=(PGCamera*)th;
    thiss->capture_loop();
};

int PGCamera::camera_no=0;
BusManager PGCamera::busMgr;

PGCamera *newCamera(const char *name){
   return new PGCamera(name);
}

PGCamera *newCamera(const char *name, const char *filename){
  return new PGCamera(name,filename);
}

PGCamera *newMonoCamera(const char *name, const char *filename){
  return new PGCamera(name,filename,MONO);
}

void deleteCamera(PGCamera *cam){
  delete cam;
}

Frame *grabFrame(PGCamera *cam){
  return cam->grabFrame();
}
