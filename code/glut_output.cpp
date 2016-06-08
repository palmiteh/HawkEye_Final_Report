
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

//#include "base_module_kernels.h"

#include <helper_functions.h> // includes for SDK helper functions
#include <helper_cuda.h>      // includes for cuda initialization and error checking
#include "camera_2.0.h"
#include "demosaic.h"
#include "threshold.h"
#include "newfilters/edge_overlay.h"
#include "newfilters/scale_overlay.h"
#include "newfilters/digitalZoom.h"
#include <signal.h>

#include "frame.h"

//
// Cuda example code that implements the Sobel edge detection
// filter. This code works for 8-bit monochrome images.
//

int camno=1;
typedef unsigned char Pixel;
Pixel 	edge_thresh=15; 
PGCamera *cam1,*cam2,*cam3;
Pixel *rgbImage,*monoImage;
Pixel *overlayImage,*zoomImage;


void cleanup(void);
void initializeData() ;



/*
int lineShowMode=0;
line lines[4];
bool lines_set[4];
int current_line=0;
*/
#define REFRESH_DELAY     1000/60 //1000/60 //ms


// Code to handle Auto verification
int fpsCount = 0;      // FPS count for averaging
int fpsLimit = 8;      // FPS limit for sampling
unsigned int frameCount = 0;
StopWatchInterface *timer = NULL;
unsigned int g_Bpp;

// Display Data
static GLuint pbo_buffer = 0;  // Front and back CA buffers
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

static GLuint texid = 0;       // Texture for display
unsigned char *pixels = NULL;  // Image pixel data on the host


#define OFFSET(i) ((char *)NULL + (i))

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);

        //glutSetWindowTitle(fps);
        std::cout << ifps <<'\n';
        fflush(stdout);
        fpsCount = 0;

        sdkResetTimer(&timer);
    }
}


bool filterOn=false;
bool zoomPreview=false;

int zoomPix=1920/2;
Pixel threshold=40;
Frame *f=0;
Frame *irf=0;
// This is the normal display path
void display(void)
{
    sdkStartTimer(&timer);
    Frame *f2;
    Pixel *output;
    
    Pixel *data = NULL;
 
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&data, &num_bytes,cuda_pbo_resource));
    
    rgbImage = data;

    switch(camno){
      case 1:
	f=grabFrame(cam1);
	if(filterOn) demosaic_GBRG(rgbImage,monoImage,(Pixel*)f->get_mono_d(),0,*(f->getStream()));
	else demosaic_GBRG(rgbImage,0,(Pixel*)f->get_mono_d(),0,*(f->getStream()));
	
	if(zoomPreview){
	  f2=grabFrame(cam2);
	  demosaic_GBRG(zoomImage,0,(Pixel*)f2->get_mono_d(),1,*(f2->getStream()));
	  int x = (1920-zoomPix)/2;
	  int y = (1080-int((1080.f*(float)zoomPix/1920.f)))/2;
	  digitalZoom(overlayImage,zoomImage,x,y,zoomPix,*(f2->getStream()));
	  
	}
	if(filterOn) edgeOverlay(rgbImage,monoImage,edge_thresh,*(f->getStream()));
	
	if(zoomPreview){
	  f2->synchronize();
	  scaleOverlay(rgbImage,overlayImage,1200,500,712,*(f->getStream()));
	}
      break;
      case 2:
	f=grabFrame(cam3);
	thresholdFilter(rgbImage,(Pixel*)f->get_mono_d(),threshold,*(f->getStream()));
      break;
      case 3:
	f=grabFrame(cam2);
	demosaic_GBRG(overlayImage,monoImage,(Pixel*)f->get_mono_d(),1,*(f->getStream()));
	if(filterOn) edgeOverlay(overlayImage,monoImage,edge_thresh,*(f->getStream()));
	
	int x = (1920-zoomPix)/2;
	int y = (1080-int((1080.f*(float)zoomPix/1920.f)))/2;
	
	digitalZoom(rgbImage,overlayImage,x,y,zoomPix,*(f->getStream()));
      break;
    }
    
							 
    //cudaMemcpy((void*)data,(void*)rgbImage,1920*1080*4,cudaMemcpyDeviceToDevice);	
  
    f->synchronize();
 
    
    //if(sobelize) g_SobelDisplayMode = SOBELDISPLAY_SOBELTEX;
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
    //comboFilter(frame1,frame2);
    

    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_2D, texid);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1920, 1080,
                    GL_RGBA, GL_UNSIGNED_BYTE, OFFSET(0));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBegin(GL_QUADS);
    glVertex2f(0, 0);
    glTexCoord2f(0, 0);
    glVertex2f(0, 1);
    glTexCoord2f(1, 0);
    glVertex2f(1, 1);
    glTexCoord2f(1, 1);
    glVertex2f(1, 0);
    glTexCoord2f(0, 1);
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    glutSwapBuffers();

    sdkStopTimer(&timer);

    computeFPS();
}

void timerEvent(int value)
{
    if(glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}


void reshape(int x, int y)
{
    glViewport(0, 0, x, y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void interrupt(int sig){
    cleanup();
}
void cleanup(void)
{
    cudaGraphicsUnregisterResource(cuda_pbo_resource);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glDeleteBuffers(1, &pbo_buffer);
    glDeleteTextures(1, &texid);
    deleteCamera(cam1);
    deleteCamera(cam2);
    deleteCamera(cam3);

    sdkDeleteTimer(&timer);
  
    cudaDeviceReset();
}

void initializeData()
{
    GLint bsize;
    unsigned int w, h;
    int width=1920;
    int height=1080;
    int channels=4;
    pixels= (unsigned char *) malloc(sizeof(unsigned char) * width * height *channels);
    w = width;
    h = height;
    g_Bpp = 4;

    //deleteTexture();
    memset(pixels, 0x0, g_Bpp * sizeof(Pixel) * 1920 * 1080);

    glGenBuffers(1, &pbo_buffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 g_Bpp * sizeof(Pixel) * 1920 * 1080,
                 pixels, GL_STREAM_DRAW);

    glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bsize);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo_buffer, cudaGraphicsMapFlagsWriteDiscard));

    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexImage2D(GL_TEXTURE_2D, 0, ((g_Bpp==1) ? GL_LUMINANCE : GL_RGBA),
                 1920, 1080,  0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    
}


void initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(1920,1080);
    glutCreateWindow("CUDA Edge Detection");
    glutFullScreen();

    glewInit();
}


void keyboard(unsigned char key, int /*x*/, int /*y*/){
  
  switch(key){
     case 'q': raise(SIGINT);
     case '.':
     case '>': threshold++;
      break;
     case ',':
     case '<': threshold--;
      break;
  
     case '1': camno=1;
     break;
     case '2': camno=2;
     break;
     case '3': camno=3;
     break;
     
     case 'f': filterOn=true;
     break;
     case 'g': filterOn=false;
     break;
     case 'h': edge_thresh--;
     break;
     case 'j': edge_thresh++;
     break;
     
     
     case 'a': threshold--;
     break;
     case 's': threshold++;
     break;
     
     case '[': zoomPix+=5;
     break;
     case ']': zoomPix-=5;
     break;
     
     
     case 'z': zoomPreview=true;
     break;
     case 'x': zoomPreview=false;
     break;
  }
  printf("%d\n",threshold);
}


/*
void mouse_move(int x, int y){
  lines[current_line].x2=x;
  lines[current_line].y2=y;
  lines[current_line].setup();
  lineShowMode=1;
}
  */

int main(int argc, char **argv)
{
    
   
    //setup();
   
    initGL(&argc, argv);
    cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
    //cudaMalloc((void**)&gradout,1920*1080*sizeof(uchar2));
    cudaMalloc((void**)&rgbImage,1920*1080*sizeof(char)*4);
    cudaMalloc((void**)&monoImage,1920*1080*sizeof(char));
    cudaMalloc((void**)&overlayImage,1920*1080*sizeof(char)*4);
    cudaMalloc((void**)&zoomImage,1920*1080*sizeof(char)*4);
    //ircam = newMonoCamera("bob","41C6NIR.pgrguid");
    cam1 = newMonoCamera("bob","41C6C.pgrguid");
    cam2 = newMonoCamera("larry","23S6C.pgrguid");
    cam3 = newMonoCamera("sally","41C6NIR.pgrguid");
    // cam2 = newMonoCamera("larry","41C6NIR.pgrguid");
   // cam3 = newCamera("sally","23S6C.pgrguid");
    //cam = newCamera("larry","23S6C.pgrguid");
    //if(argc>1) cam=newMonoCamera("bob",argv[1]);	
    //else cam = newCamera("bob");
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    //glutMouseFunc(mouse);
    //glutMotionFunc(mouse_move);
    glutReshapeFunc(reshape);

    initializeData();

    glutCloseFunc(cleanup);

    glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    glutMainLoop();
    signal(SIGINT,interrupt);
}
