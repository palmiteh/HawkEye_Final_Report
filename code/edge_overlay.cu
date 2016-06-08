#include "edge_overlay.h"
#include <cuda_runtime.h>
#define BPP 4


texture<unsigned char, 2> tex;


static __device__ float getXY(int x, int y){
  //return (float)input[y*1920+x];
  return (float)tex2D(tex,x,y);
}

__global__ void _edge_overlay(Pixel *output, Pixel threshold){
  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y_start = threadIdx.y*540;
  float a,b,c,d;
  for(int i=0;i<540;i++){
    int y=y_start+i;
    int idx = ((y)*1920+x)*BPP;
    if(!(x<2 || x>=1918 || y>=1078 || y<2)){
      
	float a=getXY(x,y);
	float b=getXY(x+1,y);
      
      c=getXY(x,y+1);
      d=getXY(x+1,y+1);
      
      float gx = (b+d)-(a+c);
      float gy = (c+d)-(a+b);
      Pixel mag = Pixel(sqrt(gx*gx+gy*gy)/2.0);
      if(mag>threshold){
	output[idx] = 0;
	output[idx+1] = 255;
	output[idx+2] = 0;
      }
    }
  }
}

void edgeOverlay(Pixel *output, Pixel *input, Pixel threshold, cudaStream_t stream){
  size_t offset;
  cudaChannelFormatDesc channeldesc=cudaCreateChannelDesc<unsigned char>();
  cudaBindTexture2D(&offset,tex,input,channeldesc,1920,1080,1920);
  
  dim3 threads(128,2);
  dim3 blocks(1920/128);
  _edge_overlay<<<blocks,threads,0,stream>>>(output,threshold);
  cudaUnbindTexture(tex);
}
