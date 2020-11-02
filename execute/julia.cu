//
// Created by lidan on 27/09/2020.
//

#include "book.cuh"
#include "cpu_bitmap.cuh"

#include <glad.h>
#include <GLFW/glfw3.h>
//
struct cuComplex{
    float r;
    float i;

    __device__  cuComplex(float x,float y) : r(x),i(y){}

    __device__ float manitude2(void)
    {
        return r*r + i* i ;
    }

    __device__ cuComplex operator*(const cuComplex& a)
    {
        return cuComplex(r*a.r-i*a.i ,i*a.r + r* a.i ) ;
    }

    __device__ cuComplex operator+(const cuComplex& a)
    {
        return cuComplex(r+a.r,i+a.i ) ;
    }
};

__device__ int julia(int x,int y,float scale)
{
    float jx = scale * (float) (SCR_WIDTH/2- x)/(SCR_WIDTH/2) ;
    float jy = scale * (float) (SCR_HEIGHT/2 - y)/(SCR_HEIGHT/2) ;

    cuComplex c(-0.8, 0.156) ;
    cuComplex d(jx,jy) ;

//    for(int i = 0 ;i <200 ;i++)
//    {
//        d = d*d + c;
//        if(d.manitude2() > 1000)
//        {
//            return 0 ;
//        }
//    }
//
//    return 1 ;


    int iterations = 0;

    while (true) {
        iterations++;
        if (iterations >1000) return 0;
        d = d*d + c;
        if (d.i > 150) return iterations;
        if (d.r > 150) return iterations;
    }
    return iterations ;

}


__global__ void kernel(const unsigned int width,const unsigned int height ,unsigned char* ptr,float scale )
{
    const unsigned int  idx = (blockIdx.x* blockDim.x) + threadIdx.x ;
    const unsigned int idy = (blockIdx.y*blockDim.y) + threadIdx.y ;

    const unsigned int tid = idy*gridDim.x*blockDim.x + idx ;

    int x = tid / width ;
    int y = tid % width ;

    if(x < height)
    {
        int juliaValue = julia(x,y,scale) ;

        ptr[(x*width+y)*4 + 0] = 190+120*juliaValue;
        ptr[(x*width+y)*4 + 1] = 40+35*round(cos(juliaValue/5.0));
        ptr[(x*width+y)*4 + 2] =  18+6*(juliaValue%10);
        ptr[(x*width+y)*4 + 3] = 255;
    }
}


void processData(void* datain)
{
    unsigned char* data = reinterpret_cast<unsigned char*>(datain);
    unsigned char* usedata ;
    HANDLE_ERROR(cudaMalloc((void**) &usedata,SCR_WIDTH*SCR_HEIGHT*4)) ;
    dim3 blocknum(100,100) ;

    kernel<<<blocknum,192>>>(SCR_WIDTH,SCR_HEIGHT,usedata,scale) ;
    HANDLE_ERROR(cudaMemcpy(data,usedata,SCR_WIDTH*SCR_HEIGHT*4,cudaMemcpyDeviceToHost)) ;
    cudaFree(usedata) ;
}




int main( void ) {
    DataBlock   data;
    data.dev_bitmap = (unsigned  char*) malloc(sizeof(unsigned char)*SCR_WIDTH*SCR_HEIGHT*4) ;
//
    unsigned char* usedata ;
    HANDLE_ERROR(cudaMalloc((void**) &usedata,SCR_WIDTH*SCR_HEIGHT*4)) ;

    dim3 blocknum(100,100) ;

    kernel<<<blocknum,192>>>(SCR_WIDTH,SCR_HEIGHT,usedata,2) ;
    HANDLE_ERROR(cudaMemcpy(data.dev_bitmap,usedata,SCR_WIDTH*SCR_HEIGHT*4,cudaMemcpyDeviceToHost)) ;

    cudaFree(usedata) ;

//    for(int i = 0 ;i < SCR_WIDTH ;i++)
//    {
//        for(int j = 0 ; j < SCR_HEIGHT;j++)
//        {
//            for(int k = 0 ;k < 4;k++)
//            {
//                data.dev_bitmap[(i*SCR_HEIGHT+j)*4+k] = 100 ;
//            }
//        }
//    }

    CPUBitmap bitmap( SCR_WIDTH, SCR_HEIGHT, data.dev_bitmap );
    bitmap.display_and_exit(processData) ;

    printf("shit" );
}