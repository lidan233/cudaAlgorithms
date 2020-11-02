//
// Created by lidan on 2020/9/20.
//

#ifndef CUDASAMPLER_BOOK_CH
#define CUDASAMPLER_BOOK_CH

#include <helper_cuda.h>
#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

static void HandleError(cudaError err,const char *file ,int line)
{
    if(err != cudaSuccess)
    {
        printf("%s in %s at line %d\n" ,cudaGetErrorString(err),file,line) ;
        exit(EXIT_FAILURE) ;
    }
}

#define HANDLE_ERROR(err) (HandleError(err,__FILE__,__LINE__))

#define HANDLE_NULL(a) {if(a==NULL) {\
                        printf("Host memory failed in %s at line %d\n",\
                                __FILE__,__LINE__) ;\
                        exit(EXIT_FAILURE) ;    }}

template<typename T>
void swap(T& a,T& b)
{
    T t= a ;
    a  = b ;
    b = t;
}

void* big_random_block(int size)
{
    unsigned char *data = (unsigned char*)malloc(size) ;
    HANDLE_NULL(data) ;
    for(int i= 0;i<size;i++)
    {
        data[i] = rand() ;
    }
    return data ;
}

int* big_random_block_int(int size)
{
    int* data = (int*)malloc(size*sizeof(int)) ;
    HANDLE_NULL(data) ;
    for(int i = 0; i<size;i++)
    {
        data[i] = rand() ;
    }
    return data ;
}


//__device__ unsigned char value(float n1,float n2,int hue)
//{
//    if(hue)
//}
//


#if _WIN32
//Windows threads.
#include <windows.h>

typedef HANDLE CUTThread;
typedef unsigned (WINAPI *CUT_THREADROUTINE)(void *);

#define CUT_THREADPROC unsigned WINAPI
#define  CUT_THREADEND return 0

#else
//POSIX threads.
    #include <pthread.h>

    typedef pthread_t CUTThread;
    typedef void *(*CUT_THREADROUTINE)(void *);

    #define CUT_THREADPROC void
    #define  CUT_THREADEND
#endif

#endif //CUDASAMPLER_BOOK_CH
