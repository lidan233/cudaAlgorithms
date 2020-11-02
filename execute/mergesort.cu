//
// Created by lidan on 23/10/2020.
//

#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <stdint.h>
#include <iostream>
#include <omp.h>
#include <random>
#include <cxxopts.hpp>
#include <sys/timeb.h>
#include <math.h>


#include <book.cuh>

#define min(a, b) (a < b ? a : b)

template<typename T>
void swap(T* f,T* s)
{
    T temp = *f ;
    *f = *s ;
    *s = temp ;

}

template<typename T>
void reverse(T *arr,T n)      //逆序操作
{
    T i=0,j=n-1;
    while(i<j)
    {
        std::swap(arr[i],arr[j]);
        i++;
        j--;
    }
}


template<typename T>
void exchange(T *arr,int n,int i)
{
    reverse(arr,i);
    reverse(arr+i,n-i);
    reverse(arr,n);
}

template<typename T>
void merge(T* first, T* second, int size)
{

    T* end = first+size ;

    while(first<second & second<end)
    {
        if(*first<=*second)
        {
            first++ ;
        }else{
            T* s = second+1 ;
            while(*s < *first){ s++ ;}
            exchange(first,s-first,second-first);
            first += (s-second+1) ;
            second = s ;
        }

    }
}


template<typename T>
void mergesort(T* t,unsigned int size)
{
    if(size>2)
    {
        int split = size/2 ;
        mergesort(t,split) ;
        mergesort(t+split,size-split) ;
        merge(t,t+split,size) ;
    }else{
        if(t[0]>t[1])
        {
            swap(t,t+1) ;
        }
    }
}

__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}


__device__ void gpu_bottomUpMerge(int * from ,int* to ,int start,int mid,int end)
{
    int i = start ;
    int j = mid ;

    for(int k = start;k<end;k++)
    {
        if(i<mid && (j>= end || from[i]<from[j]))
        {
            to[k] = from[i] ;
            i++ ;
        }else{
            to[k] = from[j] ;
            j++;
        }
    }

}

__global__ void  gpu_mergesort(int *from ,int * to ,int size ,int width ,int slices ,dim3* Dthreads,dim3* Dblocks )
{
    unsigned int idx = getIdx(Dthreads,Dblocks) ;
    long start = idx * slices * width ;
    long mid , end ;

    for( long slice = 0 ; slice< slices;slice++)
    {
        if(start >= size)
        {
            break ;
        }

        mid = min(start + (width>>1),size) ;
        end = min(start + width,size) ;

        gpu_bottomUpMerge(from,to,start,mid,end) ;
        start += width ;
    }
}

void mergesort(int* array, int size,dim3 threadBlock,dim3 blockGrid)
{
    int* Tdata ;
    int* Wdata ;

    // all threads for a block less than 1024
    dim3* Dthreads ;
    // all block for a grid less than long*long
    dim3* Dblocks ;

    tm() ;

    HANDLE_ERROR(cudaMalloc( (void**) &Tdata, size * sizeof(int) ) ) ;
    HANDLE_ERROR(cudaMalloc( (void**) &Wdata , size * sizeof(int) ) ) ;
    HANDLE_ERROR(cudaMemcpy( Tdata,array,size * sizeof(int),cudaMemcpyHostToDevice ) ) ;
//    HandleError(cudaMemcpy(Wdata,array,size * sizeof(int),cudaMemcpyHostToDevice)) ;

    HANDLE_ERROR(cudaMalloc( (void**) &Dthreads,sizeof(dim3) ) ) ;
    HANDLE_ERROR(cudaMalloc( (void**) &Dblocks,sizeof(dim3) ) );
    HANDLE_ERROR(cudaMemcpy( Dthreads,&threadBlock,  sizeof(dim3),cudaMemcpyHostToDevice )) ;
    HANDLE_ERROR(cudaMemcpy( Dblocks,&blockGrid,  sizeof(dim3),cudaMemcpyHostToDevice )) ;

    int* first_T_data = Tdata ;
    int* first_W_data = Wdata ;

    long n_threads = threadBlock.x * threadBlock.y * threadBlock.z *
                    blockGrid.x * blockGrid.y * blockGrid.z ;

    int width = 2 ;
    while( width < (size<<1))
    {
        int slice = size /((n_threads) * width ) + 1 ;
        gpu_mergesort<<<blockGrid,threadBlock>>>(first_T_data,first_W_data,size,width,slice,Dthreads,Dblocks) ;

        first_T_data = (first_T_data==Tdata)?Wdata:Tdata ;
        first_W_data = (first_W_data==Wdata)?Tdata:Wdata ;
        width <<= 1 ;
    }
    tm() ;

    HANDLE_ERROR(cudaMemcpy(array,first_T_data,size * sizeof(int),cudaMemcpyDeviceToHost)) ;
    cudaFree(first_T_data) ;
    cudaFree(first_W_data) ;


}
int main(void )
{
    int t[10] = {10,7,7,3,8,8,2,3,9,10} ;

//    mergesort(t,10) ;
    mergesort(t,10,dim3(10,10),dim3(1)) ;

    for(int i = 0;i < 10 ;i++)
    {
        std::cout<<t[i] ;
    }
}