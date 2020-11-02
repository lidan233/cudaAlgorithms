//
// Created by lidan on 24/09/2020.
//

#include <stdio.h>
#include <stdlib.h>
#include <conio.h>

// blockDim 对应gpublock的dim IDx 这个线程在block中的位置

__global__ void what_is_my_id(unsigned int* const block ,unsigned int* const thread ,
                              unsigned int* const warp ,unsigned int* const calc_thread )
{
    const unsigned int thread_idx = (blockIdx.x*blockDim.x) + threadIdx.x ;
    block[thread_idx] = blockIdx.x ;
    thread[thread_idx] = threadIdx.x ;

    warp[thread_idx] = threadIdx.x / warpSize ;
    calc_thread[thread_idx] = thread_idx ;
}


#define ARRAY_SIZE 128
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int)*(ARRAY_SIZE))

unsigned int cpu_block[ARRAY_SIZE] ;
unsigned int cpu_thread[ARRAY_SIZE] ;
unsigned int cpu_wrap[ARRAY_SIZE] ;
unsigned int cpu_calc_thread[ARRAY_SIZE] ;


int main(void)
{
    const unsigned int num_block = 2;
    const unsigned int num_threads = 64 ;

    char ch ;

    unsigned int * gpu_block ;
    unsigned int * gpu_thread ;
    unsigned int * gpu_warp ;
    unsigned int * gpu_calc_thread ;

    unsigned int i ;

    cudaMalloc((void**)&gpu_block,ARRAY_SIZE_IN_BYTES) ;
    cudaMalloc((void**)&gpu_thread,ARRAY_SIZE_IN_BYTES) ;
    cudaMalloc((void**)&gpu_warp,ARRAY_SIZE_IN_BYTES) ;
    cudaMalloc((void**)&gpu_calc_thread,ARRAY_SIZE_IN_BYTES) ;

    what_is_my_id<<<num_block,num_threads>>>(gpu_block,gpu_thread,gpu_warp,gpu_calc_thread) ;

    cudaMemcpy(cpu_block,gpu_block,ARRAY_SIZE_IN_BYTES,cudaMemcpyDeviceToHost) ;
    cudaMemcpy(cpu_thread,gpu_thread,ARRAY_SIZE_IN_BYTES,cudaMemcpyDeviceToHost) ;
    cudaMemcpy(cpu_wrap,gpu_warp,ARRAY_SIZE_IN_BYTES,cudaMemcpyDeviceToHost) ;
    cudaMemcpy(cpu_calc_thread,gpu_calc_thread,ARRAY_SIZE_IN_BYTES,cudaMemcpyDeviceToHost) ;

    cudaFree(gpu_calc_thread) ;
    cudaFree(gpu_thread) ;
    cudaFree(gpu_block) ;
    cudaFree(gpu_warp) ;

    for(i = 0;i< ARRAY_SIZE;i++)
    {
        printf("calculate thread: %3u - Block: %2u - Warp : %2u - Thread %3u \n",
               cpu_calc_thread[i],cpu_block[i],cpu_wrap[i],cpu_thread[i]) ;

    }

    ch = getch() ;
}
