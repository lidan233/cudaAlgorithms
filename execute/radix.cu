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

#define NUM 10000
static uint32_t cpu_tmp_0[NUM] ;
static uint32_t cpu_tmp_1[NUM] ;


__host__ void cpusort(uint32_t* const data,const uint32_t num_elements)
{
    for(uint32_t bit = 0 ; bit <32 ;bit++)
    {
        uint32_t base_cnt_0 = 0 ;
        uint32_t base_cnt_1 = 0 ;

        for(uint32_t i = 0 ; i< num_elements;i++)
        {
            const uint32_t d = data[i] ;
            const uint32_t bit_mask = (1<<bit) ;
            if((d & bit_mask) > 0)
            {
                cpu_tmp_1[base_cnt_1++] = d ;
            }else{
                cpu_tmp_0[base_cnt_0++] = d ;
            }
        }

        for(uint32_t i =0 ; i< base_cnt_0;i++)
        {
            data[i] = cpu_tmp_0[i] ;
        }

        for(uint32_t i = 0 ; i<base_cnt_1;i++)
        {
            data[base_cnt_0+i] = cpu_tmp_1[i] ;
        }


    }
}

__global__ void gpu_radixsort2(unsigned int * const sort_tmp,
                               int NUM_ELEMENT ,
                               int NUM_LISTS ,
                               unsigned int * const sort_tmp_1,
                               const unsigned int tid) //桶排序
{
    for(unsigned int bit_mask = 1; bit_mask > 0; bit_mask <<= 1)    //32位
    {
        unsigned int base_cnt_0 = 0;
        unsigned int base_cnt_1 = 0;

        for (unsigned int i = 0; i < NUM_ELEMENT; i+=NUM_LISTS)
        {
            if(sort_tmp[i+tid] & bit_mask)  //该位是1，放到sort_tmp_1中
            {
                sort_tmp_1[base_cnt_1+tid] = sort_tmp[i+tid];
                base_cnt_1 += NUM_LISTS;
            }
            else    //该位是0，放到sort_tmp的前面的
            {
                sort_tmp[base_cnt_0+tid] = sort_tmp[i+tid];
                base_cnt_0 += NUM_LISTS;
            }
        }

        for (unsigned int i = 0; i < base_cnt_1; i+=NUM_LISTS)  //将sort_tmp_1的数据放到sort_tmp后面
        {
            sort_tmp[base_cnt_0+i+tid] = sort_tmp_1[i+tid];
        }
        __syncthreads();
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


__global__ void gpu_radixsort(uint32_t* const data, const uint32_t num_elements,int numlist, uint32_t* t,uint32_t* w )
{
    const unsigned int  idx = (blockIdx.x* blockDim.x) + threadIdx.x ;
    const unsigned int idy = (blockIdx.y*blockDim.y) + threadIdx.y ;
    const unsigned int tid = idy*gridDim.x*blockDim.x + idx ;
//    const unsigned  int tid = threadIdx.x ;

    for(int i = 0 ; i< 32; i++)
    {
        int base_0 = 0;
        int base_1 = 0 ;
        const int bit_mask = (1 << i) ;

        for(int j=0 ; j< num_elements;j+=numlist)
        {
            const int elem = data[j+tid] ;


            if((elem & bit_mask) > 0)
            {
                t[base_1+tid] = elem ;
                base_1 += numlist ;
            }else{
                w[base_0+tid] = elem ;
                base_0 += numlist ;
            }
        }


        for(int m = 0; m < base_0;m+=numlist)
        {
            data[m+tid] = w[m+tid] ;
        }
        for(int m = 0; m < base_1;m+=numlist)
        {
            data[base_0+m+tid] = t[m+tid] ;
        }
        __syncthreads() ;
    }

}


void radixsort_gpu(uint32_t* data, uint32_t num_element )
{
    uint32_t *cuda_data ;
    uint32_t *tdata ;
    uint32_t *wdata ;

    checkCudaErrors(cudaMalloc((void**)&cuda_data,num_element*sizeof(int))) ;
    checkCudaErrors(cudaMalloc((void**)&tdata,num_element*sizeof(int))) ;
    checkCudaErrors(cudaMalloc((void**)&wdata,num_element*sizeof(int))) ;

    checkCudaErrors(cudaMemcpy(cuda_data,data,sizeof(int)*num_element,cudaMemcpyHostToDevice)) ;

    gpu_radixsort<<<1,dim3(10,10,1)>>>(cuda_data, num_element,10000,tdata,wdata) ;

    checkCudaErrors(cudaMemcpy(data,cuda_data,sizeof(int)*num_element,cudaMemcpyDeviceToHost)) ;

}

#define MAX_NUM_LIST 10000
__device__ void mergeArray(const uint32_t * src_array ,  uint32_t* const dest_array,
                           const uint32_t num_list,
                           const uint32_t num_elements,
                           const uint32_t tid)
{
    const uint32_t num_elements_per_list = (num_elements/num_list) ;
    __shared__ unsigned int  list_indexs[MAX_NUM_LIST] ;
    list_indexs[tid] = 0 ;
    __syncthreads() ;

    for(int i = 0 ;i < num_elements;i++)
    {
        __shared__ int min_val ;
        __shared__ int min_idx ;

        int data ;

        if(list_indexs[tid] < num_elements_per_list)
        {
            const int src_idx = tid + (list_indexs[tid]*num_list) ;
            data = src_array[src_idx] ;
        }else{
            data = 0xFFFFFFF ;
        }

        if(tid==0)
        {
            min_val = 0xFFFFFF ;
            min_idx = 0xFFFFFF ;

        }
        __syncthreads() ;


        atomicMin(&min_val,data) ;
        __syncthreads() ;

        if(min_val==data)
        {
            atomicMin(&min_idx,tid) ;
        }
        __syncthreads() ;

        if(tid==min_idx)
        {
            list_indexs[tid]++ ;
            dest_array[i] = data ;
        }
    }
}


__device__ void merge_two(unsigned int * const data,unsigned int* const dst,const unsigned  int tid ,
                          const int num_list,const int num_element)
{
    const int num_elements_per_list = num_element/num_list ;

    __shared__ int list_indexs[MAX_NUM_LIST] ;
    __shared__ int reduce_val[MAX_NUM_LIST] ;
    __shared__ int reduce_idx[MAX_NUM_LIST] ;

    list_indexs[tid] = 0 ;
    reduce_idx[tid] = 0 ;
    reduce_val[tid] = 0 ;

    __syncthreads() ;

    for(int i = 0 ;i < num_element;i++)
    {
        int tid_max = num_list >> 1 ;
        int t ;

        if(list_indexs[tid] < num_elements_per_list)
        {
            const int src_idx = tid + (list_indexs[tid] * num_list) ;
            t = data[src_idx] ;
        }else{
            t = 0xFFFFFFF ;
        }


        reduce_val[tid] = t ;
        reduce_idx[tid] = tid ;
        __syncthreads() ;


        while(tid_max!=0)
        {
            if(tid < tid_max)
            {
                const int val2_idx = tid + tid_max ;
                const int val2 = reduce_idx[val2_idx] ;

                if(reduce_val[tid]>val2)
                {
                    reduce_val[tid] = val2 ;
                    reduce_idx[tid] = val2_idx ;
                }

            }
        }


        tid_max >>= 1 ;

        __syncthreads() ;

        if(tid == 0 )
        {
            list_indexs[reduce_idx[0]]++ ;
            dst[i] = reduce_val[0] ;
        }

        __syncthreads() ;
    }

}

#define REDUCTION_SIZE  8
#define REDUCTION_SHIFT 3

__device__ void merge_final(unsigned int * const srcData,
                            unsigned int * const dstData,
                            const unsigned int NUM_LIST ,
                            const unsigned int NUM_ELEMENTS ,
                            const unsigned int tid)
{
    __shared__ unsigned int list_reduction[MAX_NUM_LIST] ;
    unsigned int num_reduction = NUM_LIST >> REDUCTION_SHIFT ;
    unsigned int s_tid = tid >> REDUCTION_SHIFT ;
    unsigned int self_index = tid ;
    unsigned int min_val ;


    for(int i = 0 ; i< NUM_ELEMENTS;i++)
    {
        int t = 0xFFFFFF ;
        if(self_index<NUM_ELEMENTS)
        {
            t = srcData[self_index];
        }

        if(tid < NUM_LIST/REDUCTION_SIZE)
        {
            list_reduction[tid] = 0xFFFFFF ;
        }

        __syncthreads() ;

        atomicMin(&(list_reduction[s_tid]),t) ;

        __syncthreads() ;

        if(tid == 0 )
        {
            min_val = 0xFFFFFF ;
        }

        __syncthreads() ;

        if(tid<NUM_LIST/REDUCTION_SIZE)
        {
            atomicMin(&min_val,list_reduction[tid]) ;
        }

        __syncthreads() ;

        if(min_val == t)
        {
            dstData[i] = min_val ;
            self_index += NUM_LIST ;
            min_val = 0xFFFFFF ;
        }
    }
}
int main()
{

    uint32_t t[10] = {10,7,7,3,8,8,2,3,9,10} ;

//    mergesort(t,10) ;
    radixsort_gpu(t,10) ;

    for(int i = 0;i < 10 ;i++)
    {
        std::cout<<t[i] ;
    }

}