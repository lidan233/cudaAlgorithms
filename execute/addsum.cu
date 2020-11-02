
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cxxopts.hpp>
#include <random>
#include <stdlib.h>
#include <time.h>



cxxopts::Options parse_option(int argc, char** argv)
{
    cxxopts::Options options("cuda samples","This is a simple practice of cuda by lidan233.") ;
    options.add_options()
            ("d,debug", "Enable debugging") // a bool parameter
            ("i,integer", "Int param", cxxopts::value<int>())
            ("if,file", "Input file name", cxxopts::value<std::string>())
            ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
            ;
    return options ;
}

#define C_NUM_ELEMENTS 10000
#define C_NUM_LISTS 1000
#define C_SHIFT 3
#define C_SPLITSIZE 8

#define ALL_SPLIT_SHARED_MEMORY 10


template<class T> void c_swap(T&x ,T &y) { T tmp = x ; x = y ; y = tmp ;}


unsigned int gen_and_shuffle(unsigned int* const srcData)
{
    unsigned int res = 0 ;

    srand(time(NULL)) ;
    for(int i = 0 ; i< C_NUM_ELEMENTS ;i++)
    {
        srcData[i] = i ;
        res+=i ;
    }

    for(int i = C_NUM_ELEMENTS ;i > 0  ;i--)
    {
        c_swap(srcData[rand()%(i)],srcData[i-1]) ;
    }
    return res ;
}

void print_data(unsigned int * const srcData)
{
    for(int i = 0; i < C_NUM_ELEMENTS; i++)
    {
        printf("%4u", srcData[i]);
        if((i+1)%32 == 0)
            printf("\n");
    }
}

// 假定size 是numlist的数目
__device__ void addsum_shared_reduce(unsigned int * memory ,const unsigned int tid ,const unsigned int size)
{
    unsigned int tidmax = size>>1 ;
    while(tidmax!=0)
    {
        if(tid < (tidmax))
        {
            memory[tid] = memory[tid]+ memory[tid+tidmax] ;
        }
        tidmax >>= 1 ;
        __syncthreads() ;
    }

}

__device__ void addsum_shared_atomicadd(unsigned int* memory , const unsigned int tid ,const unsigned int size)
{
    if(tid!=0&&tid<size)
    {
        atomicAdd(&(memory[0]),memory[tid]);
    }
}
__device__ void addsum_shared_atomicadd_splitnum(unsigned int * memory ,const unsigned int tid ,const unsigned int size)
{
    __shared__ unsigned int fenduan[ALL_SPLIT_SHARED_MEMORY] ;

    if(tid<ALL_SPLIT_SHARED_MEMORY)
    {
        fenduan[tid] = 0 ;
        __syncthreads() ;
        for(int i = tid ; i< size;i+= ALL_SPLIT_SHARED_MEMORY)
        {
            atomicAdd(&(fenduan[tid]),memory[i]) ;
        }
        __syncthreads() ;
        atomicAdd(&(memory[0]),fenduan[tid]) ;
    }
}

__device__ void addsum2(unsigned int const* data, const unsigned  int datasize , const unsigned int threadsize ,
                       const unsigned int tid,unsigned int* res)
{
    //threadsize = C_NUM_LISTS

    __shared__ unsigned int reducedata[C_NUM_LISTS];
    __shared__ unsigned int result ;

    if(tid < C_NUM_LISTS)
    {
        reducedata[tid] = 0 ;
        result = 0 ;

        __syncthreads() ;


        for(int i =tid ; i< datasize ;i+=C_NUM_LISTS)
        {

            reducedata[tid] = data[i];
            __syncthreads() ;
            addsum_shared_atomicadd(reducedata,tid,C_NUM_LISTS) ;
//            addsum_shared_atomicadd_splitnum(reducedata,tid,C_NUM_LISTS) ;
            __syncthreads() ;
            if(tid==0)
            {
                result += reducedata[0] ;
            }
            __syncthreads() ;
        }

        *res = result ;
    }


}

__device__ void addsum(unsigned int const * data,const unsigned  int datasize ,
                       const unsigned int threadsize ,
                       const unsigned int tid ,
                       unsigned int* res  )
{
    __shared__ unsigned int reducedata[C_NUM_LISTS] ;
    __shared__ unsigned int result ;


    if(tid < C_NUM_LISTS)
    {
        reducedata[tid] = 0 ;
        result = 0 ;
        __syncthreads() ;

        for(int i =tid ; i< datasize;i+=C_NUM_LISTS)
        {
            unsigned int t = data[i]  ;
            atomicAdd(&(reducedata[tid]),t) ;
        }
        __syncthreads() ;

        atomicAdd(&result,reducedata[tid]) ;

        __syncthreads() ;
//        res[0] = result ;
        atomicAdd(res,result) ;
    }
}


__global__ void add(unsigned int *data, unsigned int datasize, unsigned int* res)
{
    const unsigned int  idx = (blockIdx.x* blockDim.x) + threadIdx.x ;
    const unsigned int idy = (blockIdx.y*blockDim.y) + threadIdx.y ;
    const unsigned int tid = idy*gridDim.x*blockDim.x + idx ;
    const unsigned int threadsize = gridDim.x*gridDim.y*gridDim.z*blockDim.x*blockDim.y*blockDim.z ;
//    addsum(data,datasize,threadsize,tid,res) ;
    addsum2(data,datasize,threadsize,tid,res) ;
}




int main(int argc, char** argv)
{

    findCudaDevice(argc,(const char **)argv) ;
    unsigned int* alldata = new unsigned int[C_NUM_ELEMENTS] ;
    int result_of_gpu = gen_and_shuffle(alldata) ;
    std::cout<<result_of_gpu <<std::endl;

    unsigned int* result1 = new unsigned int[1] ;

    unsigned int* data ;
    checkCudaErrors(cudaMalloc((void**)&data,C_NUM_ELEMENTS*sizeof(unsigned int))) ;
    checkCudaErrors(cudaMemcpy(data,alldata,C_NUM_ELEMENTS*sizeof(unsigned int),cudaMemcpyHostToDevice)) ;

    unsigned int* res ;
    checkCudaErrors(cudaMalloc((void**)&res,sizeof(unsigned int))) ;


    add<<<dim3(1,10),1000>>>(data,C_NUM_ELEMENTS,res) ;
//    add<<<dim3(1,10,1),dim3(100,10)>>>(data,C_NUM_ELEMENTS,res) ;

    unsigned int *shit = new unsigned int[1] ;

    checkCudaErrors(cudaMemcpy(shit,res,sizeof(unsigned int),cudaMemcpyDeviceToHost)) ;
    checkCudaErrors(cudaFree(data)) ;
    checkCudaErrors(cudaFree(res)) ;


    std::cout<<shit[0]<<std::endl;
}