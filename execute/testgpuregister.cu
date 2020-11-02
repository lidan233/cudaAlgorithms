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

#define DATA 100000000

//cxxopts::ParseResult parse_option(int argc, char** argv)
//{
//    cxxopts::Options options("cuda samples","This is a simple practice of cuda by lidan233.") ;
//    options.add_options()
//            ("d,debug", "Enable debugging") // a bool parameter
//            ("i,integer", "Int param", cxxopts::value<int>())
//            ("if,file", "Input file name", cxxopts::value<std::string>())
//            ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"));
//    cxxopts::ParseResult result = options.parse(argc,argv) ;
//    return result ;
//}


__global__ void test_gpu_register(unsigned int* const data, unsigned int* packed, unsigned int num_elements)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x ;
    if(tid < num_elements)
    {
        unsigned int d_tmp = 0 ;
        for(int i = 0 ; i< 8; i++)
        {
            d_tmp |= (packed[i] << i) ;
        }
        data[tid] = d_tmp ;
    }
}

__device__ static uint32_t t_tmp = 0 ;

__global__ void test_gpu_gmem(uint32_t* const data, unsigned int* packed ,const uint32_t num_elements)
{
    const uint32_t tid = (blockIdx.x*blockDim.x) + threadIdx.x ;
    if(tid < num_elements)
    {
        for(int i = 0 ; i< 8; i++)
        {
            t_tmp |= (packed[i] << i) ;
        }
        data[tid] = t_tmp ;
    }
}

int main(int argc, char** argv)
{
//    cxxopts::ParseResult op = parse_option(argc,argv) ;
//    if (op.count("help"))
//    {
//        std::cout<<"shit" ;
//        exit(-1);
//    }

    bool debug = false ;



    srand(time(NULL)) ;
    unsigned int res = time(NULL) ;

    unsigned int* gpudata = NULL ;
    unsigned int* gpuparalleldata = NULL ;

    unsigned int* data_new = new unsigned int[DATA] ;
    unsigned int* parallel_data = new unsigned int[DATA] ;


#pragma omp parallel for
    for(int i = 0; i<DATA; i++)
    {
        data_new[i] = rand() %10 ;
        parallel_data[i] = rand()%10 ;
    }
    checkCudaErrors(cudaMalloc((void**)&gpudata,DATA*sizeof(unsigned int))) ;
    checkCudaErrors(cudaMalloc((void**)&gpuparalleldata,DATA*sizeof(unsigned int))) ;
    checkCudaErrors(cudaMemcpy(gpudata,data_new,DATA*sizeof(unsigned int),cudaMemcpyHostToDevice)) ;
    checkCudaErrors(cudaMemcpy(gpuparalleldata,parallel_data,DATA*sizeof(unsigned int),cudaMemcpyHostToDevice)) ;



    if(debug)
    {
//        unsigned int begindata = time(NULL) ;
//        test_gpu_register<<<2000,2000,1024>>>(gpudata,gpuparalleldata,DATA) ;
//        unsigned int enddata = time(NULL) ;
//        std::cout<<"this is a time line for all data"<<enddata-begindata<<std::endl ;
        unsigned int begindata =clock();;
        test_gpu_register<<<2000,2000,1024>>>(gpudata,gpuparalleldata,DATA) ;
        checkCudaErrors( cudaMemcpy( data_new, gpuparalleldata, DATA * sizeof(unsigned int),
                                     cudaMemcpyDeviceToHost ) );
        std::cout<<data_new[0]<<std::endl ;
        unsigned int enddata = clock() ;
        std::cout<<"this is a time line for all data "<<(double)(enddata-begindata)/CLOCKS_PER_SEC<<std::endl ;
    }else{
        unsigned int begindata =clock();;
        test_gpu_gmem<<<2000,2000,1024>>>(gpudata,gpuparalleldata,DATA) ;
        checkCudaErrors( cudaMemcpy( data_new, gpuparalleldata, DATA * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost ) );
        std::cout<<data_new[0]<<std::endl ;
        unsigned int enddata = clock() ;

        std::cout<<"this is a time line for all data "<<(double)(enddata-begindata)/CLOCKS_PER_SEC<<std::endl ;
    }

}
