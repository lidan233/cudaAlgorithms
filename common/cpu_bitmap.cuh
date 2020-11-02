//
// Created by lidan on 27/09/2020.
//

#ifndef CUDASAMPLER_CPU_BITMAP_CUH
#define CUDASAMPLER_CPU_BITMAP_CUH

#include "gl_helper.cuh"
#include "shader.h"
#include <cuda.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>
#include <glad.h>
#include <GLFW/glfw3.h>

//#include "glad/glad.h"
//#include "GLFW/glfw3.h"
#define SCR_HEIGHT 1024
#define SCR_WIDTH 1024
float scale = 1.5 ;
cudaGraphicsResource_t viewCudaResource;


void framebuffer_size_callback(GLFWwindow *window,int height,int width)
{
    glViewport(0,0,width,height);
}
void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window,GLFW_KEY_ESCAPE)==GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window,true) ;
        glfwTerminate();
    }

    if(glfwGetKey(window,GLFW_KEY_A)==GLFW_PRESS)
    {
        scale += 0.01 ;
    }

    if(glfwGetKey(window,GLFW_KEY_B)==GLFW_PRESS)
    {
        scale-=0.01 ;
    }


}

struct CPUBitmap {
    unsigned char    *pixels;
    int     x, y;
    void    *dataBlock;
    void (*bitmapExit)(void*);

    CPUBitmap( int width, int height, void *d = NULL ) {
        pixels = new unsigned char[width * height * 4];
        x = width;
        y = height;
        dataBlock = d;
    }

    ~CPUBitmap() {
        delete [] pixels;
    }

    unsigned char* get_ptr( void ) const   { return pixels; }
    long image_size( void ) const { return x * y * 4; }

    void display_and_exit( void(*e)(void*) = NULL ) {
        CPUBitmap**   bitmap = get_bitmap_ptr();
        *bitmap = this;
        unsigned char* data = reinterpret_cast<unsigned char *>(dataBlock) ;
        for(int i = 0; i< x;i++)
        {
            for(int j = 0 ; j< y;j++)
            {
                for(int k = 0 ; k < 4; k++)
                {
                    pixels[(i*y+j)*4+k] = data[(i*y+j)*4+k] ;
                }
            }
        }

        glfwInit() ;
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3) ;
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3) ;
        glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE) ;


#ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,GL_TRUE) ;
#endif


        GLFWwindow* window = glfwCreateWindow(x,y,"shit",nullptr,nullptr) ;
        if(window==NULL)
        {
            std::cout<<"Failed to create glfw window"<<std::endl ;
            glfwTerminate() ;
            return  ;
        }

        glfwMakeContextCurrent(window) ;
        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        {
            std::cout << "Failed to initialize GLAD" << std::endl;
            return ;
        }



        CPUBitmap*  bitmapdata = *(get_bitmap_ptr());

        // set up vertex data (and buffer(s)) and configure vertex attributes
        // ------------------------------------------------------------------
        float vertices[] = {
                // positions          // colors           // texture coords
                1.0f,  1.0f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
                1.0f, -1.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
                -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
                -1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left
        };
        unsigned int indices[] = {
                0, 1, 3, // first triangle
                1, 2, 3  // second triangle
        };

        unsigned int VBO, VAO, EBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        // position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        // color attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        // texture coord attribute
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);


        GLuint texture ;
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, x, y, 0, GL_RGBA, GL_UNSIGNED_BYTE,bitmapdata->pixels);
        glGenerateMipmap(GL_TEXTURE_2D);

        cudaGraphicsGLRegisterImage(&viewCudaResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard) ;



        Shader ourShader(vs, fs);
        ourShader.use(); // don't forget to activate/use the shader before setting uniforms!


        while(!glfwWindowShouldClose(window))
        {
            processInput(window);



            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture);



            ourShader.use();
            glBindVertexArray(VAO);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

            // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
            // -------------------------------------------------------------------------------
            glfwSwapBuffers(window);
            glfwPollEvents();

            e(bitmapdata->pixels) ;
            glTexSubImage2D(GL_TEXTURE_2D,0,0,0,x,y,GL_RGBA,GL_UNSIGNED_BYTE,bitmapdata->pixels) ;

        }

        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);



        bitmapdata->bitmapExit( bitmapdata->dataBlock );
        glfwTerminate() ;


    }

    // static method used for glut callbacks
    static CPUBitmap** get_bitmap_ptr( void ) {
        static CPUBitmap   *gBitmap;
        return &gBitmap;
    }

};

struct DataBlock {
    unsigned char   *dev_bitmap;
};




#endif //CUDASAMPLER_CPU_BITMAP_CUH
