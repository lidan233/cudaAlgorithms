//
// Created by lidan on 27/09/2020.
//

#ifndef CUDASAMPLER_GL_HELPER_CUH
#define CUDASAMPLER_GL_HELPER_CUH


/*
   On 64-bit Windows, we need to prevent GLUT from automatically linking against
   glut32. We do this by defining GLUT_NO_LIB_PRAGMA. This means that we need to
   manually add opengl32.lib and glut64.lib back to the link using pragmas.
   Alternatively, this could be done on the compilation/link command-line, but
   we chose this so that compilation is identical between 32- and 64-bit Windows.
*/
#ifdef _WIN64
#define GLUT_NO_LIB_PRAGMA
#include <glad.h>
#include <GLFW/glfw3.h>
#pragma comment (lib, "opengl32.lib")  /* link with Microsoft OpenGL lib */
//#pragma comment (lib, "glut64.lib")    /* link with Win64 GLUT lib */
#endif //_WIN64

#ifdef _WIN32
#include <glad.h>
#include <GLFW/glfw3.h>
#define GET_PROC_ADDRESS( str ) wglGetProcAddress( str )

#else

/* On Linux, include the system's copy of glut.h, glext.h, and glx.h */
#include <GLFW/glfw3.h>
#define GET_PROC_ADDRESS( str ) glXGetProcAddress( (const GLubyte *)str )

#endif //_WIN32



#endif //CUDASAMPLER_GL_HELPER_CUH
