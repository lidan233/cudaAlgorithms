# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "D:\software\clion\CLion 2020.2.3\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "D:\software\clion\CLion 2020.2.3\bin\cmake\win\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\lidan\CLionProjects\cudaSampler

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug

# Include any dependencies generated for this target.
include external\cxxopts\test\CMakeFiles\link_test.dir\depend.make

# Include the progress variables for this target.
include external\cxxopts\test\CMakeFiles\link_test.dir\progress.make

# Include the compile flags for this target's objects.
include external\cxxopts\test\CMakeFiles\link_test.dir\flags.make

external\cxxopts\test\CMakeFiles\link_test.dir\link_a.cpp.obj: external\cxxopts\test\CMakeFiles\link_test.dir\flags.make
external\cxxopts\test\CMakeFiles\link_test.dir\link_a.cpp.obj: ..\external\cxxopts\test\link_a.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object external/cxxopts/test/CMakeFiles/link_test.dir/link_a.cpp.obj"
	cd C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug\external\cxxopts\test
	C:\PROGRA~2\MICROS~1\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\link_test.dir\link_a.cpp.obj /FdCMakeFiles\link_test.dir\ /FS -c C:\Users\lidan\CLionProjects\cudaSampler\external\cxxopts\test\link_a.cpp
<<
	cd C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug

external\cxxopts\test\CMakeFiles\link_test.dir\link_a.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/link_test.dir/link_a.cpp.i"
	cd C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug\external\cxxopts\test
	C:\PROGRA~2\MICROS~1\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx64\x64\cl.exe > CMakeFiles\link_test.dir\link_a.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\lidan\CLionProjects\cudaSampler\external\cxxopts\test\link_a.cpp
<<
	cd C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug

external\cxxopts\test\CMakeFiles\link_test.dir\link_a.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/link_test.dir/link_a.cpp.s"
	cd C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug\external\cxxopts\test
	C:\PROGRA~2\MICROS~1\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\link_test.dir\link_a.cpp.s /c C:\Users\lidan\CLionProjects\cudaSampler\external\cxxopts\test\link_a.cpp
<<
	cd C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug

external\cxxopts\test\CMakeFiles\link_test.dir\link_b.cpp.obj: external\cxxopts\test\CMakeFiles\link_test.dir\flags.make
external\cxxopts\test\CMakeFiles\link_test.dir\link_b.cpp.obj: ..\external\cxxopts\test\link_b.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object external/cxxopts/test/CMakeFiles/link_test.dir/link_b.cpp.obj"
	cd C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug\external\cxxopts\test
	C:\PROGRA~2\MICROS~1\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\link_test.dir\link_b.cpp.obj /FdCMakeFiles\link_test.dir\ /FS -c C:\Users\lidan\CLionProjects\cudaSampler\external\cxxopts\test\link_b.cpp
<<
	cd C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug

external\cxxopts\test\CMakeFiles\link_test.dir\link_b.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/link_test.dir/link_b.cpp.i"
	cd C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug\external\cxxopts\test
	C:\PROGRA~2\MICROS~1\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx64\x64\cl.exe > CMakeFiles\link_test.dir\link_b.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\lidan\CLionProjects\cudaSampler\external\cxxopts\test\link_b.cpp
<<
	cd C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug

external\cxxopts\test\CMakeFiles\link_test.dir\link_b.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/link_test.dir/link_b.cpp.s"
	cd C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug\external\cxxopts\test
	C:\PROGRA~2\MICROS~1\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx64\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\link_test.dir\link_b.cpp.s /c C:\Users\lidan\CLionProjects\cudaSampler\external\cxxopts\test\link_b.cpp
<<
	cd C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug

# Object files for target link_test
link_test_OBJECTS = \
"CMakeFiles\link_test.dir\link_a.cpp.obj" \
"CMakeFiles\link_test.dir\link_b.cpp.obj"

# External object files for target link_test
link_test_EXTERNAL_OBJECTS =

external\cxxopts\test\link_test.exe: external\cxxopts\test\CMakeFiles\link_test.dir\link_a.cpp.obj
external\cxxopts\test\link_test.exe: external\cxxopts\test\CMakeFiles\link_test.dir\link_b.cpp.obj
external\cxxopts\test\link_test.exe: external\cxxopts\test\CMakeFiles\link_test.dir\build.make
external\cxxopts\test\link_test.exe: external\cxxopts\test\CMakeFiles\link_test.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable link_test.exe"
	cd C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug\external\cxxopts\test
	"D:\software\clion\CLion 2020.2.3\bin\cmake\win\bin\cmake.exe" -E vs_link_exe --intdir=CMakeFiles\link_test.dir --rc=C:\PROGRA~2\WI3CF2~1\10\bin\100183~1.0\x64\rc.exe --mt=C:\PROGRA~2\WI3CF2~1\10\bin\100183~1.0\x64\mt.exe --manifests  -- C:\PROGRA~2\MICROS~1\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx64\x64\link.exe /nologo @CMakeFiles\link_test.dir\objects1.rsp @<<
 /out:link_test.exe /implib:link_test.lib /pdb:C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug\external\cxxopts\test\link_test.pdb /version:0.0    /debug /INCREMENTAL /subsystem:console   -LIBPATH:C:\Users\lidan\CLionProjects\cudaSampler\external\glfw\lib-vc2019  kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib 
<<
	cd C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug

# Rule to build all files generated by this target.
external\cxxopts\test\CMakeFiles\link_test.dir\build: external\cxxopts\test\link_test.exe

.PHONY : external\cxxopts\test\CMakeFiles\link_test.dir\build

external\cxxopts\test\CMakeFiles\link_test.dir\clean:
	cd C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug\external\cxxopts\test
	$(CMAKE_COMMAND) -P CMakeFiles\link_test.dir\cmake_clean.cmake
	cd C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug
.PHONY : external\cxxopts\test\CMakeFiles\link_test.dir\clean

external\cxxopts\test\CMakeFiles\link_test.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" C:\Users\lidan\CLionProjects\cudaSampler C:\Users\lidan\CLionProjects\cudaSampler\external\cxxopts\test C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug\external\cxxopts\test C:\Users\lidan\CLionProjects\cudaSampler\cmake-build-debug\external\cxxopts\test\CMakeFiles\link_test.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : external\cxxopts\test\CMakeFiles\link_test.dir\depend

