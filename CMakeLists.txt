# drishti: LiDAR perception on matrix, with SalsaNext in MLX, and a demo that plays SemanticKITTI.
#
#   git clone --recursive https://github.com/AlanAditya/drishti-lidar.git && cd drishti-lidar
#   cmake -S . -B build && cmake --build build
#   ./build/drishti <sequence dir> <salsanext.safetensors>
cmake_minimum_required(VERSION 3.20)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

project(drishti LANGUAGES CXX OBJCXX)

option(DRISHTI_DEMO "Build the demo (main.cpp) and the renderer it draws with" ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -fno-math-errno")
set(CMAKE_OBJCXX_FLAGS_RELEASE "-O3 -DNDEBUG -fno-math-errno")
# Executables next to default.metallib (below), where matrix's GPUManager looks for its kernels.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/matrix/CMakeLists.txt)
    message(FATAL_ERROR "matrix/ is empty: clone with --recursive, or run `git submodule update --init`")
endif()
add_subdirectory(matrix)
target_compile_options(matrix PRIVATE -fobjc-arc)

find_package(MLX CONFIG REQUIRED)
find_program(XCRUN_EXECUTABLE xcrun REQUIRED)

# --- One metallib with matrix's kernels and drishti's (drishti/FoveatedGrid.metal: gpu_func looks its
# kernels up in the same library), next to the executables.
file(GLOB DRISHTI_KERNELS ${CMAKE_CURRENT_SOURCE_DIR}/matrix/ComputeShaders/*.metal ${CMAKE_CURRENT_SOURCE_DIR}/drishti/*.metal)
set(DRISHTI_AIR "")
foreach(src ${DRISHTI_KERNELS})
    get_filename_component(name ${src} NAME_WE)
    set(air ${CMAKE_BINARY_DIR}/kernels_air/${name}.air)
    add_custom_command(
        OUTPUT ${air}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/kernels_air
        COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal -c ${src} -I ${CMAKE_CURRENT_SOURCE_DIR}/matrix/ComputeShaders -o ${air}
        DEPENDS ${src}
        COMMENT "metal: compiling ${name}.metal"
        VERBATIM)
    list(APPEND DRISHTI_AIR ${air})
endforeach()
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/default.metallib
    COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib ${DRISHTI_AIR} -o ${CMAKE_BINARY_DIR}/default.metallib
    DEPENDS ${DRISHTI_AIR}
    COMMENT "metallib: linking default.metallib (matrix + drishti kernels)"
    VERBATIM)
add_custom_target(drishti_metallib ALL DEPENDS ${CMAKE_BINARY_DIR}/default.metallib)

# --- drishti itself: drishti/drishti.cpp and the headers it uses, included by your one source file
# (#include "drishti.cpp": linking this target puts drishti/ on the include path).
add_library(drishti INTERFACE)
target_include_directories(drishti INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/drishti ${MLX_INCLUDE_DIRS})
target_link_libraries(drishti INTERFACE matrix mlx)
add_dependencies(drishti drishti_metallib)

if(DRISHTI_DEMO)
    # --- The renderer and its scene graph, camera, and shaders live under renderer/.
    # Its shaders go into their own render.metallib.
    set(RENDER_AIR "")
    foreach(name Shaders DAGShaders)
        set(air ${CMAKE_BINARY_DIR}/render_air/${name}.air)
        add_custom_command(
            OUTPUT ${air}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/render_air
            COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/renderer/${name}.metal -o ${air}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/renderer/${name}.metal
            COMMENT "metal: compiling ${name}.metal"
            VERBATIM)
        list(APPEND RENDER_AIR ${air})
    endforeach()
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/render.metallib
        COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib ${RENDER_AIR} -o ${CMAKE_BINARY_DIR}/render.metallib
        DEPENDS ${RENDER_AIR}
        COMMENT "metallib: linking render.metallib"
        VERBATIM)
    add_custom_target(render_metallib DEPENDS ${CMAKE_BINARY_DIR}/render.metallib)

    add_library(renderer STATIC renderer/renderer.mm)
    add_dependencies(renderer render_metallib)
    # The renderer uses the matrix submodule for tensor operations.
    target_include_directories(renderer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/renderer ${CMAKE_CURRENT_SOURCE_DIR}
                               ${CMAKE_CURRENT_SOURCE_DIR}/matrix/Mods)
    target_compile_options(renderer PRIVATE -fobjc-arc)
    target_compile_definitions(renderer PRIVATE RENDER_METALLIB_PATH="${CMAKE_BINARY_DIR}/render.metallib")
    target_link_libraries(renderer PUBLIC matrix
        "-framework Cocoa" "-framework MetalKit" "-framework QuartzCore" "-framework ModelIO" "-framework CoreText")

    # --- the demo. matrix.h uses @import and Metal, so the .cpp is compiled as Objective-C++.
    add_executable(drishti_demo main.cpp)
    set_target_properties(drishti_demo PROPERTIES OUTPUT_NAME drishti)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE OBJCXX)
    target_compile_options(drishti_demo PRIVATE -fobjc-arc)
    target_include_directories(drishti_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(drishti_demo PRIVATE drishti renderer)
endif()
