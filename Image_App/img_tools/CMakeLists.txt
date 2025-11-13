cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(ImageTools LANGUAGES CXX)

set(LIBRARY_NAME imagetools)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found! Check your installation!")
endif()

include_directories(include)

add_library(${LIBRARY_NAME} SHARED
    src/imgtools.cpp
)

set_target_properties(${LIBRARY_NAME} PROPERTIES
    OUTPUT_NAME ${LIBRARY_NAME}
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
)

target_link_libraries(${LIBRARY_NAME} PUBLIC ${OpenCV_LIBS})

# ===== Test =====
set(TEST ${LIBRARY_NAME}_test)

add_executable(${TEST} src/test.cpp)

target_link_libraries(${TEST} PRIVATE ${LIBRARY_NAME} ${OpenCV_LIBS})
