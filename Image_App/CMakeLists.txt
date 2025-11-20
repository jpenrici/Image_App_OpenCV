cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(ImageTools_App LANGUAGES CXX)

# ------------------------------------------------------------
# Global settings
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output root directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# ============================================================
#  FIND PACKAGES (OpenCV + Qt6)
# ============================================================
find_package(OpenCV REQUIRED)
message(STATUS "[OK] OpenCV version: ${OpenCV_VERSION}")

find_package(Qt6 6.5 COMPONENTS Widgets REQUIRED)
message(STATUS "[OK] Qt6 found")

# ------------------------------------------------------------
# Add subprojects (independent)
# ------------------------------------------------------------

# 1. Image processing library
add_subdirectory(./img_tools)

# 2. Qt6 GUI application
add_subdirectory(./gui_qt6/ImageTools)

# ------------------------------------------------------------
# Status message
# ------------------------------------------------------------
message(STATUS "")
message(STATUS "==============================")
message(STATUS "  Image_App superbuild ready  ")
message(STATUS "==============================")
message(STATUS "Library output : ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "Binary output  : ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "")

# ------------------------------------------------------------
# Resources
# ------------------------------------------------------------

file(GLOB RESOURCE_FILES "resources/*")
file(COPY ${RESOURCE_FILES} DESTINATION ${CMAKE_BINARY_DIR}/resources)
