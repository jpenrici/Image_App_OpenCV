cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(ImageToolsGui VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ImageTools Library
set(LIBRARY_NAME imagetools)
set(IMGTOOLS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../img_tools")
set(IMGTOOLS_INCLUDE_DIR "${IMGTOOLS_ROOT}/include")
set(IMGTOOLS_LIBRARY_DIR "${IMGTOOLS_ROOT}/lib")
set(IMGTOOLS_LIBRARY "${IMGTOOLS_LIBRARY_DIR}/lib${LIBRARY_NAME}.so")

if(NOT EXISTS "${IMGTOOLS_LIBRARY}")
    message(WARNING "\n[ERROR] lib${LIBRARY_NAME}.so not found in:\n  ${IMGTOOLS_LIBRARY}\n")
else()
    message(STATUS "[OK] Found lib${LIBRARY_NAME}.so in: ${IMGTOOLS_LIBRARY}")
endif()

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# QT6
set(QT_MIN_VERSION "6.5.0")
find_package(Qt6 ${QT_MIN_VERSION} COMPONENTS Widgets REQUIRED)

set(PROJECT_SOURCES
        main.cpp
        mainwindow.cpp
        mainwindow.h
        mainwindow.ui
        imagetoolsgui.hpp
        imagetoolsgui.cpp
)

set(QTGUI_EXEC_NAME imageTools_qt6)
qt_add_executable(${QTGUI_EXEC_NAME} ${PROJECT_SOURCES})

target_include_directories(${QTGUI_EXEC_NAME}
    PRIVATE
        ${IMGTOOLS_INCLUDE_DIR}
)

target_link_directories(${QTGUI_EXEC_NAME}
    PRIVATE
        ${IMGTOOLS_LIBRARY_DIR}
)

target_link_libraries(${QTGUI_EXEC_NAME}
    PRIVATE
        Qt6::Widgets
        ${LIBRARY_NAME}
        ${OpenCV_LIBS}
)

add_library(imgtools SHARED IMPORTED)
set_target_properties(imgtools PROPERTIES
    IMPORTED_LOCATION "${IMGTOOLS_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${IMGTOOLS_INCLUDE_DIR}"
)


include(GNUInstallDirs)

install(TARGETS ${QTGUI_EXEC_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES "${IMGTOOLS_LIBRARY}" DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY "${IMGTOOLS_INCLUDE_DIR}/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
