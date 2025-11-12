cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(ImageToolsGui VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(QT_MIN_VERSION "6.5.0")
find_package(Qt6 ${QT_MIN_VERSION} COMPONENTS Widgets REQUIRED)

set(PROJECT_SOURCES
        main.cpp
        mainwindow.cpp
        mainwindow.h
        mainwindow.ui
)

set(QTGUI_EXEC_NAME imageTools_qt6)
qt_add_executable(${QTGUI_EXEC_NAME} ${PROJECT_SOURCES})

target_link_libraries(${QTGUI_EXEC_NAME} PRIVATE Qt6::Widgets)

include(GNUInstallDirs)
install(TARGETS ${QTGUI_EXEC_NAME}
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
