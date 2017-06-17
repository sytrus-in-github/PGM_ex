cmake_minimum_required(VERSION 2.6)
project(in2329_exercise_07_supp)

#set(CMAKE_CXX_STANDARD 11)

SET(OpenCV_STATIC OFF)

FIND_PACKAGE(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES
        multilabel_demo.cpp)

add_executable(in2329_exercise_07_supp ${SOURCE_FILES})

IF(WIN32)
# SET(CMAKE_CXX_FLAGS "-lws2_32")
# -lws2_32 "ws2_32.lib"
TARGET_LINK_LIBRARIES(in2329_exercise_07_supp "ws2_32.lib")
ENDIF(WIN32)

TARGET_LINK_LIBRARIES(in2329_exercise_07_supp ${OpenCV_LIBS})
