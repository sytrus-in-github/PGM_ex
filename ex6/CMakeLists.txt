CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(OpenCV_STATIC OFF)

FIND_PACKAGE(OpenCV REQUIRED)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

PROJECT(ex6)

ADD_EXECUTABLE(main main.cpp ../maxflow/graph.cpp ../maxflow/maxflow.cpp)

TARGET_LINK_LIBRARIES(main ${OpenCV_LIBS})





