cmake_minimum_required( VERSION 2.8 )

# Create Project
project( solution )

# On inclut notre bibliothèque dans le processus de CMake
add_subdirectory(libs)


set(SRCS
  main.cpp
  # miscellaneous/main.cpp
  # miscellaneous/video_test.cpp
  # miscellaneous/persistent_world.cpp
  )
  
  
set(HEADERS
  libs/pairwise_segmentation/pairwise_segmentation.h
  libs/pairwise_segmentation/pairwise_segmentation.hpp
  libs/image_processing/HistogramFactory.hpp
  libs/image_processing/tools.hpp)

add_executable( project main.cpp ${SRCS} ${HEADERS})

find_package(PCL 1.8 REQUIRED )




include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

MESSAGE(STATUS "PCL_VERSION: " ${PCL_VERSION})

target_link_libraries(project ${PCL_LIBRARIES})

#find cppzmq wrapper, installed by make of cppzmq
find_package(cppzmq)

# Link local libraries
target_link_libraries(project lib_papon)
target_link_libraries(project cppzmq tbb)
