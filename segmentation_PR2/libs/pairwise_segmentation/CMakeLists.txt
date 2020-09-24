# Lets get a recet version of CMake
cmake_minimum_required(VERSION 3.1)

# Project name
project(pairwise_segmentation)

# Flags
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -ltbb")

# Crée des variables avec les fichiers à compiler
set(SRCS
  main.cpp
  # miscellaneous/main.cpp
  # miscellaneous/video_test.cpp
  # miscellaneous/persistent_world.cpp
  )

set(HEADERS pairwise_segmentation.hpp
  pairwise_segmentation.h)

# Notre exécutable
add_executable(${PROJECT_NAME} ${SRCS} ${HEADERS})

########################### PCL Library ##########################
#find_package (PCL 1.9 REQUIRED)

#if (PCL_FOUND)
#  # Une fois la dépendance trouvée, nous l'incluons au projet
#  target_include_directories(${PROJECT_NAME} PUBLIC ${PCL_INCLUDE_DIRS})
#  target_link_libraries (${PROJECT_NAME} ${PCL_LIBRARIES})
#else ()
#  # Sinon, nous affichons un message
#  message(FATAL_ERROR "PCL not found")
#endif ()

########################### TBB Library ##########################
#target_link_libraries(${PROJECT_NAME} tbb)

########################### OpenMP Library ##########################
# Recherche de la dépendance externe
#find_package(OpenMP)

#if (OPENMP_FOUND)
#  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#else()
#  #    Sinon, nous affichons un message
#  message(FATAL_ERROR "OpenMP not found")
#endif()

