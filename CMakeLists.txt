SET(PROJECT_NAME NovoCloudSegmentation)
set(CMAKE_CXX_STANDARD 17)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED(VERSION 3.2)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common search kdtree features segmentation)


set(SRC NovoCloudSegmentation/main.cpp)
aux_source_directory("NovoCloudSegmentation/cloud" SRC)
aux_source_directory("NovoCloudSegmentation/3d_visualization" SRC)
aux_source_directory("NovoCloudSegmentation/utils" SRC)

add_executable(${PROJECT_NAME} ${SRC})


set(PCL_LIBS pcl_search pcl_kdtree pcl_common pcl_features pcl_segmentation)
set(Pangolin_LIBS pango_windowing pango_core pango_display pango_geometry pango_glgeometry pango_image pango_opengl pango_packetstream pango_plot pango_python pango_scene pango_tools pango_vars pango_video)

#link_directories(/usr/local/opt/glew/lib)
set(GLEW_LIBRARIES /usr/local/opt/glew/lib/libGLEW.a)

target_compile_definitions(${PROJECT_NAME} PUBLIC _OSX_)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Pangolin_LIBS} ${PCL_LIBS} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} ${Pangolin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS} /usr/local/include)

message("••• ${PCL_LIBS}")