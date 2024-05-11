cmake_minimum_required (VERSION 3.29)

add_executable (VulkanViewer
	"VulkanViewer.cpp"
	"VulkanViewer.h"
)

find_package(sdl2 REQUIRED)

target_link_libraries(VulkanViewer
	PRIVATE
		engine 
		SDL2::SDL2main
		SDL2::SDL2
)


add_subdirectory("data/shaders")
add_dependencies(VulkanViewer app_shaders)

target_compile_features(VulkanViewer PRIVATE cxx_std_23 )
