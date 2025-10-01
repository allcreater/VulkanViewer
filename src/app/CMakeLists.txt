cmake_minimum_required (VERSION 3.29)

add_executable (VulkanViewer
	"VulkanViewer.cpp"
	
)

target_link_libraries(VulkanViewer
	PRIVATE
		engine 
		SDL3::SDL3
)


add_subdirectory("data/shaders")
add_dependencies(VulkanViewer app_shaders)

target_compile_features(VulkanViewer PRIVATE cxx_std_23 )
