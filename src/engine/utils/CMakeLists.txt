add_library(utils STATIC)

target_sources(utils
PUBLIC
	FILE_SET utils_set TYPE CXX_MODULES FILES
		"utils.cpp"
		"gltf_utils.cpp"
)

set_property(TARGET utils PROPERTY CXX_STANDARD 23)

find_path(TINYGLTF_INCLUDE_DIRS "tiny_gltf.h")
target_include_directories(utils PRIVATE ${TINYGLTF_INCLUDE_DIRS})

# target_link_libraries(utils PUBLIC glm)

# add_subdirectory("Tests")