add_library(utils STATIC)

target_sources(utils
PUBLIC
	FILE_SET utils_set TYPE CXX_MODULES FILES
		"utils.cpp"
)

set_property(TARGET utils PROPERTY CXX_STANDARD 23)

# target_link_libraries(utils PUBLIC glm)

# add_subdirectory("Tests")