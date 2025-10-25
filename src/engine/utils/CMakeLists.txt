add_library(utils STATIC)

target_sources(utils
PUBLIC
	FILE_SET utils_set TYPE CXX_MODULES FILES
		"utils.cpp"
		"stb_image_loader.cpp"
)

set_property(TARGET utils PROPERTY CXX_STANDARD 23)
target_compile_definitions(utils PRIVATE
	STBI_NO_BMP
	STBI_NO_TGA
	STBI_NO_PSD
	STBI_NO_PIC
	STBI_NO_GIF
	STBI_NO_PNM
)

target_link_libraries(utils PRIVATE stb_image)

# add_subdirectory("Tests")