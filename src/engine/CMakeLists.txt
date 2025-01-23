cmake_minimum_required (VERSION 3.29)


find_package(sdl2 REQUIRED)

# Vulkan-related stuff
find_package(Vulkan REQUIRED)

# Require Vulkan version ≥ 1.3.256 (earliest version when the Vulkan module was available)
if( ${Vulkan_VERSION} VERSION_LESS "1.3.256" )
  message( FATAL_ERROR "Minimum required Vulkan version for C++ modules is 1.3.256. "
           "Found ${Vulkan_VERSION}."
  )
endif()

# set up Vulkan C++ module as a library
add_library( VulkanHppModule )
target_sources( VulkanHppModule PUBLIC
  FILE_SET CXX_MODULES
  BASE_DIRS ${Vulkan_INCLUDE_DIR}
  FILES ${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm
)

target_compile_definitions(VulkanHppModule
    PUBLIC
        VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
        VULKAN_HPP_NO_SMART_HANDLE
        VULKAN_HPP_NO_SPACESHIP_OPERATOR
        VULKAN_HPP_NO_TO_STRING
        VK_NO_PROTOTYPES
)

target_compile_features( VulkanHppModule PUBLIC cxx_std_23 )
target_link_libraries( VulkanHppModule PUBLIC Vulkan::Vulkan )
# end of Vulkan-related stuff


add_library(engine STATIC)
target_link_libraries(engine PRIVATE VulkanHppModule SDL2::SDL2)
target_compile_definitions(engine 
    PRIVATE
        $<$<CONFIG:Debug>:AT3_DEBUG>
)

target_sources(engine
    PRIVATE
        "engine.cpp"

    PUBLIC
        FILE_SET headers TYPE HEADERS
            FILES 
                
        FILE_SET partitions TYPE CXX_MODULES
            FILES
                "vulkan/context.cpp"
                "vulkan/framebuffer.cpp"
                "vulkan/renderer.cpp"
                "vulkan/sdl_context.cpp"

        FILE_SET files TYPE CXX_MODULES 
            BASE_DIRS
            FILES
                "engine.ixx"
                "utils.cpp"
)

set_target_properties(engine PROPERTIES
    CXX_EXTENSIONS off
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED on
)