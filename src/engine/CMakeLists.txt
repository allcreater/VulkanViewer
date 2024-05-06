cmake_minimum_required (VERSION 3.29)


find_package(sdl2 REQUIRED)
# Vulkan-related stuff
find_package(Vulkan REQUIRED)

#[[
# Require Vulkan version ≥ 1.3.256 (earliest version when the Vulkan module was available)
if( ${Vulkan_VERSION} VERSION_LESS "1.3.256" )
  message( FATAL_ERROR "Minimum required Vulkan version for C++ modules is 1.3.256. "
           "Found ${Vulkan_VERSION}."
  )
endif()

# set up Vulkan C++ module as a library
add_library( VulkanHppModule )
target_sources( VulkanHppModule PRIVATE
  FILE_SET CXX_MODULES
  BASE_DIRS ${Vulkan_INCLUDE_DIR}
  FILES ${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm
)

target_compile_definitions(VulkanHppModule PUBLIC 
    #VULKAN_HPP_NO_SMART_HANDLE
    #VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
)

target_compile_features( VulkanHppModule PUBLIC cxx_std_23 )
target_link_libraries( VulkanHppModule PUBLIC Vulkan::Vulkan )
# end of Vulkan-related stuff
#]]


add_library(engine)
#target_link_libraries(engine PRIVATE VulkanHppModule )
target_link_libraries(engine PRIVATE Vulkan::Vulkan SDL2::SDL2)
target_compile_definitions(engine 
    PRIVATE
        VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
        $<$<CONFIG:Debug>:AT3_DEBUG>
)

target_sources(engine
    PRIVATE
        "engine.cpp"

    PUBLIC
        FILE_SET headers TYPE HEADERS
            FILES 
                
        FILE_SET files TYPE CXX_MODULES 
            BASE_DIRS
            FILES
                "vulkan/context.cpp"
                "vulkan/sdl_context.cpp"
                "engine.ixx"
                "utils.cpp"
)

target_compile_features(engine PUBLIC cxx_std_23 )
set_target_properties(engine PROPERTIES
    CXX_EXTENSIONS off
    CXX_STANDARD_REQUIRED on
)