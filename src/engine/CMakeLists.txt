cmake_minimum_required (VERSION 3.29)

add_subdirectory("utils")

add_library(engine STATIC)
target_link_libraries(engine PRIVATE VulkanHppModule SDL3::SDL3 GPUOpen::VulkanMemoryAllocator glm::glm utils fastgltf::module)
target_compile_definitions(engine
    PRIVATE
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_FORCE_LEFT_HANDED
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
                "vulkan/memory_allocator.cpp"
                "vulkan/renderer.cpp"
                "vulkan/resource_factory.cpp"
                "vulkan/sdl_context.cpp"

        FILE_SET files TYPE CXX_MODULES 
            BASE_DIRS
            FILES
                "engine.ixx"
)

set_target_properties(engine PROPERTIES
    CXX_EXTENSIONS off
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED on
)