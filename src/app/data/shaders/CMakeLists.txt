# pretty dumb for now :)

# very ugly hack to add slang support until it's not officially supported by CMake's FindVulkan module
if (WIN32 AND NOT DEFINED Vulkan::slangc)
    add_executable(Vulkan::slangc IMPORTED)
    set_property(TARGET Vulkan::slangc PROPERTY IMPORTED_LOCATION "%VULKAN_SDK%/bin/slangc.exe")
endif()


add_custom_target(
    app_shaders
    COMMAND Vulkan::slangc -fvk-use-entrypoint-name "${CMAKE_CURRENT_SOURCE_DIR}/hello_world.slang" -o "hello_world.spv"
    DEPENDS
        "hello_world.slang"

    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Compile the shaders"
    VERBATIM
)
