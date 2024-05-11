# pretty dumb for now :)

if (WIN32)
    set(APP_GLSLC_COMMAND "%VULKAN_SDK%/bin/glslc.exe")
endif()

add_custom_target(
    app_shaders
    COMMAND ${APP_GLSLC_COMMAND} "-fshader-stage=vert" "${CMAKE_CURRENT_SOURCE_DIR}/hello_world.vs.glsl" -o "hello_world.vs.spv"
    COMMAND ${APP_GLSLC_COMMAND} "-fshader-stage=frag" "${CMAKE_CURRENT_SOURCE_DIR}/hello_world.fs.glsl" -o "hello_world.fs.spv"
    DEPENDS
        "hello_world.vs.glsl"
        "hello_world.fs.glsl"

    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Compile the shaders"
    VERBATIM
)