cmake_minimum_required (VERSION 3.29)

include(FetchContent)

FetchContent_Declare(
  fastgltf
  GIT_REPOSITORY https://github.com/spnda/fastgltf.git
  GIT_TAG        v0.9.0
  EXCLUDE_FROM_ALL
  CMAKE_CACHE_ARGS
)
set(FASTGLTF_COMPILE_AS_CPP20 ON CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_CPP_MODULES ON CACHE BOOL "" FORCE)
set(FASTGLTF_USE_STD_MODULE ON CACHE BOOL "" FORCE)


FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG        1.0.1
  EXCLUDE_FROM_ALL
)
set(GLM_ENABLE_CXX_20 ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  SDL3
  GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
  GIT_TAG        release-3.2.22
  EXCLUDE_FROM_ALL
)
set(SDL_VULKAN ON CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  VulkanMemoryAllocator
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
  GIT_TAG        v3.3.0
  EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(fastgltf glm SDL3 VulkanMemoryAllocator)

find_package(Vulkan REQUIRED)

# Vulkan-related stuff
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
        VULKAN_HPP_STD_MODULE=std
        VK_NO_PROTOTYPES
)

target_compile_features( VulkanHppModule PUBLIC cxx_std_23 )
target_link_libraries( VulkanHppModule PUBLIC Vulkan::Vulkan )
# end of Vulkan-related stuff
