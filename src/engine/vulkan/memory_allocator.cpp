module;
// #include <cassert>
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

export module engine:vulkan.memory_allocator;

import vulkan_hpp;
import std;

import utils.core;

export using VmaError = std::runtime_error;

export enum class CasualUsage { Auto, AutoMapped, AutoPreferDevice };

export VmaAllocationInfo;

export template <typename Resource>
struct AllocatedResource {
    Resource          resource;
    VmaAllocation     allocation;
    VmaAllocationInfo allocationInfo;
};

export class VulkanMemoryAllocator : private MoveOnly {
public:
    VulkanMemoryAllocator(const vk::raii::Instance& instance, vk::PhysicalDevice physicalDevice, const vk::raii::Device& device);
    ~VulkanMemoryAllocator();

    VulkanMemoryAllocator(VulkanMemoryAllocator&& rhv) noexcept;
    VulkanMemoryAllocator& operator=(VulkanMemoryAllocator&& rhv) noexcept;

    AllocatedResource<vk::Image>  Create(vk::ImageCreateInfo imageInfo, CasualUsage usage);
    AllocatedResource<vk::Buffer> Create(vk::BufferCreateInfo bufferInfo, CasualUsage usage);

    void Destroy(const AllocatedResource<vk::Image>& resource);
    void Destroy(const AllocatedResource<vk::Buffer>& resource);

private:
    VmaAllocator m_allocator;
};


// Implementation
namespace
{
VmaVulkanFunctions GetVmaDispatcher(const vk::raii::InstanceDispatcher* instanceDispatcher, const vk::raii::DeviceDispatcher* deviceDispatcher) {
    if (!instanceDispatcher || !deviceDispatcher)
        throw VmaError("nullptr dispatchers");

    return {
        .vkGetInstanceProcAddr               = instanceDispatcher->vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr                 = deviceDispatcher->vkGetDeviceProcAddr,
        .vkGetPhysicalDeviceProperties       = instanceDispatcher->vkGetPhysicalDeviceProperties,
        .vkGetPhysicalDeviceMemoryProperties = instanceDispatcher->vkGetPhysicalDeviceMemoryProperties,
        .vkAllocateMemory                    = deviceDispatcher->vkAllocateMemory,
        .vkFreeMemory                        = deviceDispatcher->vkFreeMemory,
        .vkMapMemory                         = deviceDispatcher->vkMapMemory,
        .vkUnmapMemory                       = deviceDispatcher->vkUnmapMemory,
        .vkFlushMappedMemoryRanges           = deviceDispatcher->vkFlushMappedMemoryRanges,
        .vkInvalidateMappedMemoryRanges      = deviceDispatcher->vkInvalidateMappedMemoryRanges,
        .vkBindBufferMemory                  = deviceDispatcher->vkBindBufferMemory,
        .vkBindImageMemory                   = deviceDispatcher->vkBindImageMemory,
        .vkGetBufferMemoryRequirements       = deviceDispatcher->vkGetBufferMemoryRequirements,
        .vkGetImageMemoryRequirements        = deviceDispatcher->vkGetImageMemoryRequirements,
        .vkCreateBuffer                      = deviceDispatcher->vkCreateBuffer,
        .vkDestroyBuffer                     = deviceDispatcher->vkDestroyBuffer,
        .vkCreateImage                       = deviceDispatcher->vkCreateImage,
        .vkDestroyImage                      = deviceDispatcher->vkDestroyImage,
        .vkCmdCopyBuffer                     = deviceDispatcher->vkCmdCopyBuffer,
#if VMA_DEDICATED_ALLOCATION || VMA_VULKAN_VERSION >= 1001000
        .vkGetBufferMemoryRequirements2KHR = deviceDispatcher->vkGetBufferMemoryRequirements2,
        .vkGetImageMemoryRequirements2KHR  = deviceDispatcher->vkGetImageMemoryRequirements2,
#endif
#if VMA_BIND_MEMORY2 || VMA_VULKAN_VERSION >= 1001000
        .vkBindBufferMemory2KHR = deviceDispatcher->vkBindBufferMemory2,
        .vkBindImageMemory2KHR  = deviceDispatcher->vkBindImageMemory2,
#endif
#if VMA_MEMORY_BUDGET || VMA_VULKAN_VERSION >= 1001000
        .vkGetPhysicalDeviceMemoryProperties2KHR = instanceDispatcher->vkGetPhysicalDeviceMemoryProperties2,
#endif
#if VMA_VULKAN_VERSION >= 1003000
        .vkGetDeviceBufferMemoryRequirements = deviceDispatcher->vkGetDeviceBufferMemoryRequirements,
        .vkGetDeviceImageMemoryRequirements  = deviceDispatcher->vkGetDeviceImageMemoryRequirements,
#endif
    };
}

VmaAllocationCreateInfo SelectAllocationCreateInfo(CasualUsage casualUsage) {
    VmaAllocationCreateInfo allocationCreateInfo{
        .flags          = 0,
        .usage          = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags  = 0,
        .preferredFlags = 0,
        .memoryTypeBits = 0,
        .pool           = nullptr,
        .pUserData      = nullptr,
        .priority       = 1.0f,
    };

    switch (casualUsage) {
        case CasualUsage::AutoMapped:
            allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
            break;
        case CasualUsage::AutoPreferDevice:
            allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            break;
        default:
            break;
    }

    return allocationCreateInfo;
}

template <typename Resource, typename ResourceCreateInfo, auto factoryFunction>
AllocatedResource<Resource> CreateResource(VmaAllocator allocator, const ResourceCreateInfo& createInfo, CasualUsage usage) {
    const VmaAllocationCreateInfo allocationCreateInfo = SelectAllocationCreateInfo(usage);
    const auto                    info                 = std::bit_cast<typename ResourceCreateInfo::NativeType>(createInfo);


    typename Resource::NativeType outResource;
    AllocatedResource<Resource>   result;
    if (factoryFunction(allocator, &info, &allocationCreateInfo, &outResource, &result.allocation, &result.allocationInfo) != VK_SUCCESS)
        throw VmaError("can not create resource");

    result.resource = outResource;
    return result;
}
} // namespace


VulkanMemoryAllocator::VulkanMemoryAllocator(const vk::raii::Instance& instance, vk::PhysicalDevice physicalDevice, const vk::raii::Device& device) {
    const auto functions = GetVmaDispatcher(instance.getDispatcher(), device.getDispatcher());

    const VmaAllocatorCreateInfo createInfo{
        .flags                       = {},
        .physicalDevice              = physicalDevice,
        .device                      = *device,
        .preferredLargeHeapBlockSize = 0,
        .pAllocationCallbacks        = nullptr,
        .pDeviceMemoryCallbacks      = nullptr,
        .pHeapSizeLimit              = nullptr,
        .pVulkanFunctions            = &functions,
        .instance                    = *instance,
        .vulkanApiVersion            = VK_API_VERSION_1_3,
    };

    if (vmaCreateAllocator(&createInfo, &m_allocator) != VK_SUCCESS)
        throw VmaError("failed to initialize Vulkan Memory Allocator");
}

VulkanMemoryAllocator::~VulkanMemoryAllocator() {
    if (m_allocator == nullptr)
        return;

    vmaDestroyAllocator(m_allocator);
}

VulkanMemoryAllocator::VulkanMemoryAllocator(VulkanMemoryAllocator&& rhv) noexcept : m_allocator(std::exchange(rhv.m_allocator, nullptr)) {}

VulkanMemoryAllocator& VulkanMemoryAllocator::operator=(VulkanMemoryAllocator&& rhv) noexcept {
    VulkanMemoryAllocator temp{std::move(rhv)};
    std::swap(*this, temp);

    return *this;
}

AllocatedResource<vk::Image> VulkanMemoryAllocator::Create(vk::ImageCreateInfo imageInfo, CasualUsage usage) {
    return CreateResource<vk::Image, vk::ImageCreateInfo, vmaCreateImage>(m_allocator, imageInfo, usage);
}


AllocatedResource<vk::Buffer> VulkanMemoryAllocator::Create(vk::BufferCreateInfo bufferInfo, CasualUsage usage) {
    return CreateResource<vk::Buffer, vk::BufferCreateInfo, vmaCreateBuffer>(m_allocator, bufferInfo, usage);
}

void VulkanMemoryAllocator::Destroy(const AllocatedResource<vk::Buffer>& resource) {
    vmaDestroyBuffer(m_allocator, resource.resource, resource.allocation);
}

void VulkanMemoryAllocator::Destroy(const AllocatedResource<vk::Image>& resource) {
    vmaDestroyImage(m_allocator, resource.resource, resource.allocation);
}
