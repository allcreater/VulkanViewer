module;
#include <cassert>
#include <vk_mem_alloc.h>

export module engine:vulkan.resource_factory;

import vulkan_hpp;
import std;

import utils.core;
import :vulkan.memory_allocator;

export class ResourceFactory : MoveOnly {
public:
    // NOTE: Resources lifetimes are actually bound to the ResourceFactory, but may be released automatically if they are not used
    template <typename T>
    class Handle : private std::shared_ptr<void> {
    public:
        friend class ResourceFactory;

        T operator*() const noexcept;
        operator bool() const noexcept;
        std::span<std::byte> mappedMemory() const;

    private:
        using std::shared_ptr<void>::shared_ptr;
    };

public:
    ResourceFactory(const vk::raii::Instance& instance, vk::PhysicalDevice physicalDevice, const vk::raii::Device& device);
    ~ResourceFactory();

    ResourceFactory(ResourceFactory&&);
    ResourceFactory& operator=(ResourceFactory&&);

    Handle<vk::Buffer> CreateBuffer(const vk::BufferCreateInfo& createInfo, CasualUsage usage);
    Handle<vk::Image> CreateImage(const vk::ImageCreateInfo& createInfo, CasualUsage usage);

    void FreeUnusedResources();

private:
    template <typename T, typename ... Args>
    ResourceFactory::Handle<T> CreateResource(Args&& ... args);

private:
    VulkanMemoryAllocator m_resourceAllocator;

    std::shared_ptr<ResourceFactory*> m_sharedSelf = std::make_shared<ResourceFactory*>(this);
    // std::vector<Handle<vk::Buffer>> m_usedBuffers;

    uint64_t m_currentTick = 0;
    struct DelayedDeleter {
        uint64_t                                    invalidationTimestamp;
        std::function<void(VulkanMemoryAllocator&)> deleter;
    };
    std::deque<DelayedDeleter> m_deleters;
};


// Implementation

template <typename Resource>
struct ResourceTraits {
    using InternalType  = AllocatedResource<Resource>;
    using SharedPtrType = std::shared_ptr<InternalType>;
    using HandleType    = ResourceFactory::Handle<Resource>;
};

template <typename Resource>
Resource ResourceFactory::Handle<Resource>::operator*() const noexcept {
    const auto& actualData = *std::static_pointer_cast<typename ResourceTraits<Resource>::InternalType>(*this);
    return actualData.resource;
}

template <typename Resource>
ResourceFactory::Handle<Resource>::operator bool() const noexcept {
    return std::shared_ptr<void>::operator bool();
}

template <typename Resource>
std::span<std::byte> ResourceFactory::Handle<Resource>::mappedMemory() const {
    const auto& actualData = *std::static_pointer_cast<typename ResourceTraits<Resource>::InternalType>(*this);
    assert(actualData.allocationInfo.pMappedData);

    return {reinterpret_cast<std::byte*>(actualData.allocationInfo.pMappedData), actualData.allocationInfo.size};
}

ResourceFactory::ResourceFactory(const vk::raii::Instance& instance, vk::PhysicalDevice physicalDevice, const vk::raii::Device& device)
    : m_resourceAllocator{instance, physicalDevice, device} {}

ResourceFactory::ResourceFactory(ResourceFactory&& rhv)
    : m_resourceAllocator(std::move(rhv.m_resourceAllocator))
    //	, m_usedBuffers(std::move(rhv.m_usedBuffers))
    , m_currentTick(std::exchange(rhv.m_currentTick, 0))
    , m_deleters(std::move(rhv.m_deleters)) {}

ResourceFactory& ResourceFactory::operator=(ResourceFactory&& rhv) {
    ResourceFactory copy{std::move(rhv)};
    std::swap(*this, copy);

    return *this;
}


ResourceFactory::~ResourceFactory() {
    *m_sharedSelf = nullptr;
    while (!m_deleters.empty()) {
        FreeUnusedResources();
    }
}

template <typename T, typename ... Args>
ResourceFactory::Handle<T> ResourceFactory::CreateResource(Args&& ... args) {
    using Traits  =  ResourceTraits<T>;
    using InternalType = typename Traits::InternalType;
    using SharedPtrType = typename Traits::SharedPtrType;

    auto resource = m_resourceAllocator.Create(std::forward<Args>(args)...);

    auto deleter = [factoryPtr = m_sharedSelf](const InternalType* resourceInfo) {
        if (*factoryPtr) {
            auto& factory = **factoryPtr;

            // note that resourceInfo is an owning ptr, it will be deleted with resource later
            factory.m_deleters.emplace_back(factory.m_currentTick, [resourceInfo](VulkanMemoryAllocator& allocator) {
                allocator.Destroy(*resourceInfo);
                delete resourceInfo;
            });
            return;
        }
        assert(false);
    };

     SharedPtrType ptr{new InternalType(resource), deleter};
    // m_usedBuffers.push_back(ptr);

    return {std::move(ptr)};
}

ResourceFactory::Handle<vk::Buffer> ResourceFactory::CreateBuffer(const vk::BufferCreateInfo& createInfo, CasualUsage usage) {
    return CreateResource<vk::Buffer>(createInfo, usage);
}

ResourceFactory::Handle<vk::Image> ResourceFactory::CreateImage(const vk::ImageCreateInfo& createInfo, CasualUsage usage) {
    return  CreateResource<vk::Image>(createInfo, usage);
}

void ResourceFactory::FreeUnusedResources() {
    constexpr uint64_t kDestroyDelay  = 5;
    auto               nearestAliveIt = std::ranges::find_if(
        m_deleters, [this](uint64_t invalidationTimestamp) { return m_currentTick - invalidationTimestamp < kDestroyDelay; },
        &DelayedDeleter::invalidationTimestamp);
    std::ranges::for_each(m_deleters.begin(), nearestAliveIt, [this](auto&& function) { function(m_resourceAllocator); }, &DelayedDeleter::deleter);
    m_deleters.erase(m_deleters.begin(), nearestAliveIt);

    m_currentTick++;
}
