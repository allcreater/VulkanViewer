module;
#include <algorithm>
#include <numeric>
#include <memory>
#include <ranges>

#include <functional>
#include <iostream>
#include <format>

#include <vulkan/vulkan_raii.hpp>

export module engine : vulkan.context;

import :utils;


export class IWindowingSystem
{
public:
	IWindowingSystem() {};
	IWindowingSystem(const IWindowingSystem&) = delete;
	IWindowingSystem& operator=(const IWindowingSystem&) = delete;

	virtual~IWindowingSystem() = default;
	virtual vk::raii::SurfaceKHR createSurface(const vk::raii::Instance& instance) const = 0;
	virtual vk::Extent2D getExtent() const = 0;
};


class VulkanContext;

struct SwapchainData {
	vk::raii::SwapchainKHR swapchain;
	vk::SurfaceFormatKHR surfaceFormat;
	//std::vector<vk::raii::Image> images;
	std::vector<vk::raii::ImageView> imageViews;
};

export class VulkanGraphicsContext final {
public:
	explicit VulkanGraphicsContext(std::unique_ptr<IWindowingSystem> windowingSystem, const VulkanContext& context);

	const IWindowingSystem& getWindowingSystem() const { return *windowingSystem;  }
	const VulkanContext& getContext() const { return context; }
	const vk::raii::Device& getDevice() const { return device; }
	const vk::raii::CommandBuffer& getCommandBuffer() const { return commandBuffer; }

	vk::Extent2D getExtent() const;
	vk::SurfaceFormatKHR getSurfaceFormat() const { return swapchain.surfaceFormat; }
	const SwapchainData& getSwapchainData() const { return swapchain; }

	const vk::raii::Queue& getGraphicsQueue() const { return graphicsQueue; }
	const vk::raii::Queue& getPresentQueue() const { return presentQueue; }


private:
	std::unique_ptr<IWindowingSystem> windowingSystem;
	const VulkanContext& context;
	vk::raii::SurfaceKHR surface;
	std::array<uint32_t, 2> graphicsAndPresentQueueFamilyIndices;
	vk::raii::Device device;
	vk::raii::Queue graphicsQueue, presentQueue;
	SwapchainData swapchain;
	vk::raii::CommandPool commandPool;
	vk::raii::CommandBuffer commandBuffer;
};



export class VulkanContext final {
public:
	explicit VulkanContext(std::span<const char*> requiredExtensions);
	VulkanGraphicsContext makeGraphicsContext(std::unique_ptr<IWindowingSystem> windowingSystem);

	const vk::raii::Context& getContext() const { return context; }
	const vk::raii::Instance& getInstance() const { return instance; }
	const vk::raii::PhysicalDevice& getPhysicalDevice() const { return physicalDevice; }

private:
	vk::raii::Context context;
	vk::raii::Instance instance;
	vk::raii::PhysicalDevice physicalDevice;
};



// Implementation
namespace {

std::vector<const char*> GetDesiredLayers() {
	std::vector<const char*> desiredLayers{
#ifdef AT3_DEBUG
		"VK_LAYER_KHRONOS_validation",
		"VK_LAYER_LUNARG_standard_validation",
#endif
	};

	auto available_layers /*?*/ = vk::enumerateInstanceLayerProperties();
	std::erase_if(desiredLayers, [&available_layers](const char* desiredLayerName) {
		return std::ranges::none_of(available_layers, [desiredLayerName](const auto& layerName) {return layerName == desiredLayerName; }, [](const vk::LayerProperties& props) { return static_cast<std::string_view>(props.layerName); });
	});

	return desiredLayers;
}

vk::raii::Instance MakeInstance(const vk::raii::Context& context, std::span<const char*> requiredExtensions) {
	constexpr vk::ApplicationInfo appInfo{
		.pApplicationName = "TinyVulkan",
		.applicationVersion = VK_MAKE_VERSION(0, 0, 1),
		.pEngineName = "TinyVulkan",
		.engineVersion = VK_MAKE_VERSION(0, 0, 1),
		.apiVersion = VK_API_VERSION_1_3,
	};

	const auto desired_layers = GetDesiredLayers();
//#ifdef AT3_DEBUG
//	required_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
//#endif
	vk::InstanceCreateInfo instance_create_info{
		.pApplicationInfo = &appInfo,
		.enabledLayerCount = static_cast<uint32_t>(desired_layers.size()),
		.ppEnabledLayerNames = desired_layers.data(),
		.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
		.ppEnabledExtensionNames = requiredExtensions.data(),
	};

	
	return context.createInstance(instance_create_info);
}

vk::raii::PhysicalDevice GetAppropriatePhysicalDevice(const vk::raii::Instance& instance)
{
	auto devices = instance.enumeratePhysicalDevices();
	if (devices.empty())
		throw std::runtime_error("SelectPhysicalDevice: no physical devices O_o");

	std::erase_if(devices, [](const vk::PhysicalDevice& device) {
		const auto extensionProperties = device.enumerateDeviceExtensionProperties();
		return std::ranges::none_of(
			extensionProperties, [](const vk::ExtensionProperties& props) {
				return static_cast<std::string_view>(props.extensionName) == VK_KHR_SWAPCHAIN_EXTENSION_NAME;
			});

		//device.getSurfaceSupportKHR
		//const auto queueFamilyProperties = device.getQueueFamilyProperties();
		//	std::ranges::find_if(queueFamilyProperties, [](const vk::QueueFamilyProperties& props) {
		//		return static_cast<bool>(props.queueFlags & vk::QueueFlagBits::eGraphics);
		//	});
		});

	if (devices.empty())
		throw std::runtime_error("SelectPhysicalDevice: no physical devices with a swapchain");

	const auto it = std::ranges::find_if(devices, [](const vk::PhysicalDevice& device) {
		return device.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	});

	return it != devices.end() ? *it : devices.front();
}

std::array<uint32_t, 2> findGraphicsAndPresentQueueFamilyIndex(const vk::PhysicalDevice& physicalDevice, const vk::SurfaceKHR& surface)
{
	std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
	assert(queueFamilyProperties.size() < std::numeric_limits<uint32_t>::max());

	const auto supportGraphics = [&queueFamilyProperties](uint32_t index) { return static_cast<bool>(queueFamilyProperties[index].queueFlags & vk::QueueFlagBits::eGraphics);  };
	const auto supportPresent = [&physicalDevice, &surface](uint32_t index) { return physicalDevice.getSurfaceSupportKHR(index, surface) != 0;  };
	const auto familyIndices = std::ranges::iota_view<uint32_t, uint32_t>(0, queueFamilyProperties.size());

	for (auto universalQueueIndex : familyIndices | std::views::filter(supportGraphics) | std::views::filter(supportPresent)) {
		return { universalQueueIndex, universalQueueIndex };
	}

	auto graphicsQueues = familyIndices | std::views::filter(supportGraphics);
	auto presentQueues = familyIndices | std::views::filter(supportPresent);
	for (auto [graphicsQueueIndex, presentQueueIndex] : std::views::zip(graphicsQueues, presentQueues)) {
		return { graphicsQueueIndex, presentQueueIndex };
	}

	throw std::runtime_error("Could not find queues for both graphics or present -> terminating");
}

vk::raii::Device CreateDevice(const vk::raii::PhysicalDevice& physicalDevice, const vk::SurfaceKHR& surface) {
	auto [graphicsQueueIndex, presentQueueIndex] = findGraphicsAndPresentQueueFamilyIndex(physicalDevice, surface);

	constexpr std::array<float, 1> queue_priorities{ 1.0f };
	std::array <vk::DeviceQueueCreateInfo, 2> queueCreateInfo{ 
		vk::DeviceQueueCreateInfo{
			.queueFamilyIndex = graphicsQueueIndex,
			.queueCount = 1,
			.pQueuePriorities = queue_priorities.data(),
		},
		vk::DeviceQueueCreateInfo {
			.queueFamilyIndex = presentQueueIndex,
			.queueCount = 1,
			.pQueuePriorities = queue_priorities.data(),
		},
	};

	vk::PhysicalDeviceFeatures deviceFeatures{};
	constexpr std::array<const char*, 1> requiredDeviceExtensions{
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

	const auto desiredLayers = GetDesiredLayers();
	vk::DeviceCreateInfo deviceCreateInfo{
		.queueCreateInfoCount = (graphicsQueueIndex != presentQueueIndex) ? 2u : 1u, // static_cast<uint32_t>(queueCreateInfo.size()),
		.pQueueCreateInfos = queueCreateInfo.data(),
		.enabledLayerCount = static_cast<uint32_t>(desiredLayers.size()),
		.ppEnabledLayerNames = desiredLayers.data(),
		.enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtensions.size()),
		.ppEnabledExtensionNames = requiredDeviceExtensions.data(),
		.pEnabledFeatures = &deviceFeatures,
	};

	return physicalDevice.createDevice(deviceCreateInfo);
}

vk::SurfaceFormatKHR selectSurfaceFormat(const vk::PhysicalDevice& physicalDevice, const vk::SurfaceKHR& surface) {
	constexpr std::array<vk::SurfaceFormatKHR, 2> desiredFormats{
		vk::SurfaceFormatKHR{
			.format = vk::Format::eB8G8R8A8Srgb, 
			.colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear
		},
		vk::SurfaceFormatKHR{
			.format = vk::Format::eR8G8B8A8Srgb,
			.colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear
		},
	};
	
	const auto formats = physicalDevice.getSurfaceFormatsKHR(surface);
	for (auto format : desiredFormats | intersectsFilter(formats)) {
		return format;
	}

	return formats[0];
}

vk::Extent2D chooseExtent(const vk::SurfaceCapabilitiesKHR& capabilities, vk::Extent2D actualExtent) {
	if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
		return capabilities.currentExtent;
	}

	actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
	actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

	return actualExtent;
}

SwapchainData createSwapchain(const vk::PhysicalDevice& physicalDevice, const vk::raii::Device& device, const vk::SurfaceKHR& surface, const IWindowingSystem& windowingSystem, const std::array<uint32_t, 2>& graphicsAndPresentQueueFamilyIndices) {
	
	const auto capabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
	const auto surfaceFormat = selectSurfaceFormat(physicalDevice, surface);

	vk::SwapchainCreateInfoKHR createInfo{
		.flags = {},
		.surface = surface,
		.minImageCount = capabilities.maxImageCount > 0 ? std::min(capabilities.minImageCount + 1, capabilities.maxImageCount) : capabilities.minImageCount + 1,
		.imageFormat = surfaceFormat.format,
		.imageColorSpace = surfaceFormat.colorSpace,
		.imageExtent = chooseExtent(capabilities, windowingSystem.getExtent()),
		.imageArrayLayers = 1,
		.imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
		.imageSharingMode = vk::SharingMode::eExclusive,
		.queueFamilyIndexCount = 0,
		.pQueueFamilyIndices = nullptr,
		.preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity,
		.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
		.presentMode = vk::PresentModeKHR::eFifo,
		.clipped = {},
	};

	if (graphicsAndPresentQueueFamilyIndices[0] != graphicsAndPresentQueueFamilyIndices[1]) {
		createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
		createInfo.queueFamilyIndexCount = 2;
		createInfo.pQueueFamilyIndices = graphicsAndPresentQueueFamilyIndices.data();
	}

	auto swapchain = device.createSwapchainKHR(createInfo);

	const auto makeView = [&device, surfaceFormat](vk::Image image) {
		const vk::ImageViewCreateInfo createInfo{
			.flags = {},
			.image = image,
			.viewType = vk::ImageViewType::e2D,
			.format = surfaceFormat.format,
			.components = {},
			.subresourceRange = vk::ImageSubresourceRange{
				.aspectMask = vk::ImageAspectFlagBits::eColor,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};

		return device.createImageView(createInfo);
	};

	auto imageViews = swapchain.getImages() | std::views::transform(makeView) | std::ranges::to<std::vector>();

	return SwapchainData{
		.swapchain = std::move(swapchain),
		.surfaceFormat = surfaceFormat,
		.imageViews = std::move(imageViews),
	};
}

vk::raii::CommandPool createCommandPool (const vk::raii::Device& device, const std::array<uint32_t, 2>& graphicsAndPresentQueueFamilyIndices) {
	//command pool
	const vk::CommandPoolCreateInfo createInfo{
		.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
		.queueFamilyIndex = graphicsAndPresentQueueFamilyIndices[0],
	};

	return device.createCommandPool(createInfo);
}

vk::raii::CommandBuffer createCommandBuffer(const vk::raii::Device& device, vk::CommandPool commandPool) {
	const vk::CommandBufferAllocateInfo allocateInfo{
		.commandPool        = commandPool,
		.level              = vk::CommandBufferLevel::ePrimary,
		.commandBufferCount = 1,
	};

	return std::move(device.allocateCommandBuffers(allocateInfo)[0]);
}

} // end namespace

VulkanGraphicsContext::VulkanGraphicsContext(std::unique_ptr<IWindowingSystem> _windowingSystem, const VulkanContext& context)
	: windowingSystem{ std::move(_windowingSystem) }
	, context{context}
	, surface{ windowingSystem->createSurface(context.getInstance()) }
	, graphicsAndPresentQueueFamilyIndices{ findGraphicsAndPresentQueueFamilyIndex(context.getPhysicalDevice(), surface) }
	, device{ CreateDevice(context.getPhysicalDevice(), surface)}
	, graphicsQueue{device, graphicsAndPresentQueueFamilyIndices[0], 0}
	, presentQueue{device, graphicsAndPresentQueueFamilyIndices[1], 0}
	, swapchain{ createSwapchain(context.getPhysicalDevice(), device, surface, *windowingSystem, graphicsAndPresentQueueFamilyIndices)}
	, commandPool{ createCommandPool(device, graphicsAndPresentQueueFamilyIndices)}
	, commandBuffer { createCommandBuffer(device, commandPool)}
{

}

vk::Extent2D VulkanGraphicsContext::getExtent() const {
	const auto capabilities = context.getPhysicalDevice().getSurfaceCapabilitiesKHR(surface);
	return chooseExtent(capabilities, windowingSystem->getExtent());
}


VulkanContext::VulkanContext(std::span<const char*> requiredExtensions)
	: instance{ MakeInstance(context, requiredExtensions)}
	, physicalDevice { GetAppropriatePhysicalDevice(instance)}
{
	std::println(std::cout, "Selected device {}", static_cast<std::string_view>(physicalDevice.getProperties().deviceName));
}

VulkanGraphicsContext VulkanContext::makeGraphicsContext(std::unique_ptr<IWindowingSystem> windowingSystem) {
	return VulkanGraphicsContext{ std::move(windowingSystem), *this};
}