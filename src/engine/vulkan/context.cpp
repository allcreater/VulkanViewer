module;
#include <algorithm>
#include <numeric>
#include <memory>
#include <ranges>

#include <functional>
#include <iostream>
#include <format>

#include <vulkan/vulkan_raii.hpp>

export module engine : vulkan;

import :utils;


export class IWindowingSystem
{
public:
	IWindowingSystem() {};
	IWindowingSystem(const IWindowingSystem&) = delete;
	IWindowingSystem& operator=(const IWindowingSystem&) = delete;

	virtual~IWindowingSystem() = default;
	virtual vk::raii::SurfaceKHR createSurface(const vk::raii::Instance& instance) const = 0;
	virtual vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) const = 0;
};


class VulkanContext;

export class VulkanGraphicsContext {
public:
	explicit VulkanGraphicsContext(std::unique_ptr<IWindowingSystem> windowingSystem, const VulkanContext& context);

private:
	vk::raii::SurfaceKHR surface;
	vk::raii::Device device;
	vk::raii::SwapchainKHR swapchain; 
};


export class VulkanContext {
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

vk::raii::SwapchainKHR createSwapchain(const vk::PhysicalDevice& physicalDevice, const vk::raii::Device& device, const vk::SurfaceKHR& surface, const IWindowingSystem& windowingSystem) {
	
	const auto capabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
	const auto surfaceFormat = selectSurfaceFormat(physicalDevice, surface);

	vk::SwapchainCreateInfoKHR createInfo{
		.flags = {},
		.surface = surface,
		.minImageCount = capabilities.maxImageCount > 0 ? std::min(capabilities.minImageCount + 1, capabilities.maxImageCount) : capabilities.minImageCount + 1,
		.imageFormat = surfaceFormat.format,
		.imageColorSpace = surfaceFormat.colorSpace,
		.imageExtent = windowingSystem.chooseSwapExtent(capabilities),
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

	const auto queueFamilyIndices = findGraphicsAndPresentQueueFamilyIndex(physicalDevice, surface);
	if (queueFamilyIndices[0] != queueFamilyIndices[1]) {
		createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
		createInfo.queueFamilyIndexCount = 2;
		createInfo.pQueueFamilyIndices = queueFamilyIndices.data();
	}

	return device.createSwapchainKHR(createInfo);
}

} // end namespace

VulkanGraphicsContext::VulkanGraphicsContext(std::unique_ptr<IWindowingSystem> windowingSystem, const VulkanContext& context)
	: surface{ std::move(windowingSystem->createSurface(context.getInstance())) }
	, device{ CreateDevice(context.getPhysicalDevice(), surface)}
	, swapchain{ createSwapchain(context.getPhysicalDevice(), device, surface, *windowingSystem)}
{
	auto images = swapchain.getImages();
}

VulkanContext::VulkanContext(std::span<const char*> requiredExtensions)
	: instance{ MakeInstance(context, requiredExtensions)}
	, physicalDevice { GetAppropriatePhysicalDevice(instance)}
{
	std::print(std::cout, "Selected device {}", static_cast<std::string_view>(physicalDevice.getProperties().deviceName));
}

VulkanGraphicsContext VulkanContext::makeGraphicsContext(std::unique_ptr<IWindowingSystem> windowingSystem) {
	return VulkanGraphicsContext{ std::move(windowingSystem), *this};
}