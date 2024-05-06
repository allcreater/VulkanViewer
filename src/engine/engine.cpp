module;
#include <algorithm>
#include <memory>
#include <ranges>

#include <vulkan/vulkan_raii.hpp>

module engine;

import :vulkan;
import :vulkan.sdl;

class Engine : public IEngine {
public:
	Engine(void* window, std::span<const char*> requiredExtensions)
		: vulkan_context{ requiredExtensions }
		, graphics_context{ vulkan_context.makeGraphicsContext(std::make_unique<SdlWindowingSystem>(reinterpret_cast<SDL_Window*>(window))) }
	{


	}

private:
	VulkanContext vulkan_context;
	VulkanGraphicsContext graphics_context;
};


std::unique_ptr<IEngine> MakeEngine(void* window) {
	auto extensions = getRequiredExtensions();
	return std::make_unique<Engine>(window, extensions);
}
