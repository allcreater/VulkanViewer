module;
#include <algorithm>
#include <memory>
#include <ranges>

#include <vulkan/vulkan_raii.hpp>
#include "SDL2/SDL_vulkan.h"

export module engine : vulkan.sdl;

import :vulkan.context;

export std::vector<const char*> getRequiredExtensions() {
	std::vector<const char*> result;

	unsigned int array_size = 0;
	SDL_Vulkan_GetInstanceExtensions(nullptr, &array_size, nullptr);

	result.resize(array_size);
	SDL_Vulkan_GetInstanceExtensions(nullptr, &array_size, result.data());

	std::span a = result;

	return result;
}

export class SdlWindowingSystem : public IWindowingSystem
{
public:
	SdlWindowingSystem(SDL_Window* window_) : window{ window_ } {};

	vk::raii::SurfaceKHR createSurface(const vk::raii::Instance& instance) const override {
		VkSurfaceKHR surface;

		if (SDL_Vulkan_CreateSurface(window, *instance, &surface) == SDL_FALSE)
			throw std::runtime_error("SDL: cannot create Vulkan surface for specified window");

	
		return { instance, surface };
	}


	vk::Extent2D getExtent() const override {
		int width, height;
		SDL_Vulkan_GetDrawableSize(window, &width, &height);

		return {
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height),
		};
	}

private:
	SDL_Window* window;
};

