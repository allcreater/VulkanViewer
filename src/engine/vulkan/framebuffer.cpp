module;
#include <algorithm>
#include <numeric>
#include <memory>
#include <ranges>

#include <functional>
#include <iostream>
#include <format>

#include <vulkan/vulkan_raii.hpp>

export module engine : vulkan.framebuffer;

export class Framebuffer {
public:
	Framebuffer(const vk::raii::Device& device, vk::RenderPass renderPass);

private:

};



Framebuffer::Framebuffer(const vk::raii::Device& device, vk::RenderPass renderPass) {

}