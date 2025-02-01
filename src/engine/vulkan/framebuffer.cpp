export module engine:vulkan.framebuffer;
import vulkan_hpp;
import std;

export class Framebuffer {
public:
    Framebuffer(const vk::raii::Device& device, vk::RenderPass renderPass);

private:
};


Framebuffer::Framebuffer(const vk::raii::Device& device, vk::RenderPass renderPass) {}
