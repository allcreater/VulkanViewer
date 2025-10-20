module;

module engine;
import std;
import :vulkan.context;
import :vulkan.renderer;
import :vulkan.sdl;

class Engine : public IEngine {
public:
    Engine(void* window, std::span<const char*> requiredExtensions)
        : vulkan_context{requiredExtensions}
        , renderer{vulkan_context.makeGraphicsContext(std::make_unique<SdlWindowingSystem>(reinterpret_cast<SDL_Window*>(window)))} {}

    void Render() override { renderer.Render(); }

    void LoadModel(const std::filesystem::path& path) override {
        renderer.LoadModel(path);
    }

private:
    VulkanContext  vulkan_context;
    VulkanRenderer renderer;
};


std::unique_ptr<IEngine> MakeEngine(void* window) {
    auto extensions = getRequiredExtensions();
    return std::make_unique<Engine>(window, extensions);
}
