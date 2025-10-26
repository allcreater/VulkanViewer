module;
#include <glm/glm.hpp>

#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
// #include <fastgltf/glm_element_traits.hpp>

export module engine:vulkan.renderer;
import vulkan_hpp;
import std;
import fastgltf;

import utils.core;
import utils.image_loader;
import :vulkan.context;

struct Synchronization {
    explicit Synchronization(const vk::raii::Device& device)
        : imageAvailable{device, vk::SemaphoreCreateInfo{}}
        , renderFinished{device, vk::SemaphoreCreateInfo{}}
        , inFlightFence{device, {.flags = vk::FenceCreateFlagBits::eSignaled}} {}

    vk::raii::Semaphore imageAvailable;
    vk::raii::Semaphore renderFinished;
    vk::raii::Fence     inFlightFence;
};

struct ImageWithView {
    ResourceFactory::Handle<vk::Image> image;
    vk::raii::ImageView                view;
};

// TODO: it seems almost all fields should be shared (i.e. use handles)
struct Model {
    ResourceFactory::Handle<vk::Buffer> vertexBuffer, indexBuffer;
    std::vector<ImageWithView> images;
    std::vector<vk::raii::Sampler> samplers;
    vk::raii::DescriptorSet descriptorSet {nullptr};
};

export class VulkanRenderer final {
public:
    VulkanRenderer(VulkanGraphicsContext&& graphicsContext);
    ~VulkanRenderer();

    void Render();
    void LoadModel(const std::filesystem::path& path);
private:
    void CreatePipeline();

private:
    constexpr static int num_inflight_frames = 2;

    VulkanGraphicsContext               graphicsContext;
    vk::raii::ShaderModule              shaderModule{nullptr};
    vk::raii::PipelineLayout            pipelineLayout{nullptr};
    vk::raii::DescriptorSetLayout       descriptorSetLayout{nullptr};
    vk::raii::Pipeline                  trianglePipeline{nullptr};
    vk::raii::DescriptorPool            descriptorPool{nullptr};

    std::optional<Model>                model;

    // vk::raii::CommandPool commandPool;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    std::vector<Synchronization>         sync;
    std::vector<ImageWithView>           depthBuffers;
    uint32_t                             currentFrame = 0;
};

namespace
{

vk::raii::ShaderModule loadShaderModule(const vk::raii::Device& device, const std::filesystem::path& path) {
    const auto  data = readFile<std::uint32_t>(path);

    vk::ShaderModuleCreateInfo create_info{
        .flags    = {},
        .codeSize = std::span{data}.size_bytes(),
        .pCode    = data.data(),
    };

    return device.createShaderModule(create_info);
}

struct Vertex {
    fastgltf::math::fvec3 position;
    fastgltf::math::fvec3 normal;
    fastgltf::math::fvec2 uv;
};

struct Camera{
    glm::fmat4x4 viewProjection;
};

} // namespace


void VulkanRenderer::CreatePipeline() {
    const auto& device = graphicsContext.getDevice();
    shaderModule = loadShaderModule(device, "data/shaders/simple_model.spv");

    const std::array stages{
        vk::PipelineShaderStageCreateInfo{
            .flags               = {},
            .stage               = vk::ShaderStageFlagBits::eVertex,
            .module              = shaderModule,
            .pName               = "vertexMain",
            .pSpecializationInfo = nullptr,
        },
        vk::PipelineShaderStageCreateInfo{
            .flags               = {},
            .stage               = vk::ShaderStageFlagBits::eFragment,
            .module              = shaderModule,
            .pName               = "fragmentMain",
            .pSpecializationInfo = nullptr,
        },
    };

    const std::array vertexBindingDescriptions{
        vk::VertexInputBindingDescription{
            .binding   = 0,
            .stride    = sizeof(Vertex),
            .inputRate = vk::VertexInputRate::eVertex,
        },
    };
    constexpr std::array vertexAttributeDescriptions{
        vk::VertexInputAttributeDescription{
            .location = 0,
            .binding  = 0,
            .format   = vk::Format::eR32G32B32Sfloat,
            .offset   = offsetof(Vertex, position),
        },
        vk::VertexInputAttributeDescription{
            .location = 1,
            .binding  = 0,
            .format   = vk::Format::eR32G32B32Sfloat,
            .offset   = offsetof(Vertex, normal),
        },
        vk::VertexInputAttributeDescription{
            .location = 2,
            .binding  = 0,
            .format   = vk::Format::eR32G32Sfloat,
            .offset   = offsetof(Vertex, uv),
        },
    };

    const vk::PipelineVertexInputStateCreateInfo vertexInputState{
        .flags                           = {},
        .vertexBindingDescriptionCount   = vertexBindingDescriptions.size(),
        .pVertexBindingDescriptions      = vertexBindingDescriptions.data(),
        .vertexAttributeDescriptionCount = vertexAttributeDescriptions.size(),
        .pVertexAttributeDescriptions    = vertexAttributeDescriptions.data(),
    };

    const vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState{
        .flags                  = {},
        .topology               = vk::PrimitiveTopology::eTriangleList,
        .primitiveRestartEnable = vk::False,
    };

    const vk::PipelineTessellationStateCreateInfo tesselationState{
        .flags              = {},
        .patchControlPoints = 0,
    };

    const auto  swapChainExtent = graphicsContext.getExtent();
    const vk::Viewport viewport{
        .x        = 0.0f,
        .y        = 0.0f,
        .width    = static_cast<float>(swapChainExtent.width),
        .height   = static_cast<float>(swapChainExtent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const vk::Rect2D scissor{
        .offset = {0, 0},
        .extent = swapChainExtent,
    };

    const vk::PipelineViewportStateCreateInfo viewportState{
        .flags         = {},
        .viewportCount = 1,
        .pViewports    = &viewport,
        .scissorCount  = 1,
        .pScissors     = &scissor,
    };

    vk::PipelineRasterizationStateCreateInfo rasterizationState{
        .flags                   = {},
        .depthClampEnable        = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode             = vk::PolygonMode::eFill,
        .cullMode                = vk::CullModeFlagBits::eBack,
        .frontFace               = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable         = vk::False,
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp          = 0.0f,
        .depthBiasSlopeFactor    = 0.0f,
        .lineWidth               = 1.0f,
    };

    const vk::PipelineMultisampleStateCreateInfo multisampleState{
        .flags                 = {},
        .rasterizationSamples  = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable   = vk::False,
        .minSampleShading      = 1.0f,
        .pSampleMask           = nullptr,
        .alphaToCoverageEnable = vk::False,
        .alphaToOneEnable      = vk::False,
    };

    constexpr std::array<vk::DynamicState, 2> dynamicStates{
        vk::DynamicState::eViewport, vk::DynamicState::eScissor,
        // vk::DynamicState::eViewportWithCount,
        // vk::DynamicState::eScissorWithCount,
    };
    const vk::PipelineDynamicStateCreateInfo dynamicState{
        .flags             = {},
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates    = dynamicStates.data(),
    };

    const std::array colorBlendAttachments{
        vk::PipelineColorBlendAttachmentState{
            .blendEnable         = vk::False,
            .srcColorBlendFactor = vk::BlendFactor::eOne,
            .dstColorBlendFactor = vk::BlendFactor::eZero,
            .colorBlendOp        = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOne,
            .dstAlphaBlendFactor = vk::BlendFactor::eZero,
            .alphaBlendOp        = vk::BlendOp::eAdd,
            .colorWriteMask =
                vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        },
    };

    constexpr vk::PipelineDepthStencilStateCreateInfo depthStencilState{
        .depthTestEnable       = true,
        .depthWriteEnable      = true,
        .depthCompareOp        = vk::CompareOp::eLess,
        .depthBoundsTestEnable = false,
        .stencilTestEnable     = false,
        .front                 = {},
        .back                  = {},
        .minDepthBounds        = {},
        .maxDepthBounds        = {},
    };

    const vk::PipelineColorBlendStateCreateInfo colorBlendState{
        .flags           = {},
        .logicOpEnable   = {},
        .logicOp         = vk::LogicOp::eClear,
        .attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size()),
        .pAttachments    = colorBlendAttachments.data(),
        .blendConstants  = std::array{0.0f, 0.0f, 0.0f, 0.0f},
    };

    std::array bindings{
        vk::DescriptorSetLayoutBinding{
            .binding            = 0,
            .descriptorType     = vk::DescriptorType::eSampledImage,
            .descriptorCount    = 1,
            .stageFlags         = vk::ShaderStageFlagBits::eFragment,
            .pImmutableSamplers = nullptr,
        },
        vk::DescriptorSetLayoutBinding{
            .binding            = 1,
            .descriptorType     = vk::DescriptorType::eSampler,
            .descriptorCount    = 1,
            .stageFlags         = vk::ShaderStageFlagBits::eFragment,
            .pImmutableSamplers = nullptr,
        },
    };

    vk::DescriptorSetLayoutCreateInfo descriptorLayoutCreateInfo{
        .flags        = {},
        .bindingCount = bindings.size(),
        .pBindings    = bindings.data(),
    };

    descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayoutCreateInfo);
    const std::array setLayouts{
        *descriptorSetLayout,
    };

    constexpr std::array pushConstantsRanges{
        vk::PushConstantRange{
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
            .offset     = 0,
            .size       = sizeof(glm::mat4),
        },
    };

    const vk::PipelineLayoutCreateInfo layoutCreateInfo{
        .flags                  = {},
        .setLayoutCount         = setLayouts.size(),
        .pSetLayouts            = setLayouts.data(),
        .pushConstantRangeCount = pushConstantsRanges.size(),
        .pPushConstantRanges    = pushConstantsRanges.data(),
    };
    pipelineLayout = device.createPipelineLayout(layoutCreateInfo);


    const std::array colorAttachmentFormats{graphicsContext.getSurfaceFormat().format};

    vk::StructureChain createInfo{vk::GraphicsPipelineCreateInfo{
                                      .flags               = {},
                                      .stageCount          = static_cast<uint32_t>(stages.size()),
                                      .pStages             = stages.data(),
                                      .pVertexInputState   = &vertexInputState,
                                      .pInputAssemblyState = &inputAssemblyState,
                                      .pTessellationState  = &tesselationState,
                                      .pViewportState      = &viewportState,
                                      .pRasterizationState = &rasterizationState,
                                      .pMultisampleState   = &multisampleState,
                                      .pDepthStencilState  = &depthStencilState,
                                      .pColorBlendState    = &colorBlendState,
                                      .pDynamicState       = &dynamicState,
                                      .layout              = pipelineLayout,
                                      .subpass             = 0,
                                      .basePipelineHandle  = {nullptr},
                                      .basePipelineIndex   = -1,
                                  },
                                  vk::PipelineRenderingCreateInfo{
                                      .colorAttachmentCount    = static_cast<uint32_t>(colorAttachmentFormats.size()),
                                      .pColorAttachmentFormats = colorAttachmentFormats.data(),
                                      .depthAttachmentFormat = vk::Format::eD32Sfloat,
                                  }};

    trianglePipeline = device.createGraphicsPipeline({nullptr}, createInfo.get());
}

VulkanRenderer::VulkanRenderer(VulkanGraphicsContext&& _graphicsContext) : graphicsContext{std::move(_graphicsContext)} {
    CreatePipeline();

    const auto createDepthBuffer = [&] -> ImageWithView{
        auto image = graphicsContext.getResourceFactory().CreateImage(vk::ImageCreateInfo{
            .flags                 = {},
            .imageType             = vk::ImageType::e2D,
            .format                = vk::Format::eD32Sfloat,
            .extent                = vk::Extent3D{graphicsContext.getExtent().width, graphicsContext.getExtent().height, 1},
            .mipLevels             = 1,
            .arrayLayers           = 1,
            .samples               = vk::SampleCountFlagBits::e1,
            .tiling                = vk::ImageTiling::eOptimal,
            .usage                 = vk::ImageUsageFlagBits::eDepthStencilAttachment,
            .sharingMode           = vk::SharingMode::eExclusive,
            .queueFamilyIndexCount = {},
            .pQueueFamilyIndices   = {},
            .initialLayout         = vk::ImageLayout::eUndefined,
        }, CasualUsage::Auto);

        auto view = vk::raii::ImageView{graphicsContext.getDevice(),
                                             vk::ImageViewCreateInfo{
                                                 .flags      = {},
                                                 .image      = *image,
                                                 .viewType   = vk::ImageViewType::e2D,
                                                 .format     = vk::Format::eD32Sfloat,
                                                 .components = {},
                                                 .subresourceRange =
                                                     vk::ImageSubresourceRange{
                                                         .aspectMask     = vk::ImageAspectFlagBits::eDepth,
                                                         .baseMipLevel   = 0,
                                                         .levelCount     = 1,
                                                         .baseArrayLayer = 0,
                                                         .layerCount     = 1,
                                                     },
                                             }};

        return ImageWithView{
            .image = std::move(image),
            .view  = std::move(view),
        };
    };


    descriptorPool = [&] {
        std::array descritporPoolSizes {
            vk::DescriptorPoolSize{.type = vk::DescriptorType::eSampledImage, .descriptorCount = 128},
            vk::DescriptorPoolSize{.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 128},
            vk::DescriptorPoolSize{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = 128},
        };

        return graphicsContext.getDevice().createDescriptorPool(vk::DescriptorPoolCreateInfo{
            .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets       = 128,
            .poolSizeCount = descritporPoolSizes.size(),
            .pPoolSizes    = descritporPoolSizes.data(),
        });
    }();

    sync = std::views::iota(0, num_inflight_frames) | std::views::transform([&](auto _) { return Synchronization{graphicsContext.getDevice()}; }) |
        std::ranges::to<std::vector>();

    depthBuffers = std::views::iota(0, num_inflight_frames) | std::views::transform([&](auto _) { return createDepthBuffer(); }) |
        std::ranges::to<std::vector>();

    commandBuffers = graphicsContext.createCommandBuffers(num_inflight_frames);
}

VulkanRenderer::~VulkanRenderer() {
    const vk::raii::Device& device = graphicsContext.getDevice();
    device.waitIdle();
}

void VulkanRenderer::Render() {
    const vk::raii::Device&        device          = graphicsContext.getDevice();
    const vk::raii::CommandBuffer& commandBuffer   = commandBuffers[currentFrame];
    const auto                     swapChainExtent = graphicsContext.getExtent();
    const auto&                    frameSync       = sync[currentFrame];

    device.waitForFences(std::array{*frameSync.inFlightFence}, vk::True, std::numeric_limits<uint64_t>::max());
    device.resetFences(std::array{*frameSync.inFlightFence});

    const vk::raii::SwapchainKHR& swapchain = graphicsContext.getSwapchainData().swapchain;
    const auto [success, frameImageIndex]   = swapchain.acquireNextImage(std::numeric_limits<uint64_t>::max(), frameSync.imageAvailable, nullptr);


    commandBuffer.reset();
    {
        commandBuffer.begin({});

        {
            const vk::ImageMemoryBarrier2 barrier{
                .srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
                .dstStageMask  = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .oldLayout     = vk::ImageLayout::eUndefined,
                .newLayout     = vk::ImageLayout::eColorAttachmentOptimal,
                .image         = graphicsContext.getSwapchainData().images[frameImageIndex],
                .subresourceRange =
                    {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1},
            };
            commandBuffer.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(barrier));
        }

        const std::array colorAttachments{
            vk::RenderingAttachmentInfo{
                .imageView   = graphicsContext.getSwapchainData().imageViews[frameImageIndex],
                .imageLayout = vk::ImageLayout::eAttachmentOptimal,
                .loadOp      = vk::AttachmentLoadOp::eClear,
                .storeOp     = vk::AttachmentStoreOp::eStore,
                .clearValue  = {vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f}},
            },
        };

        const vk::RenderingAttachmentInfo depthAttachment{
            .imageView   = depthBuffers[currentFrame].view,
            .imageLayout = vk::ImageLayout::eAttachmentOptimal,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eDontCare,
            .clearValue  = {vk::ClearDepthStencilValue{1.0f }},
        };

        commandBuffer.beginRendering({
            .renderArea           = {.offset = {}, .extent = swapChainExtent},
            .layerCount           = 1,
            .colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size()),
            .pColorAttachments    = colorAttachments.data(),
            .pDepthAttachment     = &depthAttachment,
        });

        const std::array viewports{
            vk::Viewport{
                .x        = 0.0f,
                .y        = 0.0f,
                .width    = (float)swapChainExtent.width,
                .height   = (float)swapChainExtent.height,
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            },
        };
        commandBuffer.setViewport(0, viewports);


        const std::array<vk::Rect2D, 1> scissorRects{
            vk::Rect2D{
                .offset = {0, 0},
                .extent = swapChainExtent,
            },
        };
        commandBuffer.setScissor(0, scissorRects);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, trianglePipeline);

        if (model) {
            auto& [vertexBuffer, indexBuffer, images_, samplers_, descriptorSet] = *model;
            commandBuffer.bindVertexBuffers(0, *vertexBuffer, {0});
            commandBuffer.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, {descriptorSet}, {});

            auto proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 100.0f);
            auto view = glm::lookAt(glm::vec3(20.0f, 20.0f, 20.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            Camera camera{
                .viewProjection = proj * view,
            };
            commandBuffer.pushConstants<Camera>(pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, vk::ArrayProxy{camera});

            int count = indexBuffer.mappedMemory().size() / sizeof(uint32_t);
            commandBuffer.drawIndexed(count, 1, 0, 0, 0);
            //commandBuffer.drawIndirect()
        }

        commandBuffer.endRendering();

        {
            const vk::ImageMemoryBarrier2 barrier{
                .srcStageMask  = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .dstStageMask  = vk::PipelineStageFlagBits2::eNone,
                .dstAccessMask = vk::AccessFlagBits2::eNone,
                .oldLayout     = vk::ImageLayout::eColorAttachmentOptimal,
                .newLayout     = vk::ImageLayout::ePresentSrcKHR,
                .image         = graphicsContext.getSwapchainData().images[frameImageIndex],
                .subresourceRange =
                    {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1},
            };
            commandBuffer.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(barrier));
        }

        commandBuffer.end();
    }

    const std::array<vk::Semaphore, 1>          semaphoresForWait   = {frameSync.imageAvailable};
    const std::array<vk::Semaphore, 1>          semaphoresForSignal = {frameSync.renderFinished};
    const std::array<vk::PipelineStageFlags, 1> waitStages          = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
    const vk::SubmitInfo                        submitInfo{
                               .waitSemaphoreCount   = static_cast<uint32_t>(semaphoresForWait.size()),
                               .pWaitSemaphores      = semaphoresForWait.data(),
                               .pWaitDstStageMask    = waitStages.data(),
                               .commandBufferCount   = 1,
                               .pCommandBuffers      = &*commandBuffer,
                               .signalSemaphoreCount = static_cast<uint32_t>(semaphoresForSignal.size()),
                               .pSignalSemaphores    = semaphoresForSignal.data(),
    };

    graphicsContext.getGraphicsQueue().submit(submitInfo, frameSync.inFlightFence);

    const vk::PresentInfoKHR presentInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores    = &*frameSync.renderFinished,
        .swapchainCount     = 1,
        .pSwapchains        = &*swapchain,
        .pImageIndices      = &frameImageIndex,
        .pResults           = nullptr,
    };

    graphicsContext.getPresentQueue().presentKHR(presentInfo);

    currentFrame = (currentFrame + 1) % num_inflight_frames;
}

template <typename T>
auto iterateAccessor (const fastgltf::Asset& asset, const fastgltf::Primitive& primitive, std::string_view name) {
    auto it = primitive.findAttribute(name);
    if (it == primitive.attributes.end())
        throw std::runtime_error{"Model does not contain attribute"};

    return fastgltf::iterateAccessor<T>(asset, asset.accessors[it->accessorIndex]);
};

template <typename T, std::ranges::range R>
requires std::is_standard_layout_v<T>
auto writeRangeToMemory(R&& dataRange, std::span<std::byte> destMemory) {
    std::ranges::for_each(dataRange, [&destMemory](const T& vertex) {
        std::memcpy(destMemory.data(), &vertex, sizeof(T));
        destMemory = destMemory.subspan(sizeof(T));
    });
}

// template <typename... Args>
// auto iterateAccessors (const fastgltf::Asset& asset, const fastgltf::Primitive& primitive, std::array<std::string_view, sizeof...(Args)> name) {
//
// }

auto loadAssetImage(const fastgltf::Image& image) {
    auto* data = std::get_if<fastgltf::sources::Array>(&image.data);
    assert(data);

    return loadImageAsRgba<const std::uint8_t>(std::span{reinterpret_cast<const std::uint8_t*>(data->bytes.data()), data->bytes.size()});
}

ImageWithView createTextureFromImage(image2d_rgba<const std::uint8_t> imageData, VulkanGraphicsContext& graphicsContext) {
        auto image = graphicsContext.getResourceFactory().CreateImage(vk::ImageCreateInfo{
            .flags                 = {},
            .imageType             = vk::ImageType::e2D,
            .format                = vk::Format::eR8G8B8A8Srgb,
            .extent                = vk::Extent3D{imageData.extent(1), imageData.extent(0), 1},
            .mipLevels             = 1,
            .arrayLayers           = 1,
            .samples               = vk::SampleCountFlagBits::e1,
            .tiling                = vk::ImageTiling::eOptimal,
            .usage                 = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            .sharingMode           = vk::SharingMode::eExclusive,
            .queueFamilyIndexCount = {},
            .pQueueFamilyIndices   = {},
            .initialLayout         = vk::ImageLayout::eUndefined,
        }, CasualUsage::Auto);

        auto view = vk::raii::ImageView{graphicsContext.getDevice(),
                                             vk::ImageViewCreateInfo{
                                                 .flags      = {},
                                                 .image      = *image,
                                                 .viewType   = vk::ImageViewType::e2D,
                                                 .format     = vk::Format::eR8G8B8A8Srgb,
                                                 .components = {},
                                                 .subresourceRange =
                                                     vk::ImageSubresourceRange{
                                                         .aspectMask     = vk::ImageAspectFlagBits::eColor,
                                                         .baseMipLevel   = 0,
                                                         .levelCount     = 1,
                                                         .baseArrayLayer = 0,
                                                         .layerCount     = 1,
                                                     },
                                             }};

        {
            auto transferBuffer = graphicsContext.getResourceFactory().CreateBuffer({
               .flags       = {},
               .size        = imageData.size(),
               .usage       = vk::BufferUsageFlagBits::eTransferSrc,
               .sharingMode = vk::SharingMode::eExclusive
            }, CasualUsage::AutoMapped);
            std::memcpy(transferBuffer.mappedMemory().data(), imageData.data_handle().get(), imageData.size());


            vk::raii::CommandBuffer commandBuffer = std::move(graphicsContext.createCommandBuffers(1)[0]);
            vk::BufferImageCopy2 copyRegion {
                .bufferOffset      = 0,
                .bufferRowLength   = 0,//imageData.size() / imageData.extent(0)/4,
                .bufferImageHeight = 0,//imageData.extent(0),
                .imageSubresource  = {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
                .imageOffset       = {.x =  0, .y = 0, .z = 0},
                .imageExtent       = {.width = imageData.extent(1), .height = imageData.extent(0), .depth = 1},
            };

            commandBuffer.begin({});
            {
                const vk::ImageMemoryBarrier2 barrier{
                    .srcStageMask  = vk::PipelineStageFlagBits2::eTopOfPipe,
                    .srcAccessMask = vk::AccessFlagBits2::eNone,
                    .dstStageMask  = vk::PipelineStageFlagBits2::eTransfer,
                    .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                    .oldLayout     = vk::ImageLayout::eUndefined,
                    .newLayout     = vk::ImageLayout::eTransferDstOptimal,
                    .image         = *image,
                    .subresourceRange =
                        {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1},
                };
                commandBuffer.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(barrier));
            }

            commandBuffer.copyBufferToImage2({
                .srcBuffer = *transferBuffer,
                .dstImage = *image,
                .dstImageLayout = vk::ImageLayout::eTransferDstOptimal,
                .regionCount = 1,
                .pRegions = &copyRegion,
            });

            {
                const vk::ImageMemoryBarrier2 barrier{
                    .srcStageMask  = vk::PipelineStageFlagBits2::eTransfer,
                    .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                    .dstStageMask  = vk::PipelineStageFlagBits2::eFragmentShader,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                    .oldLayout     = vk::ImageLayout::eTransferDstOptimal,
                    .newLayout     = vk::ImageLayout::eShaderReadOnlyOptimal,
                    .image         = *image,
                    .subresourceRange =
                        {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1},
                };
                commandBuffer.pipelineBarrier2(vk::DependencyInfo().setImageMemoryBarriers(barrier));
            }

            commandBuffer.end();
            graphicsContext.getGraphicsQueue().submit(vk::SubmitInfo{
                .commandBufferCount = 1,
                .pCommandBuffers    = &*commandBuffer,
            });
            graphicsContext.getGraphicsQueue().waitIdle();
        }

        return ImageWithView{
            .image = std::move(image),
            .view  = std::move(view),
        };
}

void VulkanRenderer::LoadModel(const std::filesystem::path& path) {
    fastgltf::GltfFileStream stream{path};

    fastgltf::Parser parser{{}};
    auto result = parser.loadGltf(stream, path.parent_path(), fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages);
    if (!result) {
        throw std::runtime_error{std::format("Failed to load model: {}", fastgltf::getErrorName(result.error()))};
    }

    // make vertex buffer

    //result->meshes[0].primitives[0]

    const auto& primitive = result->meshes[0].primitives[0];

    Model resultModel;
    // vertex
    {
        auto positions = iterateAccessor<fastgltf::math::fvec3>(result.get(), primitive, "POSITION");
        auto normals = iterateAccessor<fastgltf::math::fvec3>(result.get(), primitive, "NORMAL");
        auto uvs = iterateAccessor<fastgltf::math::fvec2>(result.get(), primitive, "TEXCOORD_0");
        auto interleavedVertexAttribs = std::views::zip(positions, normals, uvs) | std::views::transform([](auto&& tuple) {
            return std::make_from_tuple<Vertex>(tuple);
        });

        auto buffer = graphicsContext.getResourceFactory().CreateBuffer(
            {
                .flags       = {},
                .size        = std::ranges::distance(interleavedVertexAttribs) * sizeof(Vertex),
                .usage       = vk::BufferUsageFlagBits::eVertexBuffer,
                .sharingMode = vk::SharingMode::eExclusive,
            },
            CasualUsage::AutoMapped);

        writeRangeToMemory<Vertex>(interleavedVertexAttribs, buffer.mappedMemory());
        std::span target {reinterpret_cast<Vertex*>(buffer.mappedMemory().data()), buffer.mappedMemory().size_bytes() / sizeof(Vertex)};

        resultModel.vertexBuffer = std::move(buffer);
    }

    // indices
    {
        const auto& accessor = result->accessors[primitive.indicesAccessor.value()];
        auto buffer = graphicsContext.getResourceFactory().CreateBuffer(
           {
               .flags       = {},
               .size        = accessor.count * sizeof(std::uint32_t),
               .usage       = vk::BufferUsageFlagBits::eIndexBuffer,
               .sharingMode = vk::SharingMode::eExclusive,
           },
           CasualUsage::AutoMapped);

        fastgltf::copyFromAccessor<uint32_t>(result.get(), accessor, buffer.mappedMemory().data());

        std::span target {reinterpret_cast<uint32_t*>(buffer.mappedMemory().data()), buffer.mappedMemory().size_bytes() / sizeof(uint32_t)};

        resultModel.indexBuffer = std::move(buffer);
    }

    resultModel.images.push_back(createTextureFromImage(loadAssetImage(result.get().images[0]), graphicsContext));
    resultModel.samplers.push_back(
        graphicsContext.getDevice().createSampler(vk::SamplerCreateInfo{})
    );

    resultModel.descriptorSet = std::move(graphicsContext.getDevice().allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
        .descriptorPool = descriptorPool,
        .descriptorSetCount =  1,
        .pSetLayouts = &*descriptorSetLayout,
    })[0]);

    {
        vk::DescriptorImageInfo descriptorImageInfo{
            .sampler     = nullptr,
            .imageView   = resultModel.images[0].view,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };
        vk::DescriptorImageInfo descriptorSamplerInfo{
            .sampler     = resultModel.samplers[0],
            .imageView   = nullptr,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        graphicsContext.getDevice().updateDescriptorSets(
            {
                vk::WriteDescriptorSet{
                    .dstSet = resultModel.descriptorSet,
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eSampledImage,
                    .pImageInfo = &descriptorImageInfo,
                    .pBufferInfo = nullptr,
                    .pTexelBufferView = nullptr,

                },
                vk::WriteDescriptorSet{
                   .dstSet = resultModel.descriptorSet,
                   .dstBinding = 1,
                   .dstArrayElement = 0,
                   .descriptorCount = 1,
                   .descriptorType = vk::DescriptorType::eSampler,
                   .pImageInfo = &descriptorSamplerInfo,
                   .pBufferInfo = nullptr,
                   .pTexelBufferView = nullptr,

               },
            },
            {
            }
        );
    }


    model = std::move(resultModel);
}
