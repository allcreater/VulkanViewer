module;
#include <glm/glm.hpp>

export module engine:vulkan.renderer;
import vulkan_hpp;
import std;

import :utils;
import :vulkan.context;

struct Synchronization {
    explicit Synchronization(const vk::raii::Device& device)
        : imageAvailable{device, vk::SemaphoreCreateInfo{}}
        , rederFinished{device, vk::SemaphoreCreateInfo{}}
        , inFlightFence{device, {.flags = vk::FenceCreateFlagBits::eSignaled}} {}

    vk::raii::Semaphore imageAvailable;
    vk::raii::Semaphore rederFinished;
    vk::raii::Fence     inFlightFence;
};

export class VulkanRenderer final {
public:
    VulkanRenderer(VulkanGraphicsContext&& graphicsContext);
    ~VulkanRenderer();

    void Render();

private:
    constexpr static int num_inflight_frames = 2;

    VulkanGraphicsContext               graphicsContext;
    vk::raii::ShaderModule              vs{nullptr}, fs{nullptr};
    vk::raii::PipelineLayout            pipelineLayout{nullptr};
    vk::raii::Pipeline                  trianglePipeline{nullptr};
    ResourceFactory::Handle<vk::Buffer> vertexBuffer;
    // vk::raii::CommandPool commandPool;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    std::vector<Synchronization>         sync;
    uint32_t                             currentFrame = 0;
};

namespace
{

vk::raii::ShaderModule loadShaderModule(const vk::raii::Device& device, const std::filesystem::path& path) {
    const auto                      data = readFile(path);
    const std::span<const uint32_t> dataView{reinterpret_cast<const uint32_t*>(data.data()), data.size() / sizeof(uint32_t)};

    vk::ShaderModuleCreateInfo create_info{
        .flags    = {},
        .codeSize = dataView.size_bytes(),
        .pCode    = dataView.data(),
    };

    return device.createShaderModule(create_info);
}

} // namespace


VulkanRenderer::VulkanRenderer(VulkanGraphicsContext&& _graphicsContext) : graphicsContext{std::move(_graphicsContext)}, vertexBuffer{nullptr} {
    sync = std::views::iota(0, num_inflight_frames) | std::views::transform([&](auto _) { return Synchronization{graphicsContext.getDevice()}; }) |
        std::ranges::to<std::vector>();

    const auto& device = graphicsContext.getDevice();

    vs = loadShaderModule(device, "data/shaders/hello_world.vs.spv");
    fs = loadShaderModule(device, "data/shaders/hello_world.fs.spv");

    const std::array<vk::PipelineShaderStageCreateInfo, 2> stages{
        vk::PipelineShaderStageCreateInfo{
            .flags               = {},
            .stage               = vk::ShaderStageFlagBits::eVertex,
            .module              = vs,
            .pName               = "main",
            .pSpecializationInfo = nullptr,
        },
        vk::PipelineShaderStageCreateInfo{
            .flags               = {},
            .stage               = vk::ShaderStageFlagBits::eFragment,
            .module              = fs,
            .pName               = "main",
            .pSpecializationInfo = nullptr,
        },
    };

    const std::array                             vertexBindingDescriptions{vk::VertexInputBindingDescription{
                                    .binding   = 0,
                                    .stride    = sizeof(float) * 5,
                                    .inputRate = vk::VertexInputRate::eVertex,
    }};
    const std::array                             vertexAttributeDescriptions{vk::VertexInputAttributeDescription{
                                                                                 .location = 0,
                                                                                 .binding  = 0,
                                                                                 .format   = vk::Format::eR32G32Sfloat,
                                                                                 .offset   = 0,
                                                 },
                                                 vk::VertexInputAttributeDescription{
                                                                                 .location = 1,
                                                                                 .binding  = 0,
                                                                                 .format   = vk::Format::eR32G32B32Sfloat,
                                                                                 .offset   = sizeof(float) * 2,
                                                 }};
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

    const auto         swapChainExtent = graphicsContext.getExtent();
    const vk::Viewport viewport{
        .x        = 0.0f,
        .y        = 0.0f,
        .width    = (float)swapChainExtent.width,
        .height   = (float)swapChainExtent.height,
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
        .frontFace               = vk::FrontFace::eClockwise,
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

    // const vk::PipelineDepthStencilStateCreateInfo depthStencilState{
    //	.flags                 = {},
    //	.depthTestEnable       = vk::False,
    //	.depthWriteEnable      = vk::False,
    //	.depthCompareOp        = vk::CompareOp::eAlways,
    //	.depthBoundsTestEnable = {},
    //	.stencilTestEnable     = {},
    //	.front                 = {},
    //	.back                  = {},
    //	.minDepthBounds        = {},
    //	.maxDepthBounds        = {},
    // };

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

    const std::array<vk::PipelineColorBlendAttachmentState, 1> colorBlendAttachments{
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

    const vk::PipelineColorBlendStateCreateInfo colorBlendState{
        .flags           = {},
        .logicOpEnable   = {},
        .logicOp         = vk::LogicOp::eClear,
        .attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size()),
        .pAttachments    = colorBlendAttachments.data(),
        .blendConstants  = std::array{0.0f, 0.0f, 0.0f, 0.0f},
    };

    const vk::PipelineLayoutCreateInfo layoutCreateInfo{
        .flags                  = {},
        .setLayoutCount         = 0,
        .pSetLayouts            = nullptr,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges    = nullptr,
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
                                      .pDepthStencilState  = nullptr,
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
                                  }};

    trianglePipeline = device.createGraphicsPipeline({nullptr}, createInfo.get());

    struct Vertex {
        glm::vec2 pos;
        glm::vec3 color;
    };
    std::array vertices = {
        Vertex{.pos = {0, -0.5}, .color = {1.0, 0.0, 0.0}},
        Vertex{.pos = {0.5, 0.5}, .color = {0.0, 1.0, 0.0}},
        Vertex{.pos = {-0.5, 0.5}, .color = {0.0, 0.0, 1.0}},
    };
    const auto bufferData = std::as_bytes(std::span{vertices});

    auto buffer = graphicsContext.getResourceFactory().CreateBuffer(
        {
            .flags       = {},
            .size        = bufferData.size_bytes(),
            .usage       = vk::BufferUsageFlagBits::eVertexBuffer,
            .sharingMode = vk::SharingMode::eExclusive,
        },
        CasualUsage::AutoMapped);
    // std::memcpy(allocationInfo.pMappedData, vertices.data(), bufferData.size_bytes());
    std::ranges::copy(bufferData, buffer.mappedMemory().begin());

    vertexBuffer = std::move(buffer);


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

        const std::array colorAttachments{vk::RenderingAttachmentInfo{.imageView   = graphicsContext.getSwapchainData().imageViews[frameImageIndex],
                                                                      .imageLayout = vk::ImageLayout::eAttachmentOptimal,
                                                                      .loadOp      = vk::AttachmentLoadOp::eClear,
                                                                      .storeOp     = vk::AttachmentStoreOp::eStore,
                                                                      .clearValue  = {vk::ClearColorValue{1.0f, 1.0f, 0.0f, 1.0f}}}};
        commandBuffer.beginRendering({
            .renderArea           = {.offset = {}, .extent = swapChainExtent},
            .layerCount           = 1,
            .colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size()),
            .pColorAttachments    = colorAttachments.data(),
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
        commandBuffer.bindVertexBuffers(0, *vertexBuffer, {0});

        commandBuffer.draw(3, 1, 0, 0);

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
    const std::array<vk::Semaphore, 1>          semaphoresForSignal = {frameSync.rederFinished};
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
        .pWaitSemaphores    = &*frameSync.rederFinished,
        .swapchainCount     = 1,
        .pSwapchains        = &*swapchain,
        .pImageIndices      = &frameImageIndex,
        .pResults           = nullptr,
    };

    graphicsContext.getPresentQueue().presentKHR(presentInfo);

    currentFrame = (currentFrame + 1) % num_inflight_frames;
}
