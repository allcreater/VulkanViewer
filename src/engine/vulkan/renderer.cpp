module;
#include <algorithm>
#include <numeric>
#include <memory>
#include <ranges>

#include <functional>
#include <iostream>
#include <format>
#include <filesystem>

#include <vulkan/vulkan_raii.hpp>

export module engine : vulkan.renderer;

import :utils;
import :vulkan.context;


struct Synchronization {
	explicit Synchronization(const vk::raii::Device& device)
		: imageAvailable{ device, vk::SemaphoreCreateInfo{} }
		, rederFinished{ device, vk::SemaphoreCreateInfo{} }
		, inFlightFence{ device, {.flags = vk::FenceCreateFlagBits::eSignaled} }
	{}

	vk::raii::Semaphore imageAvailable;
	vk::raii::Semaphore rederFinished;
	vk::raii::Fence inFlightFence;
};

export class VulkanRenderer final {
public:
	VulkanRenderer(VulkanGraphicsContext&& graphicsContext);
	~VulkanRenderer();

	void Render();

private:
	VulkanGraphicsContext graphicsContext;
	vk::raii::ShaderModule vs{ nullptr }, fs{ nullptr };
	vk::raii::PipelineLayout pipelineLayout{ nullptr };
	vk::raii::RenderPass renderPass{ nullptr };
	vk::raii::Pipeline trianglePipeline{nullptr};
	std::vector<vk::raii::Framebuffer> framebuffers;
	//vk::raii::CommandPool commandPool;
	//vk::raii::CommandBuffer commandBuffer;
	Synchronization sync;
};

namespace {

vk::raii::ShaderModule loadShaderModule(const vk::raii::Device& device, const std::filesystem::path& path) {
	const auto data = readFile(path);
	const std::span<const uint32_t> dataView{ reinterpret_cast<const uint32_t*>(data.data()), data.size() / sizeof(uint32_t)};

	vk::ShaderModuleCreateInfo create_info{
		.flags = {},
		.codeSize = dataView.size_bytes(),
		.pCode = dataView.data(),
	};

	return device.createShaderModule(create_info);
}

}


VulkanRenderer::VulkanRenderer(VulkanGraphicsContext&& _graphicsContext)
	: graphicsContext{std::move(_graphicsContext)}
	, sync { graphicsContext.getDevice() }
{
	const auto& device = graphicsContext.getDevice();

	vs = loadShaderModule(device, "data/shaders/hello_world.vs.spv");
	fs = loadShaderModule(device, "data/shaders/hello_world.fs.spv");
	
	const std::array< vk::PipelineShaderStageCreateInfo, 2> stages{
		vk::PipelineShaderStageCreateInfo{
			.flags = {},
			.stage = vk::ShaderStageFlagBits::eVertex,
			.module = vs,
			.pName = "main",
			.pSpecializationInfo = nullptr,
		}, vk::PipelineShaderStageCreateInfo{
			.flags = {},
			.stage = vk::ShaderStageFlagBits::eFragment,
			.module = fs,
			.pName = "main",
			.pSpecializationInfo = nullptr,
		},
	};

	const vk::PipelineVertexInputStateCreateInfo vertexInputState{
		.flags = {},
		.vertexBindingDescriptionCount = 0,
		.pVertexBindingDescriptions = {},
		.vertexAttributeDescriptionCount = 0,
		.pVertexAttributeDescriptions = {},
	};

	const vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState{
		.flags = {},
		.topology = vk::PrimitiveTopology::eTriangleList,
		.primitiveRestartEnable = vk::False,
	};

	const vk::PipelineTessellationStateCreateInfo tesselationState{
		.flags = {},
		.patchControlPoints = 0,
	};

	const auto swapChainExtent = graphicsContext.getExtent();
	const vk::Viewport viewport{
		.x = 0.0f,
		.y = 0.0f,
		.width = (float)swapChainExtent.width,
		.height = (float)swapChainExtent.height,
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};

	const vk::Rect2D scissor{
		.offset = {0, 0},
		.extent = swapChainExtent,
	};

	const vk::PipelineViewportStateCreateInfo viewportState{
		.flags = {},
		.viewportCount = 1,
		.pViewports = &viewport,
		.scissorCount = 1,
		.pScissors = &scissor,
	};

	vk::PipelineRasterizationStateCreateInfo rasterizationState{
		.flags = {},
		.depthClampEnable = vk::False,
		.rasterizerDiscardEnable = vk::False,
		.polygonMode = vk::PolygonMode::eFill,
		.cullMode = vk::CullModeFlagBits::eBack,
		.frontFace = vk::FrontFace::eClockwise,
		.depthBiasEnable = vk::False,
		.depthBiasConstantFactor = 0.0f,
		.depthBiasClamp = 0.0f,
		.depthBiasSlopeFactor = 0.0f,
		.lineWidth = 1.0f,
	};

	const vk::PipelineMultisampleStateCreateInfo multisampleState{
		.flags = {},
		.rasterizationSamples = vk::SampleCountFlagBits::e1,
		.sampleShadingEnable = vk::False,
		.minSampleShading = 1.0f,
		.pSampleMask = nullptr,
		.alphaToCoverageEnable = vk::False,
		.alphaToOneEnable = vk::False,
	};

	//const vk::PipelineDepthStencilStateCreateInfo depthStencilState{
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
	//};

	constexpr std::array<vk::DynamicState, 2> dynamicStates{
		vk::DynamicState::eViewport,
		vk::DynamicState::eScissor,
		//vk::DynamicState::eViewportWithCount,
		//vk::DynamicState::eScissorWithCount,
	};
	const vk::PipelineDynamicStateCreateInfo dynamicState{
		.flags = {},
		.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
		.pDynamicStates = dynamicStates.data(),
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
			.colorWriteMask      = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
		},
	};

	const vk::PipelineColorBlendStateCreateInfo colorBlendState{
		.flags = {},
		.logicOpEnable = {},
		.logicOp = VULKAN_HPP_NAMESPACE::LogicOp::eClear,
		.attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size()),
		.pAttachments = colorBlendAttachments.data(),
		.blendConstants = std::array{0.0f, 0.0f, 0.0f, 0.0f},
	};

	const vk::PipelineLayoutCreateInfo layoutCreateInfo{
		.flags = {},
		.setLayoutCount = 0,
		.pSetLayouts = nullptr,
		.pushConstantRangeCount = 0,
		.pPushConstantRanges = nullptr,
	};
	pipelineLayout = device.createPipelineLayout(layoutCreateInfo);


	const std::array<vk::AttachmentDescription, 1> attachmentDescriptions{
		vk::AttachmentDescription {
			.flags = {},
			.format = graphicsContext.getSurfaceFormat().format,
			.samples = vk::SampleCountFlagBits::e1,
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eStore,
			.stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
			.stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
			.initialLayout = vk::ImageLayout::eUndefined,
			.finalLayout = vk::ImageLayout::ePresentSrcKHR,
		},
	};
	
	const std::array<vk::AttachmentReference, 1> subpassAttachmentRefs{
		vk::AttachmentReference {
			.attachment = 0,
			.layout = vk::ImageLayout::eColorAttachmentOptimal,
		},
	};

	const std::array<vk::SubpassDescription, 1> subpassDescriptions{
		vk::SubpassDescription {
			.flags                   = {},
			.pipelineBindPoint       = vk::PipelineBindPoint::eGraphics,
			.inputAttachmentCount    = 0,
			.pInputAttachments       = nullptr,
			.colorAttachmentCount    = static_cast<uint32_t>(subpassAttachmentRefs.size()),
			.pColorAttachments       = subpassAttachmentRefs.data(),
			.pResolveAttachments     = nullptr,
			.pDepthStencilAttachment = nullptr,
			.preserveAttachmentCount = 0,
			.pPreserveAttachments    = nullptr,
		}
	};

	const std::array<vk::SubpassDependency, 1> dependencies{
		vk::SubpassDependency{
			.srcSubpass      = vk::SubpassExternal,
			.dstSubpass      = 0,
			.srcStageMask    = vk::PipelineStageFlagBits::eColorAttachmentOutput,
			.dstStageMask    = vk::PipelineStageFlagBits::eColorAttachmentOutput,
			.srcAccessMask   = {},
			.dstAccessMask   = vk::AccessFlagBits::eColorAttachmentWrite,
			.dependencyFlags = {},
		},
	};

	const vk::RenderPassCreateInfo renderPassCreateInfo{
		.flags           = {},
		.attachmentCount = static_cast<uint32_t>(attachmentDescriptions.size()),
		.pAttachments    = attachmentDescriptions.data(),
		.subpassCount    = static_cast<uint32_t>(subpassDescriptions.size()),
		.pSubpasses      = subpassDescriptions.data(),
		.dependencyCount = static_cast<uint32_t>(dependencies.size()),
		.pDependencies   = dependencies.data(),
	};

	renderPass = device.createRenderPass(renderPassCreateInfo);

	const vk::GraphicsPipelineCreateInfo create_info{
		.flags = {},
		.stageCount = static_cast<uint32_t>(stages.size()),
		.pStages = stages.data(),
		.pVertexInputState = &vertexInputState,
		.pInputAssemblyState = &inputAssemblyState,
		.pTessellationState = &tesselationState,
		.pViewportState = &viewportState,
		.pRasterizationState = &rasterizationState,
		.pMultisampleState = &multisampleState,
		.pDepthStencilState = nullptr,
		.pColorBlendState = &colorBlendState,
		.pDynamicState = &dynamicState,
		.layout = pipelineLayout,
		.renderPass = renderPass,
		.subpass = 0,
		.basePipelineHandle = {nullptr},
		.basePipelineIndex = -1,
	};

	trianglePipeline = device.createGraphicsPipeline({nullptr}, create_info);



	// framebuffers
	const auto makeFramebuffer = [renderPass = *renderPass, swapChainExtent, &device](vk::ImageView imageView) {
		const vk::FramebufferCreateInfo createInfo{
			.flags = {},
			.renderPass = renderPass,
			.attachmentCount = 1,
			.pAttachments = &imageView,
			.width = swapChainExtent.width,
			.height = swapChainExtent.height,
			.layers = 1,
		};

		return device.createFramebuffer(createInfo);
	};

	framebuffers = graphicsContext.getSwapchainData().imageViews | std::views::transform(makeFramebuffer) | std::ranges::to<std::vector>();

}

VulkanRenderer::~VulkanRenderer() {
	const vk::raii::Device& device = graphicsContext.getDevice();
	device.waitIdle();
}

void VulkanRenderer::Render() {
	const vk::raii::Device& device = graphicsContext.getDevice();
	const vk::raii::CommandBuffer& commandBuffer = graphicsContext.getCommandBuffer();
	const auto swapChainExtent = graphicsContext.getExtent();

	device.waitForFences(std::array{ *sync.inFlightFence }, vk::True, std::numeric_limits<uint64_t>::max());
	device.resetFences(std::array{ *sync.inFlightFence });

	const vk::raii::SwapchainKHR& swapchain = graphicsContext.getSwapchainData().swapchain;
	const auto [success, frameImageIndex] = swapchain.acquireNextImage(std::numeric_limits<uint64_t>::max(), sync.imageAvailable, nullptr);

	commandBuffer.reset();
	{
		commandBuffer.begin({});

		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, trianglePipeline);

		const std::array<vk::ClearValue, 1> clearValues = {
			vk::ClearValue{vk::ClearColorValue{1.0f, 1.0f, 0.0f, 1.0f}},
		};
		const vk::RenderPassBeginInfo renderPassBeginInfo{
			.renderPass = renderPass,
			.framebuffer = framebuffers[frameImageIndex],
			.renderArea = {.offset = {}, .extent = swapChainExtent },
			.clearValueCount = static_cast<uint32_t>(clearValues.size()),
			.pClearValues = clearValues.data(),
		};
		commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

		const std::array<vk::Viewport, 1> viewports{
			vk::Viewport{
				.x = 0.0f,
				.y = 0.0f,
				.width = (float)swapChainExtent.width,
				.height = (float)swapChainExtent.height,
				.minDepth = 0.0f,
				.maxDepth = 1.0f,
			},
		};
		commandBuffer.setViewport(0, viewports);


		const std::array<vk::Rect2D, 1> scissorRects{
			vk::Rect2D {
				.offset = {0, 0},
				.extent = swapChainExtent,
			},
		};
		commandBuffer.setScissor(0, scissorRects);

		commandBuffer.draw(3, 1, 0, 0);

		commandBuffer.endRenderPass();
		commandBuffer.end();
	}

	const std::array<vk::Semaphore, 1> semaphoresForWait = { sync.imageAvailable };
	const std::array<vk::Semaphore, 1> semaphoresForSignal = { sync.rederFinished };
	const std::array<vk::PipelineStageFlags, 1> waitStages = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
	const vk::SubmitInfo submitInfo{
		.waitSemaphoreCount   = static_cast<uint32_t>(semaphoresForWait.size()),
		.pWaitSemaphores      = semaphoresForWait.data(),
		.pWaitDstStageMask    = waitStages.data(),
		.commandBufferCount   = 1,
		.pCommandBuffers      = &*commandBuffer,
		.signalSemaphoreCount = static_cast<uint32_t>(semaphoresForSignal.size()),
		.pSignalSemaphores    = semaphoresForSignal.data(),
	};

	graphicsContext.getGraphicsQueue().submit(submitInfo, sync.inFlightFence);

	const vk::PresentInfoKHR presentInfo{
		.waitSemaphoreCount = 1,
		.pWaitSemaphores    = &*sync.rederFinished,
		.swapchainCount     = 1,
		.pSwapchains        = &*swapchain,
		.pImageIndices      = &frameImageIndex,
		.pResults           = nullptr,
	};

	graphicsContext.getPresentQueue().presentKHR(presentInfo);

}