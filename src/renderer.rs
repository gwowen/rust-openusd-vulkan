//! Vulkan renderer: one lit pass for the USD meshes on a sky-blue clear.

use std::ffi::{CStr, c_void};
use std::io::Cursor;

use anyhow::{Context, Result, bail};
use ash::khr::{surface, swapchain};
use ash::vk::{self, Handle};
use ash_window::{create_surface, enumerate_required_extensions};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

use crate::scene::{SceneObject, Vertex};

const FRAMES_IN_FLIGHT: usize = 2;
const SKY_BLUE: [f32; 4] = [0.53, 0.81, 0.92, 1.0];

const VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shader.vert.spv"));
const FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shader.frag.spv"));

pub struct Renderer {
    window: Window,
    /// The loader must outlive the instance; its function pointers are not copied.
    #[allow(dead_code)]
    entry: ash::Entry,
    instance: ash::Instance,
    debug: Option<DebugMessenger>,
    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    allocator: Option<Allocator>,
    swapchain_loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    color_format: vk::Format,
    depth_format: vk::Format,
    extent: vk::Extent2D,
    image_views: Vec<vk::ImageView>,
    depth_image: vk::Image,
    depth_allocation: Option<Allocation>,
    depth_view: vk::ImageView,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    render_finished: Vec<vk::Semaphore>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_layout: vk::DescriptorSetLayout,
    vertex_buffer: vk::Buffer,
    vertex_allocation: Option<Allocation>,
    ranges: Vec<MeshRange>,
    frames: Vec<Frame>,
    frame_index: usize,
    framebuffer_resized: bool,
    non_coherent_atom_size: u64,
}

struct MeshRange {
    first_vertex: u32,
    vertex_count: u32,
}

struct Frame {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    image_available: vk::Semaphore,
    uniform: vk::Buffer,
    uniform_allocation: Option<Allocation>,
    descriptor_set: vk::DescriptorSet,
}

struct DebugMessenger {
    loader: ash::ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl Renderer {
    pub fn new(window: Window, objects: &[SceneObject]) -> Result<Self> {
        if objects.is_empty() {
            bail!("the stage has no meshes to draw");
        }
        let entry = unsafe { ash::Entry::load() }.context("failed to load the Vulkan loader")?;
        let display = window.display_handle()?.as_raw();
        let window_handle = window.window_handle()?.as_raw();

        let validation = has_validation(&entry);
        let mut extension_names = enumerate_required_extensions(display)?.to_vec();
        if validation {
            extension_names.push(ash::ext::debug_utils::NAME.as_ptr());
        }
        let layer_storage = [c"VK_LAYER_KHRONOS_validation".as_ptr()];
        let enabled_layers: &[*const i8] = if validation { &layer_storage } else { &[] };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"rust-openusd-vulkan")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"rust-openusd-vulkan")
            .api_version(vk::API_VERSION_1_2);
        let mut debug_info = debug_create_info();
        let mut instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(enabled_layers);
        if validation {
            instance_info = instance_info.push_next(&mut debug_info);
        }
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .context("failed to create the Vulkan instance")?;
        let debug = if validation {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let messenger = unsafe { loader.create_debug_utils_messenger(&debug_info, None)? };
            Some(DebugMessenger { loader, messenger })
        } else {
            None
        };

        let surface_loader = surface::Instance::new(&entry, &instance);
        let surface = unsafe { create_surface(&entry, &instance, display, window_handle, None) }
            .context("failed to create the window surface")?;
        let (physical_device, queue_family) = pick_device(&instance, &surface_loader, surface)?;
        let queue_priorities = [1.0_f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priorities);
        let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extensions);
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .context("failed to create the Vulkan device")?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let limits = unsafe { instance.get_physical_device_properties(physical_device) }.limits;
        let mut allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;
        let depth_format = pick_depth_format(&instance, physical_device)?;

        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);
        let descriptor_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(std::slice::from_ref(&binding)),
                None,
            )?
        };
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(FRAMES_IN_FLIGHT as u32);
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(FRAMES_IN_FLIGHT as u32)
                    .pool_sizes(std::slice::from_ref(&pool_size)),
                None,
            )?
        };
        let layouts = vec![descriptor_layout; FRAMES_IN_FLIGHT];
        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )?
        };

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<[[f32; 4]; 4]>() as u32);
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&descriptor_layout))
                    .push_constant_ranges(std::slice::from_ref(&push_range)),
                None,
            )?
        };

        let (vertices, ranges) = pack_vertices(objects);
        let (vertex_buffer, vertex_allocation) = create_buffer(
            &device,
            &mut allocator,
            bytemuck::cast_slice(&vertices),
            vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryLocation::CpuToGpu,
            limits.non_coherent_atom_size,
        )?;

        let mut frames = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for descriptor_set in descriptor_sets {
            let (uniform, uniform_allocation) = create_buffer(
                &device,
                &mut allocator,
                bytemuck::bytes_of(&[[0.0_f32; 4]; 4]),
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
                limits.non_coherent_atom_size,
            )?;
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform)
                .offset(0)
                .range(std::mem::size_of::<[[f32; 4]; 4]>() as u64);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info));
            unsafe { device.update_descriptor_sets(&[write], &[]) };

            let command_pool = unsafe {
                device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(queue_family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )?
            };
            let command_buffer = unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )?[0]
            };
            let fence = unsafe {
                device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?
            };
            let image_available =
                unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };
            frames.push(Frame {
                command_pool,
                command_buffer,
                fence,
                image_available,
                uniform,
                uniform_allocation: Some(uniform_allocation),
                descriptor_set,
            });
        }

        let swapchain_loader = swapchain::Device::new(&instance, &device);
        let mut renderer = Self {
            window,
            entry,
            instance,
            debug,
            surface_loader,
            surface,
            physical_device,
            device,
            queue,
            allocator: Some(allocator),
            swapchain_loader,
            swapchain: vk::SwapchainKHR::null(),
            color_format: vk::Format::UNDEFINED,
            depth_format,
            extent: vk::Extent2D::default(),
            image_views: Vec::new(),
            depth_image: vk::Image::null(),
            depth_allocation: None,
            depth_view: vk::ImageView::null(),
            render_pass: vk::RenderPass::null(),
            framebuffers: Vec::new(),
            render_finished: Vec::new(),
            pipeline_layout,
            pipeline: vk::Pipeline::null(),
            descriptor_pool,
            descriptor_layout,
            vertex_buffer,
            vertex_allocation: Some(vertex_allocation),
            ranges,
            frames,
            frame_index: 0,
            framebuffer_resized: false,
            non_coherent_atom_size: limits.non_coherent_atom_size,
        };
        renderer.rebuild_swapchain()?;
        Ok(renderer)
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn note_resize(&mut self) {
        self.framebuffer_resized = true;
    }

    pub fn draw(&mut self, translations: &[[f32; 3]]) -> Result<()> {
        if translations.len() != self.ranges.len() {
            bail!("translation count does not match the meshes");
        }
        if self.framebuffer_resized || self.swapchain.is_null() {
            self.rebuild_swapchain()?;
            self.framebuffer_resized = false;
        }
        if self.extent.width == 0 || self.extent.height == 0 || self.swapchain.is_null() {
            return Ok(());
        }

        let frame_index = self.frame_index;
        let fence = self.frames[frame_index].fence;
        unsafe { self.device.wait_for_fences(&[fence], true, u64::MAX)? };

        let image_available = self.frames[frame_index].image_available;
        let image_index = match unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                image_available,
                vk::Fence::null(),
            )
        } {
            Ok((index, suboptimal)) => {
                if suboptimal {
                    self.framebuffer_resized = true;
                }
                index
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.framebuffer_resized = true;
                return Ok(());
            }
            Err(err) => return Err(err.into()),
        };

        unsafe { self.device.reset_fences(&[fence])? };
        self.write_uniform(frame_index)?;
        self.record(frame_index, image_index, translations)?;

        let command_buffer = self.frames[frame_index].command_buffer;
        let render_finished = self.render_finished[image_index as usize];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(std::slice::from_ref(&image_available))
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&command_buffer))
            .signal_semaphores(std::slice::from_ref(&render_finished));
        unsafe { self.device.queue_submit(self.queue, &[submit], fence)? };

        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(std::slice::from_ref(&render_finished))
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        match unsafe { self.swapchain_loader.queue_present(self.queue, &present) } {
            Ok(suboptimal) => {
                if suboptimal {
                    self.framebuffer_resized = true;
                }
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.framebuffer_resized = true,
            Err(err) => return Err(err.into()),
        }
        self.frame_index = (frame_index + 1) % FRAMES_IN_FLIGHT;
        Ok(())
    }

    fn write_uniform(&mut self, frame_index: usize) -> Result<()> {
        let size = self.window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let view_proj = view_projection(aspect);
        let bytes = bytemuck::bytes_of(&view_proj);
        let allocation = self.frames[frame_index]
            .uniform_allocation
            .as_mut()
            .context("uniform buffer is missing")?;
        let mapped = allocation
            .mapped_slice_mut()
            .context("uniform buffer is not mapped")?;
        mapped[..bytes.len()].copy_from_slice(bytes);
        flush_allocation(&self.device, allocation, self.non_coherent_atom_size)
    }

    fn record(
        &mut self,
        frame_index: usize,
        image_index: u32,
        translations: &[[f32; 3]],
    ) -> Result<()> {
        let command_buffer = self.frames[frame_index].command_buffer;
        let descriptor_set = self.frames[frame_index].descriptor_set;
        let device = &self.device;
        unsafe {
            device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue { float32: SKY_BLUE },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];
        let begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[image_index as usize])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.extent,
            })
            .clear_values(&clear_values);
        unsafe {
            device.cmd_begin_render_pass(command_buffer, &begin, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.vertex_buffer], &[0]);
            device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: self.extent.width as f32,
                    height: self.extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device.cmd_set_scissor(
                command_buffer,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: self.extent,
                }],
            );
        }

        for (range, translation) in self.ranges.iter().zip(translations) {
            let model = translation_matrix(*translation);
            unsafe {
                device.cmd_push_constants(
                    command_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    bytemuck::bytes_of(&model),
                );
                device.cmd_draw(command_buffer, range.vertex_count, 1, range.first_vertex, 0);
            }
        }

        unsafe {
            device.cmd_end_render_pass(command_buffer);
            device.end_command_buffer(command_buffer)?;
        }
        Ok(())
    }

    fn rebuild_swapchain(&mut self) -> Result<()> {
        if !self.swapchain.is_null() || !self.render_pass.is_null() {
            unsafe { self.device.device_wait_idle()? };
        }
        self.destroy_swapchain_objects();

        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            self.extent = vk::Extent2D::default();
            return Ok(());
        }

        let capabilities = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)?
        };
        let formats = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(self.physical_device, self.surface)?
        };
        let surface_format = formats
            .iter()
            .copied()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| formats.first().copied())
            .context("the surface has no formats")?;
        self.color_format = surface_format.format;
        self.extent = if capabilities.current_extent.width == u32::MAX {
            vk::Extent2D {
                width: size.width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: size.height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        } else {
            capabilities.current_extent
        };
        if self.extent.width == 0 || self.extent.height == 0 {
            return Ok(());
        }

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count > 0 {
            image_count = image_count.min(capabilities.max_image_count);
        }
        let alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::INHERIT,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        ]
        .into_iter()
        .find(|flag| capabilities.supported_composite_alpha.contains(*flag))
        .unwrap_or(vk::CompositeAlphaFlagsKHR::OPAQUE);
        let transform = if capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            capabilities.current_transform
        };

        self.swapchain = unsafe {
            self.swapchain_loader.create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .surface(self.surface)
                    .min_image_count(image_count)
                    .image_format(self.color_format)
                    .image_color_space(surface_format.color_space)
                    .image_extent(self.extent)
                    .image_array_layers(1)
                    .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(transform)
                    .composite_alpha(alpha)
                    .present_mode(vk::PresentModeKHR::FIFO)
                    .clipped(true),
                None,
            )?
        };
        let images = unsafe { self.swapchain_loader.get_swapchain_images(self.swapchain)? };
        self.image_views = images
            .iter()
            .map(|image| {
                image_view(
                    &self.device,
                    *image,
                    self.color_format,
                    vk::ImageAspectFlags::COLOR,
                )
            })
            .collect::<Result<_>>()?;
        self.create_depth()?;
        self.create_render_pass()?;
        self.create_pipeline()?;
        self.framebuffers = self
            .image_views
            .iter()
            .map(|view| {
                let attachments = [*view, self.depth_view];
                unsafe {
                    self.device.create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(self.render_pass)
                            .attachments(&attachments)
                            .width(self.extent.width)
                            .height(self.extent.height)
                            .layers(1),
                        None,
                    )
                }
                .map_err(Into::into)
            })
            .collect::<Result<_>>()?;
        self.render_finished = (0..images.len())
            .map(|_| unsafe {
                self.device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            })
            .collect::<std::result::Result<_, _>>()?;
        Ok(())
    }

    fn create_depth(&mut self) -> Result<()> {
        self.depth_image = unsafe {
            self.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(self.depth_format)
                    .extent(vk::Extent3D {
                        width: self.extent.width,
                        height: self.extent.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?
        };
        let requirements = unsafe { self.device.get_image_memory_requirements(self.depth_image) };
        let allocation = self
            .allocator
            .as_mut()
            .context("allocator dropped")?
            .allocate(&AllocationCreateDesc {
                name: "depth",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::DedicatedImage(self.depth_image),
            })?;
        unsafe {
            self.device.bind_image_memory(
                self.depth_image,
                allocation.memory(),
                allocation.offset(),
            )?
        };
        self.depth_allocation = Some(allocation);
        self.depth_view = image_view(
            &self.device,
            self.depth_image,
            self.depth_format,
            vk::ImageAspectFlags::DEPTH,
        )?;
        Ok(())
    }

    fn create_render_pass(&mut self) -> Result<()> {
        let attachments = [
            vk::AttachmentDescription::default()
                .format(self.color_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
            vk::AttachmentDescription::default()
                .format(self.depth_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        ];
        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_ref))
            .depth_stencil_attachment(&depth_ref);
        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );
        self.render_pass = unsafe {
            self.device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&attachments)
                    .subpasses(std::slice::from_ref(&subpass))
                    .dependencies(std::slice::from_ref(&dependency)),
                None,
            )?
        };
        Ok(())
    }

    fn create_pipeline(&mut self) -> Result<()> {
        let vert_code = ash::util::read_spv(&mut Cursor::new(VERT_SPV))?;
        let frag_code = ash::util::read_spv(&mut Cursor::new(FRAG_SPV))?;
        let vert_module = unsafe {
            self.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&vert_code),
                None,
            )?
        };
        let frag_module = unsafe {
            self.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&frag_code),
                None,
            )?
        };
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(c"vertexMain"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(c"fragmentMain"),
        ];
        let binding = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX);
        let attributes = [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(12),
            vk::VertexInputAttributeDescription::default()
                .location(2)
                .binding(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(24),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding))
            .vertex_attribute_descriptions(&attributes);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);
        let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        let blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&blend_attachment));
        let dynamic = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(self.pipeline_layout)
            .render_pass(self.render_pass)
            .subpass(0);
        let pipelines = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        };
        unsafe {
            self.device.destroy_shader_module(vert_module, None);
            self.device.destroy_shader_module(frag_module, None);
        }
        self.pipeline = match pipelines {
            Ok(pipelines) => pipelines[0],
            Err((_, err)) => return Err(err.into()),
        };
        Ok(())
    }

    fn destroy_swapchain_objects(&mut self) {
        let device = &self.device;
        unsafe {
            for semaphore in self.render_finished.drain(..) {
                device.destroy_semaphore(semaphore, None);
            }
            for framebuffer in self.framebuffers.drain(..) {
                device.destroy_framebuffer(framebuffer, None);
            }
            for view in self.image_views.drain(..) {
                device.destroy_image_view(view, None);
            }
            if !self.depth_view.is_null() {
                device.destroy_image_view(self.depth_view, None);
                self.depth_view = vk::ImageView::null();
            }
            if !self.depth_image.is_null() {
                device.destroy_image(self.depth_image, None);
                self.depth_image = vk::Image::null();
            }
            if let Some(allocation) = self.depth_allocation.take() {
                if let Some(allocator) = self.allocator.as_mut() {
                    let _ = allocator.free(allocation);
                }
            }
            if !self.pipeline.is_null() {
                device.destroy_pipeline(self.pipeline, None);
                self.pipeline = vk::Pipeline::null();
            }
            if !self.render_pass.is_null() {
                device.destroy_render_pass(self.render_pass, None);
                self.render_pass = vk::RenderPass::null();
            }
            if !self.swapchain.is_null() {
                self.swapchain_loader
                    .destroy_swapchain(self.swapchain, None);
                self.swapchain = vk::SwapchainKHR::null();
            }
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        self.destroy_swapchain_objects();
        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
            if !self.vertex_buffer.is_null() {
                self.device.destroy_buffer(self.vertex_buffer, None);
                self.vertex_buffer = vk::Buffer::null();
            }
            for frame in &self.frames {
                self.device.destroy_buffer(frame.uniform, None);
                self.device.destroy_semaphore(frame.image_available, None);
                self.device.destroy_fence(frame.fence, None);
                self.device.destroy_command_pool(frame.command_pool, None);
            }
        }
        if let Some(allocator) = self.allocator.as_mut() {
            if let Some(allocation) = self.vertex_allocation.take() {
                let _ = allocator.free(allocation);
            }
            for frame in &mut self.frames {
                if let Some(allocation) = frame.uniform_allocation.take() {
                    let _ = allocator.free(allocation);
                }
            }
        }
        drop(self.allocator.take());
        unsafe {
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            if let Some(debug) = self.debug.take() {
                debug
                    .loader
                    .destroy_debug_utils_messenger(debug.messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn has_validation(entry: &ash::Entry) -> bool {
    let Ok(layers) = (unsafe { entry.enumerate_instance_layer_properties() }) else {
        return false;
    };
    layers
        .iter()
        .any(|layer| layer.layer_name_as_c_str().ok() == Some(c"VK_LAYER_KHRONOS_validation"))
}

fn debug_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback))
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*data).p_message) };
    eprintln!(
        "[vulkan {severity:?} {message_type:?}] {}",
        message.to_string_lossy()
    );
    vk::FALSE
}

fn pick_device(
    instance: &ash::Instance,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32)> {
    let devices = unsafe { instance.enumerate_physical_devices() }?;
    let mut fallback = None;
    for device in devices {
        let Some(family) = queue_family(instance, surface_loader, device, surface)? else {
            continue;
        };
        let properties = unsafe { instance.get_physical_device_properties(device) };
        if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            return Ok((device, family));
        }
        fallback.get_or_insert((device, family));
    }
    fallback.context("no Vulkan device can present to the window")
}

fn queue_family(
    instance: &ash::Instance,
    surface_loader: &surface::Instance,
    device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<Option<u32>> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(device) };
    for (index, family) in families.iter().enumerate() {
        let graphics = family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
        let present = unsafe {
            surface_loader.get_physical_device_surface_support(device, index as u32, surface)?
        };
        if graphics && present {
            return Ok(Some(index as u32));
        }
    }
    Ok(None)
}

fn pick_depth_format(instance: &ash::Instance, device: vk::PhysicalDevice) -> Result<vk::Format> {
    for format in [
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ] {
        let properties = unsafe { instance.get_physical_device_format_properties(device, format) };
        if properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        {
            return Ok(format);
        }
    }
    bail!("no depth format supports depth attachments")
}

fn pack_vertices(objects: &[SceneObject]) -> (Vec<Vertex>, Vec<MeshRange>) {
    let mut vertices = Vec::new();
    let mut ranges = Vec::new();
    for object in objects {
        let first_vertex = vertices.len() as u32;
        vertices.extend_from_slice(&object.vertices);
        ranges.push(MeshRange {
            first_vertex,
            vertex_count: object.vertices.len() as u32,
        });
    }
    (vertices, ranges)
}

fn image_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
) -> Result<vk::ImageView> {
    Ok(unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(aspect)
                        .level_count(1)
                        .layer_count(1),
                ),
            None,
        )?
    })
}

fn create_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    bytes: &[u8],
    usage: vk::BufferUsageFlags,
    location: MemoryLocation,
    atom_size: u64,
) -> Result<(vk::Buffer, Allocation)> {
    let buffer = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(bytes.len() as u64)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )?
    };
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mut allocation = allocator.allocate(&AllocationCreateDesc {
        name: "buffer",
        requirements,
        location,
        linear: true,
        allocation_scheme: AllocationScheme::DedicatedBuffer(buffer),
    })?;
    if let Some(mapped) = allocation.mapped_slice_mut() {
        mapped[..bytes.len()].copy_from_slice(bytes);
    }
    flush_allocation(device, &allocation, atom_size)?;
    unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())? };
    Ok((buffer, allocation))
}

fn flush_allocation(device: &ash::Device, allocation: &Allocation, _atom_size: u64) -> Result<()> {
    if allocation
        .memory_properties()
        .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        || allocation.mapped_ptr().is_none()
    {
        return Ok(());
    }
    unsafe {
        device.flush_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
            .memory(allocation.memory())
            .offset(allocation.offset())
            .size(vk::WHOLE_SIZE)])?;
    }
    Ok(())
}

fn view_projection(aspect: f32) -> [[f32; 4]; 4] {
    let projection = perspective(45.0_f32.to_radians(), aspect, 0.1, 100.0);
    let view = look_at([7.5, 5.5, 8.5], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    mul_mat(projection, view)
}

fn translation_matrix(translation: [f32; 3]) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [translation[0], translation[1], translation[2], 1.0],
    ]
}

fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let focal = 1.0 / (fovy * 0.5).tan();
    let mut matrix = [[0.0; 4]; 4];
    matrix[0][0] = focal / aspect;
    matrix[1][1] = -focal;
    matrix[2][2] = far / (near - far);
    matrix[2][3] = -1.0;
    matrix[3][2] = (far * near) / (near - far);
    matrix
}

fn look_at(eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let forward = normalize(sub(center, eye));
    let side = normalize(cross(forward, up));
    let up = cross(side, forward);
    [
        [side[0], up[0], -forward[0], 0.0],
        [side[1], up[1], -forward[1], 0.0],
        [side[2], up[2], -forward[2], 0.0],
        [-dot(side, eye), -dot(up, eye), dot(forward, eye), 1.0],
    ]
}

fn mul_mat(left: [[f32; 4]; 4], right: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = left[0][row] * right[col][0]
                + left[1][row] * right[col][1]
                + left[2][row] * right[col][2]
                + left[3][row] * right[col][3];
        }
    }
    out
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = dot(v, v).sqrt();
    [v[0] / length, v[1] / length, v[2] / length]
}
