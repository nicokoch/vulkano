use std::ffi::CString;
use std::ptr;
use std::error;
use std::fmt;
use num_traits::ToPrimitive;

use device::DeviceOwned;

use check_errors;
use Error;
use VulkanObject;
use vk;

// TODO Queue, DescriptorPool, Semaphore
pub trait DebugMarker {
    fn set_object_name(&mut self, name: &str) -> Result<(), DebugMarkerError>;
    fn set_object_tag(&mut self, tag_name: u64, tag: &[u8]) -> Result<(), DebugMarkerError>;
}

impl <T> DebugMarker for T where T: DebugObject + DeviceOwned {
    fn set_object_name(&mut self, name: &str) -> Result<(), DebugMarkerError> {
        if !self.device().loaded_extensions().ext_debug_marker {
            return Err(DebugMarkerError::MissingExtension);
        }

        let vk = self.device().pointers();
        let name_ffi = CString::new(name).unwrap();
        let mut nameInfo = vk::DebugMarkerObjectNameInfoEXT {
            sType: vk::STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_NAME_INFO_EXT,
            pNext: ptr::null(),
            objectType: T::object_type(),
            object: self.internal_object().to_u64().unwrap(),
            pObjectName: name_ffi.as_ptr(),
        };

        unsafe {
            try!(check_errors(vk.DebugMarkerSetObjectNameEXT(self.device().internal_object(),
                                                             &mut nameInfo as *mut _)));
        }
        Ok(())
    }

    fn set_object_tag(&mut self, tag_name: u64, tag: &[u8]) -> Result<(), DebugMarkerError> {
        if !self.device().loaded_extensions().ext_debug_marker {
            return Err(DebugMarkerError::MissingExtension);
        }
        assert!(tag_name != 0);
        let vk = self.device().pointers();
        let mut tagInfo = vk::DebugMarkerObjectTagInfoEXT {
            sType: vk::STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_NAME_INFO_EXT,
            pNext: ptr::null(),
            objectType: T::object_type(),
            object: self.internal_object().to_u64().unwrap(),
            tagName: tag_name,
            tagSize: tag.len(),
            pTag: tag.as_ptr() as *const _,
        };

        unsafe {
            try!(check_errors(vk.DebugMarkerSetObjectTagEXT(self.device().internal_object(),
                                                            &mut tagInfo as *mut _)));
        }
        Ok(())
    }
}

/// Error that can happen when creating a debug callback.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DebugMarkerError {
    OutOfHostMemory,
    OutOfDeviceMemory,
    /// The `VK_EXT_debug_marker` extension was not enabled.
    MissingExtension,
}

impl error::Error for DebugMarkerError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            DebugMarkerError::OutOfHostMemory => "out of host memory",
            DebugMarkerError::OutOfDeviceMemory => "out of device memory",
            DebugMarkerError::MissingExtension => "the `VK_EXT_debug_marker` extension was \
                                                   not enabled",
        }
    }
}

impl fmt::Display for DebugMarkerError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<Error> for DebugMarkerError {
    #[inline]
    fn from(err: Error) -> DebugMarkerError {
        match err {
            Error::OutOfHostMemory => DebugMarkerError::OutOfHostMemory,
            Error::OutOfDeviceMemory => DebugMarkerError::OutOfDeviceMemory,
            _ => unreachable!()
        }
    }
}

pub trait DebugObject: VulkanObject {
    fn object_type() -> vk::DebugReportObjectTypeEXT;
}

macro_rules! impl_debug_object {
    ($obj_type:ty, $obj_debug_type:path) => {
        impl DebugObject for $obj_type {
            #[inline]
            fn object_type() -> vk::DebugReportObjectTypeEXT {
                $obj_debug_type
            }
        }
    };
}

// The various implementations for `DebugObject`
use instance::{PhysicalDevice, Instance};
use device::Queue;
use memory::DeviceMemory;
use buffer::sys::UnsafeBuffer;
use image::sys::UnsafeImage;
use sync::{Event, Fence, Semaphore};
use query::UnsafeQueryPool;
use buffer::view::BufferView;
use image::sys::UnsafeImageView;
use pipeline::shader::ShaderModule;
use pipeline::cache::PipelineCache;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use command_buffer::pool::CommandPool;
use sampler::Sampler;
use descriptor::descriptor_set::DescriptorPool;
use descriptor::descriptor_set::UnsafeDescriptorSet;
use framebuffer::FramebufferSys;
use command_buffer::pool::UnsafeCommandPool;
use command_buffer::sys::UnsafeCommandBuffer;
use swapchain::{Swapchain, Surface};
use instance::debug::DebugCallback;
use swapchain::display::Display;
use swapchain::display::DisplayMode;
use buffer::BufferAccess;

impl_debug_object!(Instance, vk::DEBUG_REPORT_OBJECT_TYPE_INSTANCE_EXT);
//impl_debug_object!(Queue, vk::DEBUG_REPORT_OBJECT_TYPE_QUEUE_EXT); TODO SynchronizedVulkanObject
impl_debug_object!(Semaphore, vk::DEBUG_REPORT_OBJECT_TYPE_SEMAPHORE_EXT);
impl_debug_object!(Fence, vk::DEBUG_REPORT_OBJECT_TYPE_FENCE_EXT);
impl_debug_object!(DeviceMemory, vk::DEBUG_REPORT_OBJECT_TYPE_DEVICE_MEMORY_EXT);
impl_debug_object!(UnsafeBuffer, vk::DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT);
impl_debug_object!(UnsafeImage, vk::DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT);
impl_debug_object!(Event, vk::DEBUG_REPORT_OBJECT_TYPE_EVENT_EXT);
impl_debug_object!(UnsafeQueryPool, vk::DEBUG_REPORT_OBJECT_TYPE_QUERY_POOL_EXT);
impl_debug_object!(UnsafeImageView, vk::DEBUG_REPORT_OBJECT_TYPE_IMAGE_VIEW_EXT);
impl_debug_object!(ShaderModule, vk::DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT);
impl_debug_object!(PipelineCache, vk::DEBUG_REPORT_OBJECT_TYPE_PIPELINE_CACHE_EXT);
//impl_debug_object!(UnsafePipelineLayout, vk::DEBUG_REPORT_OBJECT_TYPE_PIPELINE_LAYOUT_EXT);
//impl_debug_object!(UnsafeRenderPass, vk::DEBUG_REPORT_OBJECT_TYPE_RENDER_PASS_EXT);
//impl_debug_object!(GraphicsPipeline, vk::DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT); TODO renderpass
impl_debug_object!(UnsafeDescriptorSetLayout, vk::DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT_EXT);
impl_debug_object!(Sampler, vk::DEBUG_REPORT_OBJECT_TYPE_SAMPLER_EXT);
//impl_debug_object!(DescriptorPool, vk::DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_POOL_EXT); TODO SynchronizedVulkanObject
impl_debug_object!(UnsafeDescriptorSet, vk::DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_EXT);
impl_debug_object!(UnsafeCommandPool, vk::DEBUG_REPORT_OBJECT_TYPE_COMMAND_POOL_EXT);
impl_debug_object!(Surface, vk::DEBUG_REPORT_OBJECT_TYPE_SURFACE_KHR_EXT);
impl_debug_object!(Swapchain, vk::DEBUG_REPORT_OBJECT_TYPE_SWAPCHAIN_KHR_EXT);
impl_debug_object!(DebugCallback, vk::DEBUG_REPORT_OBJECT_TYPE_DEBUG_REPORT_EXT);
impl_debug_object!(Display, vk::DEBUG_REPORT_OBJECT_TYPE_DISPLAY_KHR_EXT);
impl_debug_object!(DisplayMode, vk::DEBUG_REPORT_OBJECT_TYPE_DISPLAY_MODE_KHR_EXT);

// Impls not possible with macro

impl<'a> DebugObject for PhysicalDevice<'a> {
    #[inline]
    fn object_type() -> vk::DebugReportObjectTypeEXT {
        vk::DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT
    }
}

impl<P: CommandPool> DebugObject for UnsafeCommandBuffer<P> {
    #[inline]
    fn object_type() -> vk::DebugReportObjectTypeEXT {
        vk::DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT
    }
}

impl<F, B: BufferAccess> DebugObject for BufferView<F, B> {
    #[inline]
    fn object_type() -> vk::DebugReportObjectTypeEXT {
        vk::DEBUG_REPORT_OBJECT_TYPE_BUFFER_VIEW_EXT
    }
}

impl<Pl> DebugObject for ComputePipeline<Pl> {
    #[inline]
    fn object_type() -> vk::DebugReportObjectTypeEXT {
        vk::DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT
    }
}

// impl<Vdef, L, Rp> DebugObject for GraphicsPipeline<Vdef, L, Rp>
//     where L: PipelineLayout,
//           Rp: RenderPass + RenderPassDesc {
//     #[inline]
//     fn object_type() -> vk::DebugReportObjectTypeEXT {
//         vk::DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT
//     }
// }

impl<'a> DebugObject for FramebufferSys<'a> {
    #[inline]
    fn object_type() -> vk::DebugReportObjectTypeEXT {
        vk::DEBUG_REPORT_OBJECT_TYPE_FRAMEBUFFER_EXT
    }
}
