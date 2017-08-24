// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::mem;
use std::ptr;
use std::sync::Arc;

use Error;
use OomError;
use SafeDeref;
use VulkanObject;
use check_errors;
use device::Device;
use device::DeviceOwned;
use vk;

/// Used to provide synchronization between command buffers during their execution.
///
/// It is similar to a fence, except that it is purely on the GPU side. The CPU can't query a
/// semaphore's status or wait for it to be signaled.
#[derive(Debug)]
pub struct Semaphore<D = Arc<Device>>
    where D: SafeDeref<Target = Device>
{
    semaphore: vk::Semaphore,
    device: D,
    must_put_in_pool: bool,
    exportable_to: Vec<ExternalSemaphoreHandleType>,
}

impl<D> Semaphore<D>
    where D: SafeDeref<Target = Device>
{
    /// Takes a semaphore from the vulkano-provided semaphore pool.
    /// If the pool is empty, a new semaphore will be allocated.
    /// Upon `drop`, the semaphore is put back into the pool.
    ///
    /// For most applications, using the pool should be preferred,
    /// in order to avoid creating new semaphores every frame.
    pub fn from_pool(device: D) -> Result<Semaphore<D>, OomError> {
        let maybe_raw_sem = device.semaphore_pool().lock().unwrap().pop();
        match maybe_raw_sem {
            Some(raw_sem) => {
                Ok(Semaphore {
                       device: device,
                       semaphore: raw_sem,
                       must_put_in_pool: true,
                       exportable_to: Vec::new(),
                   })
            },
            None => {
                // Pool is empty, alloc new semaphore
                unsafe { Semaphore::alloc_impl(device, true, None) }
            },
        }
    }

    /// Builds a new semaphore.
    #[inline]
    pub fn alloc(device: D) -> Result<Semaphore<D>, OomError> {
        unsafe { Semaphore::alloc_impl(device, false, None) }
    }

    /// Builds a new semaphore that can be exported to native handles.
    ///
    /// `handle_types` is the list of handle types this semaphore can be exported to.
    #[inline]
    pub fn exportable(device: D, handle_types: &[ExternalSemaphoreHandleType])
                      -> Result<Semaphore<D>, ExternalSemaphoreError> {
        if !device
            .instance()
            .loaded_extensions()
            .khr_get_physical_device_properties2
        {
            return Err(ExternalSemaphoreError::GetPhysicalDeviceProperties2NotEnabled);
        }
        if !device
            .instance()
            .loaded_extensions()
            .khr_external_semaphore_capabilities
        {
            return Err(ExternalSemaphoreError::ExternalSemaphoreCapabilitiesNotEnabled);
        }

        if !device.loaded_extensions().khr_external_semaphore {
            return Err(ExternalSemaphoreError::ExternalSemaphoreNotEnabled);
        }

        // Make sure the given handle types are supported and compatible
        for handle_type in handle_types.iter() {
            let physical_device = device.physical_device();
            let properties = physical_device.external_semaphore_properties().get(handle_type).unwrap();
            if properties.external_semaphore_features == 0 {
                return Err(ExternalSemaphoreError::HandleTypeNotSupported(*handle_type));
            }
            // Check if handle_type is compatible with all other handle types
            for other_handle_type in handle_types.iter() {
                if other_handle_type == handle_type {
                    continue;
                }
                if other_handle_type.to_vk() & properties.compatible_handle_types == 0 {
                    return Err(ExternalSemaphoreError::IncompatibleHandleTypes(*handle_type, *other_handle_type));
                }
            }
        }
        unsafe {
            Semaphore::alloc_impl(device, false, Some(handle_types))
                .map_err(|oom_error| ExternalSemaphoreError::OomError(oom_error))
        }
    }

    // Unsafety: if handle_type is `Some`, the given handle types must be supported and compatible.
    unsafe fn alloc_impl(device: D, must_put_in_pool: bool, export_handle_types: Option<&[ExternalSemaphoreHandleType]>)
                  -> Result<Semaphore<D>, OomError> {
        let export_create_info: Option<vk::ExportSemaphoreCreateInfoKHR> = if let Some(export_handle_types) = export_handle_types {
            debug_assert!(device.loaded_extensions().khr_external_semaphore);
            let mut handle_types = 0u32;
            for handle_type in export_handle_types.iter() {
                handle_types |= handle_type.to_vk();
            }
            Some(vk::ExportSemaphoreCreateInfoKHR {
                sType: vk::STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR,
                pNext: ptr::null_mut(),
                handleTypes: handle_types,
            })
        } else {
            None
        };
        let semaphore = {
            // since the creation is constant, we use a `static` instead of a struct on the stack
            let infos: vk::SemaphoreCreateInfo = vk::SemaphoreCreateInfo {
                sType: vk::STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                pNext: export_create_info.as_ref().map(|export_info| export_info as *const vk::ExportSemaphoreCreateInfoKHR as *const _).unwrap_or(ptr::null()),
                flags: 0, // reserved
            };

            let vk = device.pointers();
            let mut output = mem::uninitialized();
            check_errors(vk.CreateSemaphore(device.internal_object(),
                                            &infos,
                                            ptr::null(),
                                            &mut output))?;
            output
        };

        Ok(Semaphore {
               device: device,
               semaphore: semaphore,
               must_put_in_pool: must_put_in_pool,
               exportable_to: match export_handle_types {
                   Some(handle_types) => handle_types.iter().cloned().collect(),
                   None => Vec::new()
               }
           })
    }
}

unsafe impl DeviceOwned for Semaphore {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl<D> VulkanObject for Semaphore<D>
    where D: SafeDeref<Target = Device>
{
    type Object = vk::Semaphore;

    #[inline]
    fn internal_object(&self) -> vk::Semaphore {
        self.semaphore
    }
}

impl<D> Drop for Semaphore<D>
    where D: SafeDeref<Target = Device>
{
    #[inline]
    fn drop(&mut self) {
        unsafe {
            if self.must_put_in_pool {
                let raw_sem = self.semaphore;
                self.device.semaphore_pool().lock().unwrap().push(raw_sem);
            } else {
                let vk = self.device.pointers();
                vk.DestroySemaphore(self.device.internal_object(), self.semaphore, ptr::null());
            }
        }
    }
}

/// Represents handle types that semaphores can be exported to.
/// TODO: Documentation for each handle type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExternalSemaphoreHandleType {
    OpaqueFd,
    OpaqueWin32,
    OpaqueWin32Kmt,
    D3d12Fence,
    SyncFd,
}

impl ExternalSemaphoreHandleType {
    pub(crate) fn to_vk(&self) -> vk::ExternalSemaphoreHandleTypeFlagsKHR {
        match *self {
            ExternalSemaphoreHandleType::OpaqueFd =>
                vk::EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR,
            ExternalSemaphoreHandleType::OpaqueWin32 =>
                vk::EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR,
            ExternalSemaphoreHandleType::OpaqueWin32Kmt =>
                vk::EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR,
            ExternalSemaphoreHandleType::D3d12Fence =>
                vk::EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT_KHR,
            ExternalSemaphoreHandleType::SyncFd =>
                vk::EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT_KHR,
        }
    }
}

/// Error that can be returned when dealing with external semaphores.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExternalSemaphoreError {
    /// No memory available.
    OomError(OomError),

    /// Instance extension `VK_KHR_get_physical_device_properties2` not enabled.
    GetPhysicalDeviceProperties2NotEnabled,

    /// Device extension `VK_KHR_external_semaphore` not enabled.
    ExternalSemaphoreNotEnabled,

    /// Instance extension `VK_KHR_external_semaphore_capabilities` not enabled.
    ExternalSemaphoreCapabilitiesNotEnabled,

    /// Device extension `VK_KHR_external_semaphore_fd` not enabled.
    ExternalSemaphoreFdNotEnabled,

    /// Device extension `VK_KHR_external_semaphore_win32` not enabled.
    ExternalSemaphoreWin32NotEnabled,

    /// Requested handle type not supported by the implementation.
    HandleTypeNotSupported(ExternalSemaphoreHandleType),

    /// Requested handle types are not compatible.
    IncompatibleHandleTypes(ExternalSemaphoreHandleType, ExternalSemaphoreHandleType),
}

impl error::Error for ExternalSemaphoreError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            ExternalSemaphoreError::OomError(_) => "no memory available",
            ExternalSemaphoreError::GetPhysicalDeviceProperties2NotEnabled =>
                "instance extension `VK_KHR_get_physical_device_properties2` not enabled",
            ExternalSemaphoreError::ExternalSemaphoreNotEnabled =>
                "device extension `VK_KHR_external_semaphore` not enabled",
            ExternalSemaphoreError::ExternalSemaphoreCapabilitiesNotEnabled =>
                "instance extension `VK_KHR_external_semaphore_capabilities` not enabled",
            ExternalSemaphoreError::ExternalSemaphoreFdNotEnabled =>
                "device extension `VK_KHR_external_semaphore_fd` not enabled",
            ExternalSemaphoreError::ExternalSemaphoreWin32NotEnabled =>
                "device extension `VK_KHR_external_semaphore_win32` not enabled",
            ExternalSemaphoreError::HandleTypeNotSupported(_) =>
                "requested handle type not supported by the implementation",
            ExternalSemaphoreError::IncompatibleHandleTypes(_, _) =>
                "requested handle types are not compatible",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            ExternalSemaphoreError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for ExternalSemaphoreError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<Error> for ExternalSemaphoreError {
    #[inline]
    fn from(err: Error) -> ExternalSemaphoreError {
        match err {
            Error::OutOfHostMemory => ExternalSemaphoreError::OomError(From::from(err)),
            Error::OutOfDeviceMemory => ExternalSemaphoreError::OomError(From::from(err)),
            _ => panic!("Unexpected error value: {}", err as i32),
        }
    }
}

#[cfg(test)]
mod tests {
    use VulkanObject;
    use sync::Semaphore;

    #[test]
    fn semaphore_create() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = Semaphore::alloc(device.clone());
    }

    #[test]
    fn semaphore_pool() {
        let (device, _) = gfx_dev_and_queue!();

        assert_eq!(device.semaphore_pool().lock().unwrap().len(), 0);
        let sem1_internal_obj = {
            let sem = Semaphore::from_pool(device.clone()).unwrap();
            assert_eq!(device.semaphore_pool().lock().unwrap().len(), 0);
            sem.internal_object()
        };

        assert_eq!(device.semaphore_pool().lock().unwrap().len(), 1);
        let sem2 = Semaphore::from_pool(device.clone()).unwrap();
        assert_eq!(device.semaphore_pool().lock().unwrap().len(), 0);
        assert_eq!(sem2.internal_object(), sem1_internal_obj);
    }
}
