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
                   })
            },
            None => {
                // Pool is empty, alloc new semaphore
                Semaphore::alloc_impl(device, true, None)
            },
        }
    }

    /// Builds a new semaphore.
    #[inline]
    pub fn alloc(device: D) -> Result<Semaphore<D>, OomError> {
        Semaphore::alloc_impl(device, false, None)
    }

    /// Builds a new semaphore that can be exported to native handles.
    #[inline]
    pub fn exportable(device: D, handle_types: &SemaphoreHandleTypes)
                      -> Result<Semaphore<D>, ExternalSemaphoreError> {
        if !device.instance().loaded_extensions().khr_get_physical_device_properties2 {
            return Err(ExternalSemaphoreError::GetPhysicalDeviceProperties2NotEnabled);
        }
        if !device.instance().loaded_extensions().khr_external_semaphore_capabilities {
            return Err(ExternalSemaphoreError::ExternalSemaphoreCapabilitiesNotEnabled);
        }
        if !device.loaded_extensions().khr_external_semaphore {
            return Err(ExternalSemaphoreError::ExternalSemaphoreNotEnabled);
        }
        unimplemented!()
    }

    fn alloc_impl(device: D, must_put_in_pool: bool, handle_types: Option<&SemaphoreHandleTypes>)
                  -> Result<Semaphore<D>, OomError> {
        let semaphore = unsafe {
            // since the creation is constant, we use a `static` instead of a struct on the stack
            static mut INFOS: vk::SemaphoreCreateInfo = vk::SemaphoreCreateInfo {
                sType: vk::STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                pNext: 0 as *const _, // ptr::null()
                flags: 0, // reserved
            };

            let vk = device.pointers();
            let mut output = mem::uninitialized();
            check_errors(vk.CreateSemaphore(device.internal_object(),
                                            &INFOS,
                                            ptr::null(),
                                            &mut output))?;
            output
        };

        Ok(Semaphore {
               device: device,
               semaphore: semaphore,
               must_put_in_pool: must_put_in_pool,
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
#[derive(Clone, Debug, Default)]
pub struct SemaphoreHandleTypes {
    pub opaque_fd: bool,
    pub opaque_win32: bool,
    pub opaque_win32_kmt: bool,
    pub d3d12_fence: bool,
    pub sync_fd: bool,
}

impl SemaphoreHandleTypes {
    fn to_vk_flags(&self) -> vk::ExternalSemaphoreHandleTypeFlagsKHR {
        let mut flags = 0u32;
        if self.opaque_fd { flags |= vk::EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR }
        if self.opaque_win32 { flags |= vk::EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR }
        if self.opaque_win32_kmt { flags |= vk::EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR }
        if self.d3d12_fence { flags |= vk::EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT_KHR }
        if self.sync_fd { flags |= vk::EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT_KHR }
        flags
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

    /// Requested handle types not supported by the implementation.
    HandleTypesNotSupported,
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
            ExternalSemaphoreError::HandleTypesNotSupported =>
                "requested handle types not supported by the implementation",
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
