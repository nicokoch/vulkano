
use num_traits::ToPrimitive;
use std::error;
use std::ffi::CString;
use std::fmt;
use std::ptr;

use device::DeviceOwned;

use Error;
use OomError;
use VulkanObject;
use check_errors;
use vk;

/// Add user defined information to vulkan objects. Requires extension `VK_EXT_debug_marker`.
pub trait DebugMarker {
    /// Set the name of the vulkan object.
    fn set_object_name(&mut self, name: &str) -> Result<(), DebugMarkerError>;
    /// Attach arbitrary data to a vulkan object.
    fn set_object_tag(&mut self, tag_name: u64, tag: &[u8]) -> Result<(), DebugMarkerError>;
}

impl<T> DebugMarker for T
    where T: VulkanObject + DeviceOwned
{
    fn set_object_name(&mut self, name: &str) -> Result<(), DebugMarkerError> {
        if !self.device().loaded_extensions().ext_debug_marker {
            return Err(DebugMarkerError::MissingExtension);
        }

        let vk = self.device().pointers();
        let name_ffi = CString::new(name).unwrap();
        let mut name_info = vk::DebugMarkerObjectNameInfoEXT {
            sType: vk::STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_NAME_INFO_EXT,
            pNext: ptr::null(),
            objectType: T::debug_report_object_type(),
            object: self.internal_object().to_u64().unwrap(),
            pObjectName: name_ffi.as_ptr(),
        };

        unsafe {
            check_errors(vk.DebugMarkerSetObjectNameEXT(self.device().internal_object(),
                                                        &mut name_info as *mut _))?;
        }
        Ok(())
    }

    fn set_object_tag(&mut self, tag_name: u64, tag: &[u8]) -> Result<(), DebugMarkerError> {
        if !self.device().loaded_extensions().ext_debug_marker {
            return Err(DebugMarkerError::MissingExtension);
        }
        assert!(tag_name != 0);
        let vk = self.device().pointers();
        let mut tag_info = vk::DebugMarkerObjectTagInfoEXT {
            sType: vk::STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_TAG_INFO_EXT,
            pNext: ptr::null(),
            objectType: T::debug_report_object_type(),
            object: self.internal_object().to_u64().unwrap(),
            tagName: tag_name,
            tagSize: tag.len(),
            pTag: tag.as_ptr() as *const _,
        };

        unsafe {
            check_errors(vk.DebugMarkerSetObjectTagEXT(self.device().internal_object(),
                                                       &mut tag_info as *mut _))?;
        }
        Ok(())
    }
}

/// Error that can happen when using debug markers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DebugMarkerError {
    OomError(OomError),
    /// The `VK_EXT_debug_marker` extension was not enabled.
    MissingExtension,
}

impl error::Error for DebugMarkerError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            DebugMarkerError::OomError(ref err) => "not enough memory available",
            DebugMarkerError::MissingExtension => "the `VK_EXT_debug_marker` extension is not \
                                                   enabled",
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            DebugMarkerError::OomError(ref err) => Some(err),
            _ => None,
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
            e @ Error::OutOfHostMemory |
            e @ Error::OutOfDeviceMemory => DebugMarkerError::OomError(e.into()),
            _ => unreachable!(),
        }
    }
}
