//! Picking a graphics backend that actually presents frames.
//!
//! ggez defaults to `Backend::All`, which is wgpu's PRIMARY set — Vulkan, Metal and DX12. On
//! Windows that resolves to DX12, which is the right answer on a real machine and the wrong one
//! under Wine: the surface and the device are created without complaint, the first frame renders,
//! and then the present blocks forever. The window stays black and the process never exits.
//!
//! Vulkan works there, so that is what we ask for when we detect we are running under Wine. Real
//! Windows keeps the DX12 path it has always used.

use ggez::conf::Backend;

/// Wine's `ntdll` exports `wine_get_version`; a genuine Windows one does not. This is the check
/// Wine itself documents for programs that want to know, and it covers Proton too.
#[cfg(windows)]
fn running_under_wine() -> bool {
    use std::ffi::{c_char, c_void, CString};

    #[link(name = "kernel32")]
    extern "system" {
        fn GetModuleHandleA(module_name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, proc_name: *const c_char) -> *mut c_void;
    }

    let (Ok(ntdll), Ok(symbol)) = (CString::new("ntdll.dll"), CString::new("wine_get_version"))
    else {
        return false;
    };

    // SAFETY: both pointers are valid NUL-terminated strings that outlive the calls, and we only
    // ever test the results for null. GetModuleHandleA does not transfer ownership.
    unsafe {
        let module = GetModuleHandleA(ntdll.as_ptr());
        !module.is_null() && !GetProcAddress(module, symbol.as_ptr()).is_null()
    }
}

#[cfg(not(windows))]
fn running_under_wine() -> bool {
    false
}

/// The backend to hand to ggez. `GAME_BACKEND` overrides it — `vulkan`, `dx12`, `gl` or `all` —
/// so a player whose machine disagrees with this guess has a way out that does not need a rebuild.
pub(crate) fn preferred_backend() -> Backend {
    match std::env::var("GAME_BACKEND").as_deref() {
        Ok("vulkan") => return Backend::Vulkan,
        Ok("dx12") => return Backend::Dx12,
        Ok("gl") => return Backend::Gl,
        Ok("all") => return Backend::All,
        _ => {}
    }

    if running_under_wine() {
        println!("Running under Wine: using the Vulkan backend, since DX12 hangs on present.");
        Backend::Vulkan
    } else {
        Backend::All
    }
}
