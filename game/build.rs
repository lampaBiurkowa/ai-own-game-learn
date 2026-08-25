#[cfg(windows)]
fn main() {
    println!("cargo:rustc-link-arg-bin=game=/SUBSYSTEM:WINDOWS");
    embed_resource::compile("icon.rc", embed_resource::NONE)
        .manifest_optional()
        .unwrap();
}

#[cfg(not(windows))]
fn main() {}
