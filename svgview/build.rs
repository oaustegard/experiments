fn main() {
    // Embeds the icon + version block on Windows; silently does nothing
    // everywhere else, so the same build.rs works for the Mac/Linux branch.
    embed_resource::compile("assets/svgview.rc", embed_resource::NONE)
        .manifest_optional()
        .unwrap();
}
