fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let proto_root = manifest_dir.join("proto");
    let google_include = manifest_dir.join("proto");

    tonic_build::configure()
        .build_server(false)
        .compile_protos(
            &[proto_root.join("db_data_2025-10.proto")],
            &[&proto_root, &google_include],
        )?;

    Ok(())
}
