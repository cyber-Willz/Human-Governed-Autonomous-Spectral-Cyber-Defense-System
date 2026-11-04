fn main() {
    if let Err(e) = spectral_jacobi_ds::run() {
        eprintln!("pipeline failed: {e}");
        std::process::exit(1);
    }
}
