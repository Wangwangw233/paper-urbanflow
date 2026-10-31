# Legacy archive

These files are retained to document historical implementations. They are not
part of the supported execution path and the verifier skips their local
imports.

- `figures/` contains superseded figure sketches and manuscript-editing
  helpers.
- `matched_capacity/` contains the original construction modules. They import
  private external libraries (`bench_lib`, `de_lib`) that are not bundled.
- `spectra/` contains a historical joint-spectrum audit that depends on
  external transfer-round caches.
- `tools/` contains superseded validation helpers.

No archived file changes a frozen result or the reference manifest.
