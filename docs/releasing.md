# Releases

The workflows in `.github/workflows/python.yml` and
`.github/workflows/rust.yml` are adapted from LoPhat. Both run on every push,
pull request, manual workflow dispatch, and published GitHub release. Ordinary
CI runs do not publish packages.

## Checks and wheel coverage

The Rust workflow runs `cargo test --locked --no-default-features`, including
doctests, without requiring Python. The Python workflow discovers stable CPython
versions satisfying `pyproject.toml`'s `requires-python` using uv's interpreter
catalogue, selecting the latest compatible patch of each minor. It excludes
prereleases and the experimental Python 3.13 free-threaded variant; supported
free-threaded variants from Python 3.14 onward are included where available.

Wheels are built and tested on native runners:

| Platform | Architecture | Runner |
| --- | --- | --- |
| Linux, manylinux2014 | x64 | `ubuntu-24.04` |
| Linux, manylinux2014 | ARM64 | `ubuntu-24.04-arm` |
| Windows | x64 | `windows-latest` |
| Windows | ARM64 | `windows-11-arm` |
| macOS | ARM64 | `macos-latest` |

Unavailable version/platform combinations are skipped; for example, uv has no
native Windows ARM64 download for Python 3.10. An entirely missing platform fails
matrix selection. Matrix regression tests use a synthetic catalogue and require
no network access.

Each build checks generated type stubs and the typing marker, installs its wheel
with `--no-index --only-binary :all: --no-deps`, and runs pytest against the
installed extension. Test dependencies are installed from `uv.lock` without
installing the project first. This avoids testing a development build by mistake.

This follows LoPhat's native CPython coverage. It does not produce a PyPI source
distribution, Intel macOS, PyPy, musllinux, 32-bit, or other cross-compiled wheels
from Phimaker's former workflow. Those require additional build and test jobs if
they are to be restored.

## Publishing credentials

Configure these repository Actions secrets:

- `PYPI_API_TOKEN`: a PyPI token with permission to publish `phimaker`.
- `CARGO_REGISTRY_TOKEN`: a crates.io token with permission to publish `phimaker`.

The workflows use these secrets only in their release publishing jobs. They
require no extra repository write permission. A crates.io publisher must also
have access to the crate name, either as an existing owner or for its first
publication.

## Release procedure

1. Match the version in `Cargo.toml`, `Cargo.lock`, and `pyproject.toml`, and move
   the changelog's Unreleased entries into the version being released.
2. Commit and push the release changes and review both CI workflows. A manual
   workflow dispatch can run the same checks without publishing.
3. Create and publish a GitHub release for that tested commit, conventionally
   tagged `v<version>`. Draft releases do not publish packages.
4. Check both release workflow runs. The Python workflow uploads each tested
   wheel and publishes the collected wheels only after the complete matrix
   passes. The Rust workflow publishes only after its Cargo tests pass. The two
   publication workflows run independently.

Before publishing, a local crate dry run can check packaging and compilation:

```sh
cargo publish --dry-run --locked --registry crates-io --no-default-features
```

A dry run does not verify publishing credentials or crate ownership. Native
runner and manylinux compatibility checks must complete in GitHub Actions; a
local wheel test validates only its own interpreter and platform.
