# first steps
xcode-select --install
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

rustc --version

# vscode  , cursor
install rust-analyzer
material icon theme (icons for different file types)

## **rustup**

- **What it is:** Rust’s official installer and toolchain manager. It sets up `rustc`, `cargo`, and friends, and lets you keep multiple Rust versions side-by-side.
    
- **Why it’s useful:**
    
    - Install and switch between **stable**, **beta**, and **nightly** easily.
        
    - Add/remove **targets** for cross-compiling (e.g., `aarch64-apple-darwin`, `wasm32-unknown-unknown`).
        
    - Manage **components** like `rustfmt` and `clippy`.
        
    - Supports per-project **overrides** so one repo can use nightly while everything else stays on stable.
        
- **How it works (in practice):**
    
    - Installs to a user directory (no root required); keeps toolchains isolated.
        
    - Uses **profiles** (`minimal`, `default`, `complete`) to control what gets installed.
        
    - Self-updates and updates toolchains atomically.
        

**Common commands**

- Install a channel: `rustup toolchain install stable` (or `beta` / `nightly`)
    
- Make one default: `rustup default stable`
    
- Update everything: `rustup update`
    
- Add a target: `rustup target add wasm32-unknown-unknown`
    
- Add components: `rustup component add clippy rustfmt`
    
- Per-directory toolchain: `rustup override set nightly` (use `unset` to remove)
    
- See what you have: `rustup show`
    
- Read local docs: `rustup doc`
    
- Uninstall: `rustup self uninstall`
    

**Good to know**

- Env vars: `RUSTUP_HOME` (rustup data) and `CARGO_HOME` (cargo registry/bin).
    
- If you installed Rust via your OS package manager, it’s usually better to remove that and use rustup for consistency and easy upgrades.
    
# Cargo
Here’s a crisp cheat-sheet for **Cargo** (Rust’s package manager + build system) with the _top things you’ll actually use_.

# What Cargo does

- Manages deps via **`Cargo.toml`** and lockfile (**`Cargo.lock`**).
    
- Builds, runs, tests, benches, docs—consistently across platforms.
    
- Publishes crates to **crates.io**, supports **workspaces**, **features**, **profiles**, and **build scripts**.
    

# Everyday workflow (top usage)

- **Create a project**
    
    ```bash
    cargo new myapp          # binary (src/main.rs)
    cargo new mylib --lib    # library (src/lib.rs)
    ```
    
- **Build / run / test**
    
    ```bash
    cargo build              # debug build (target/debug)
    cargo run                # build + run main
    cargo test               # run unit/integration tests
    cargo build --release    # optimized build (target/release)
    cargo check              # fast type-check (no codegen)
    ```
    
- **Add / update dependencies**
    
    ```bash
    cargo add serde serde_json@^1.0    # add deps (requires cargo-edit; rustup component often bundled)
    cargo rm serde_json                # remove dep
    cargo update                       # update to latest allowed by Cargo.toml
    cargo upgrade                       # bump Cargo.toml versions (cargo-edit)
    ```
    
- **Quality & tooling**
    
    ```bash
    cargo fmt        # rustfmt
    cargo clippy     # lints (add with `rustup component add clippy`)
    cargo doc --open # build & open docs
    ```
    
- **Run specific targets**
    
    ```bash
    cargo run --bin mytool
    cargo test my_module::tests::case_name
    cargo test -- --ignored --nocapture
    ```
    

# Workspaces (multi-crate repos)

- Root `Cargo.toml`:
    
    ```toml
    [workspace]
    members = ["crates/core", "crates/cli"]
    ```
    
- Build/test everything:
    
    ```bash
    cargo build        # from the workspace root
    cargo test -p core # package-specific
    ```
    

# Features (optional deps / cfg)

- In `Cargo.toml`:
    
    ```toml
    [features]
    default = ["json"]
    json = ["serde", "serde_json"]
    cli  = []
    ```
    
- Use:
    
    ```bash
    cargo build --no-default-features --features "cli"
    ```
    

# Profiles (tune builds)

- In `Cargo.toml`:
    
    ```toml
    [profile.release]
    lto = "thin"
    codegen-units = 1
    panic = "abort"
    opt-level = "z" # or 3
    ```
    

# Cross-compiling (with rustup targets)

```bash
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown
```

# Publishing

```bash
cargo login <token>         # once
cargo package               # verify package contents
cargo publish               # ship to crates.io
```

# Useful flags & tricks

- **Faster local dev:** `CARGO_PROFILE_DEV_DEBUG=0 cargo build` (or tweak `[profile.dev]`).
    
- **Repro builds:** commit `Cargo.lock` for binaries; libs usually exclude it.
    
- **Offline:** `cargo build --offline` (after a prior successful fetch).
    
- **Env/config:** per-project `.cargo/config.toml` for target, runner, aliases:
    
    ```toml
    [alias]
    t = "test -- --nocapture"
    r = "run --release"
    ```
    
# rustc
compiler
> rustc src/main.rs
> ls
Cargo.lock Cargo.toml main       src        target
> ./main
Hello world
# 