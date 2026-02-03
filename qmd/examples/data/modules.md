# Modules in Rust

Modules organize code into namespaces and control visibility.

## Module Declaration

```rust
// In lib.rs or main.rs
mod network;      // Loads from network.rs or network/mod.rs
mod database;

// Inline module
mod helpers {
    pub fn utility() { ... }
}
```

## Visibility

Items are private by default:

```rust
mod backend {
    pub struct User {      // Public struct
        pub name: String,  // Public field
        email: String,     // Private field
    }

    pub fn create_user() -> User { ... }  // Public function
    fn validate() { ... }                  // Private function
}
```

## Use Statements

Bring items into scope:

```rust
use std::collections::HashMap;
use std::io::{self, Read, Write};

// Rename imports
use std::fmt::Result as FmtResult;
use std::io::Result as IoResult;

// Re-export
pub use self::helpers::utility;
```

## Module Hierarchy

```text
src/
├── lib.rs          // mod network; mod database;
├── network/
│   ├── mod.rs      // pub mod client; pub mod server;
│   ├── client.rs
│   └── server.rs
└── database.rs
```

## Crate Structure

- `lib.rs` - library crate root
- `main.rs` - binary crate root
- `Cargo.toml` - package manifest
