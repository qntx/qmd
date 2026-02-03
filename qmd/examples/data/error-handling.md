# Error Handling in Rust

Rust handles errors through the `Result` and `Option` types, avoiding exceptions.

## The Result Type

`Result<T, E>` represents success or failure:

```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}

fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("Division by zero".to_string())
    } else {
        Ok(a / b)
    }
}
```

## The ? Operator

Propagate errors concisely:

```rust
fn read_config() -> Result<Config, Error> {
    let content = std::fs::read_to_string("config.toml")?;
    let config: Config = toml::from_str(&content)?;
    Ok(config)
}
```

## Option Type

For values that may or may not exist:

```rust
fn find_user(id: u32) -> Option<User> {
    users.get(&id).cloned()
}

// Using Option
match find_user(42) {
    Some(user) => println!("Found: {}", user.name),
    None => println!("User not found"),
}
```

## Best Practices

- Use `anyhow` crate for applications
- Use `thiserror` crate for libraries
- Add context with `.context()` method
- Avoid `unwrap()` in production code
