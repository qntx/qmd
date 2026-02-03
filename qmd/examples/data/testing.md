# Testing in Rust

Rust has first-class support for testing built into the language.

## Unit Tests

Test functions with `#[test]` attribute:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 2), 4);
    }

    #[test]
    fn test_divide() {
        assert!(divide(10.0, 2.0).is_ok());
    }

    #[test]
    #[should_panic(expected = "zero")]
    fn test_divide_by_zero() {
        divide(1.0, 0.0).unwrap();
    }
}
```

## Assertions

```rust
assert!(condition);
assert_eq!(left, right);
assert_ne!(left, right);

// With custom message
assert!(result.is_ok(), "Expected Ok, got {:?}", result);
```

## Integration Tests

Place in `tests/` directory:

```rust
// tests/integration_test.rs
use my_crate::public_function;

#[test]
fn test_public_api() {
    let result = public_function();
    assert!(result.is_valid());
}
```

## Running Tests

```bash
cargo test              # Run all tests
cargo test test_name    # Run specific test
cargo test -- --nocapture  # Show println! output
cargo test --doc        # Run doc tests
```

## Test Organization

- Unit tests: in same file as code
- Integration tests: in `tests/` directory
- Doc tests: in documentation comments
