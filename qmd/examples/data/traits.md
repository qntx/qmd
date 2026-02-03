# Traits in Rust

Traits define shared behavior, similar to interfaces in other languages.

## Defining Traits

```rust
trait Summary {
    fn summarize(&self) -> String;
    
    // Default implementation
    fn preview(&self) -> String {
        format!("Read more: {}", self.summarize())
    }
}
```

## Implementing Traits

```rust
struct Article {
    title: String,
    content: String,
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("{}: {}", self.title, &self.content[..50])
    }
}
```

## Trait Bounds

Constrain generic types:

```rust
fn notify<T: Summary>(item: &T) {
    println!("Breaking: {}", item.summarize());
}

// Multiple bounds
fn process<T: Summary + Display>(item: &T) { ... }

// Where clause for complex bounds
fn complex<T, U>(t: &T, u: &U) -> String
where
    T: Summary + Clone,
    U: Display + Debug,
{ ... }
```

## Common Standard Traits

- `Clone` - explicit duplication
- `Copy` - implicit bitwise copy
- `Debug` - debug formatting `{:?}`
- `Display` - user-facing formatting `{}`
- `Default` - default values
- `PartialEq`, `Eq` - equality comparison
- `PartialOrd`, `Ord` - ordering
- `Hash` - hashing for HashMap keys
