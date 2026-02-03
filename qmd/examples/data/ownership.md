# Ownership in Rust

Ownership is Rust's most unique feature, enabling memory safety without garbage collection.

## Ownership Rules

1. Each value has exactly one owner
2. When the owner goes out of scope, the value is dropped
3. Ownership can be transferred (moved) or borrowed

## Move Semantics

```rust
let s1 = String::from("hello");
let s2 = s1;  // s1 is moved to s2
// println!("{}", s1);  // Error: s1 is no longer valid
```

## Borrowing

References allow using values without taking ownership:

```rust
fn calculate_length(s: &String) -> usize {
    s.len()  // s is borrowed, not owned
}

let s = String::from("hello");
let len = calculate_length(&s);
println!("{} has length {}", s, len);  // s is still valid
```

## Mutable References

Only one mutable reference at a time:

```rust
fn append_world(s: &mut String) {
    s.push_str(", world!");
}

let mut s = String::from("hello");
append_world(&mut s);
```

## Lifetimes

Explicit lifetime annotations when needed:

```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```
