# Rust Programming Basics

Rust is a systems programming language focused on safety, speed, and concurrency.

## Variables and Mutability

By default, variables are immutable in Rust:

```rust
let x = 5;      // immutable
let mut y = 5;  // mutable
y = 6;          // OK
```

## Data Types

Rust is statically typed with type inference:

```rust
let integer: i32 = 42;
let float: f64 = 3.14;
let boolean: bool = true;
let character: char = 'R';
let tuple: (i32, f64) = (500, 6.4);
let array: [i32; 5] = [1, 2, 3, 4, 5];
```

## Functions

Functions use `fn` keyword with explicit return types:

```rust
fn add(a: i32, b: i32) -> i32 {
    a + b  // implicit return (no semicolon)
}

fn greet(name: &str) {
    println!("Hello, {}!", name);
}
```

## Control Flow

```rust
// if expression
let number = if condition { 5 } else { 6 };

// loop with break value
let result = loop {
    counter += 1;
    if counter == 10 {
        break counter * 2;
    }
};

// for loop
for element in array.iter() {
    println!("{}", element);
}
```
