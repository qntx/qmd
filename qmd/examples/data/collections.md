# Collections in Rust

Rust's standard library provides efficient collection types.

## Vector

Growable array:

```rust
let mut vec: Vec<i32> = Vec::new();
vec.push(1);
vec.push(2);

// Macro shorthand
let vec = vec![1, 2, 3];

// Access elements
let third = &vec[2];
let third = vec.get(2);  // Returns Option<&T>
```

## HashMap

Key-value store:

```rust
use std::collections::HashMap;

let mut scores = HashMap::new();
scores.insert("Blue", 10);
scores.insert("Red", 50);

// Get value
let score = scores.get("Blue");

// Entry API
scores.entry("Yellow").or_insert(25);
```

## HashSet

Unique values:

```rust
use std::collections::HashSet;

let mut set = HashSet::new();
set.insert("apple");
set.insert("banana");

if set.contains("apple") {
    println!("Found apple!");
}
```

## Iterators

Powerful iteration patterns:

```rust
let v = vec![1, 2, 3, 4, 5];

// Map and collect
let doubled: Vec<_> = v.iter().map(|x| x * 2).collect();

// Filter
let evens: Vec<_> = v.iter().filter(|x| *x % 2 == 0).collect();

// Fold/reduce
let sum: i32 = v.iter().fold(0, |acc, x| acc + x);
```
