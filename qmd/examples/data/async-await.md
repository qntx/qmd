# Async/Await in Rust

Rust's async/await enables efficient concurrent programming without data races.

## Async Functions

```rust
async fn fetch_data(url: &str) -> Result<String, Error> {
    let response = reqwest::get(url).await?;
    let body = response.text().await?;
    Ok(body)
}
```

## Executing Async Code

Async functions return a `Future` that must be executed by a runtime:

```rust
#[tokio::main]
async fn main() {
    let data = fetch_data("https://api.example.com").await;
    println!("{:?}", data);
}
```

## Concurrent Execution

Run multiple futures concurrently:

```rust
use tokio::join;

async fn fetch_all() {
    let (a, b, c) = join!(
        fetch_data("url1"),
        fetch_data("url2"),
        fetch_data("url3"),
    );
}
```

## Streams

For async iteration:

```rust
use tokio_stream::StreamExt;

async fn process_stream() {
    let mut stream = some_async_stream();
    while let Some(item) = stream.next().await {
        process(item);
    }
}
```

## Common Runtimes

- **tokio**: Full-featured async runtime
- **async-std**: std library-like async runtime
- **smol**: Lightweight async runtime
