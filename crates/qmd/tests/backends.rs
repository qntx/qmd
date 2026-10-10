//! Inference backend wiring tests (P2.1): `QmdBuilder::embedder`/
//! `reranker`/`generator` injection, `Error::NoBackend` for missing
//! backends, `CancelToken`, `Error` variants, and the `testing`-feature
//! fakes' determinism (T10/T12).

#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::shadow_unrelated,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_assert_message,
    clippy::panic,
    clippy::float_cmp,
    reason = "idiomatic test-code patterns"
)]

use std::sync::Arc;

use qmd::{CancelToken, Capability, Error, Qmd, QmdBuilder};

fn builder() -> (tempfile::TempDir, QmdBuilder) {
    let dir = tempfile::tempdir().unwrap();
    let builder = Qmd::builder(dir.path().join("i.sqlite"));
    (dir, builder)
}

#[test]
fn missing_backends_report_no_backend() {
    let (_dir, builder) = builder();
    let qmd = builder.build().unwrap();
    for (res, cap) in [
        (qmd.embedder().map(|_| ()), Capability::Embed),
        (qmd.reranker().map(|_| ()), Capability::Rerank),
        (qmd.generator().map(|_| ()), Capability::Generate),
    ] {
        match res {
            Err(Error::NoBackend { capability }) => assert_eq!(capability, cap),
            other => panic!("expected NoBackend, got {other:?}"),
        }
    }
    // Display names the capability.
    assert_eq!(
        Error::NoBackend {
            capability: Capability::Embed
        }
        .to_string(),
        "no embedding backend configured"
    );
}

#[cfg(feature = "testing")]
mod fakes {
    use qmd::{Embedder, FakeEmbedder, FakeGenerator, FakeReranker, Generator, Reranker};

    use super::*;

    #[test]
    fn injected_fakes_are_returned_by_accessors() {
        let (_dir, builder) = builder();
        let qmd = builder
            .embedder(Arc::new(FakeEmbedder::new("embed-model")))
            .reranker(Arc::new(FakeReranker::new("rerank-model")))
            .generator(Arc::new(FakeGenerator::new("gen-model")))
            .build()
            .unwrap();
        assert_eq!(qmd.embedder().unwrap().model_id(), "embed-model");
        assert_eq!(qmd.reranker().unwrap().model_id(), "rerank-model");
        assert_eq!(qmd.generator().unwrap().model_id(), "gen-model");
    }

    #[test]
    fn fake_embedder_is_deterministic() {
        let e = FakeEmbedder::new("m").with_dimension(32);
        let a = e.embed(&["hello world", "other"]).unwrap();
        let b = e.embed(&["hello world", "other"]).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].len(), 32);
        // Different texts → different vectors.
        assert_ne!(a[0], a[1]);
        // Normalized to unit length for non-empty input.
        let norm = a[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fake_embedder_tokenize_detokenize_defaults_and_overrides() {
        let e = FakeEmbedder::new("m");
        assert_eq!(e.tokenize("abc").unwrap().len(), 3);
        assert_eq!(e.detokenize(&[1, 2, 3]).unwrap(), "xxx");

        let custom = FakeEmbedder::new("m").with_tokenizer(|_| vec![42]);
        assert_eq!(custom.tokenize("anything").unwrap(), vec![42]);
    }

    #[test]
    fn fake_reranker_scores_word_overlap() {
        let r = FakeReranker::new("m");
        let scores = r
            .rerank(
                "rust search engine",
                &["a rust search engine implementation", "unrelated text"],
            )
            .unwrap();
        assert_eq!(scores.len(), 2);
        assert_eq!(scores[0], 1.0);
        assert_eq!(scores[1], 0.0);
    }

    #[test]
    fn fake_generator_default_and_fixed_output() {
        let g = FakeGenerator::new("m");
        let out = g.expand("test query").unwrap();
        assert!(out.contains("lex: test query"));
        assert!(out.contains("vec: test query"));

        let fixed = FakeGenerator::fixed("m", "lex: only\nvec: only");
        assert_eq!(fixed.expand("ignored").unwrap(), "lex: only\nvec: only");

        let f = FakeGenerator::with_output("m", |q| format!("vec: {q}!"));
        assert_eq!(f.expand("q").unwrap(), "vec: q!");
    }

    #[test]
    fn fingerprint_extra_flows_from_injected_embedder() {
        // A backend reporting `fingerprint_extra` changes the fingerprint
        // used by needs_embedding bookkeeping (D14 groundwork).
        let (_dir, builder) = builder();
        let qmd = builder
            .embedder(Arc::new(
                FakeEmbedder::new("m").with_fingerprint_extra("pooling:full"),
            ))
            .build()
            .unwrap();
        assert_eq!(
            qmd.embedder().unwrap().fingerprint_extra(),
            Some("pooling:full")
        );
    }
}

#[test]
fn cancel_token_propagates() {
    let token = CancelToken::new();
    assert!(!token.is_cancelled());
    let clone = token.clone();
    token.cancel();
    assert!(clone.is_cancelled());
}

#[test]
fn error_variants_display() {
    let e: Error = qmd::InferenceError::Backend("boom".to_owned()).into();
    assert_eq!(e.to_string(), "inference error: boom");
    assert_eq!(Error::Cancelled.to_string(), "operation cancelled");
    assert_eq!(
        Error::EmbeddingDimensionMismatch {
            existing: 768,
            current: 384
        }
        .to_string(),
        "embedding dimension mismatch: stored vectors have 768 dims, embedder produced 384"
    );
    assert_eq!(
        Error::Model {
            uri: "hf:x/y".to_owned(),
            source: qmd::InferenceError::Backend("bad gguf".to_owned()),
        }
        .to_string(),
        "model hf:x/y: bad gguf"
    );
    // InvalidQuery surfaces the upstream reason verbatim.
    assert_eq!(
        Error::InvalidQuery {
            reason: "upstream says no".to_owned()
        }
        .to_string(),
        "upstream says no"
    );
}
