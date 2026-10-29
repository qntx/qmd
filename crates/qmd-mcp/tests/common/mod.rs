//! Shared test scaffolding — the gated-reranker store used by both the
//! stdio (`tools.rs`) and HTTP (`http.rs`) lifecycle tests.

#![allow(dead_code, reason = "not every test binary uses every helper")]

use std::sync::{Arc, Mutex, mpsc};

use qmd::{CollectionSpec, InferenceError, Qmd, Reranker, UpdateOptions};

/// Reranker that signals entry then blocks on a release channel —
/// parks a `query` call inside inference for lifecycle tests.
pub(crate) struct GatedReranker {
    entered: mpsc::SyncSender<()>,
    release: Mutex<mpsc::Receiver<()>>,
}

impl std::fmt::Debug for GatedReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GatedReranker").finish_non_exhaustive()
    }
}

impl Reranker for GatedReranker {
    fn model_id(&self) -> &'static str {
        "gated"
    }

    fn rerank(&self, _query: &str, docs: &[&str]) -> Result<Vec<f32>, InferenceError> {
        // A dropped receiver means the test abandoned the call — skip
        // the signal rather than panic inside the backend.
        let _entered_sent = self.entered.try_send(());
        // A dropped sender unblocks the park the same way an explicit
        // release does; a poisoned lock means the test already failed.
        if let Ok(rx) = self.release.lock() {
            let _released = rx.recv();
        }
        Ok(docs.iter().map(|_| 1.0).collect())
    }
}

/// Store with one indexed doc and a gated reranker.
pub(crate) fn gated_store() -> (
    tempfile::TempDir,
    Arc<Qmd>,
    mpsc::Receiver<()>,
    mpsc::SyncSender<()>,
) {
    let (entered_tx, entered_rx) = mpsc::sync_channel(4);
    let (release_tx, release_rx) = mpsc::sync_channel(4);
    let gate = GatedReranker {
        entered: entered_tx,
        release: Mutex::new(release_rx),
    };
    let dir = tempfile::tempdir().unwrap();
    let corpus = dir.path().join("docs");
    std::fs::create_dir_all(&corpus).unwrap();
    std::fs::write(corpus.join("a.md"), "# A\n\nindex search content\n").unwrap();
    let qmd = Qmd::builder(dir.path().join("index.sqlite"))
        .reranker(Arc::new(gate))
        .build()
        .unwrap();
    qmd.add_collection(
        "docs",
        &CollectionSpec {
            path: corpus.to_string_lossy().into_owned(),
            ..CollectionSpec::default()
        },
    )
    .unwrap();
    qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap();
    (dir, Arc::new(qmd), entered_rx, release_tx)
}
