//! Scoring — upstream `src/bench/score.ts`, ported verbatim.
//! `normalizePath`, `pathsMatch`, `scoreResults` byte-for-byte.

/// `normalizePath` (score.ts:10-18) — strip `qmd://` scheme and the
/// first path segment (the collection), lowercase, strip edge slashes.
pub(crate) fn normalize_path(p: &str) -> String {
    let mut p = p.to_owned();
    if let Some(rest) = p.strip_prefix("qmd://") {
        p = rest
            .split_once('/')
            .map_or_else(|| rest.to_owned(), |(_, tail)| tail.to_owned());
    }
    p.to_lowercase().trim_matches('/').to_owned()
}

/// `pathsMatch` (score.ts:24-30) — equal after normalize, or either is
/// a suffix of the other.
pub(crate) fn paths_match(result: &str, expected: &str) -> bool {
    let nr = normalize_path(result);
    let ne = normalize_path(expected);
    nr == ne || nr.ends_with(&ne) || ne.ends_with(&nr)
}

/// Score metrics — upstream `ScoreMetrics` field order.
#[derive(Debug, Clone)]
pub(crate) struct ScoreMetrics {
    pub(crate) precision_at_k: f64,
    pub(crate) recall: f64,
    pub(crate) recall_at_1: f64,
    pub(crate) recall_at_3: f64,
    pub(crate) recall_at_5: f64,
    pub(crate) mrr: f64,
    pub(crate) f1: f64,
    pub(crate) hits_at_k: usize,
    pub(crate) matched_files: Vec<String>,
    pub(crate) unmatched_expected_files: Vec<String>,
}

fn hits_within(result_files: &[String], expected_files: &[String], k: usize) -> usize {
    expected_files
        .iter()
        .filter(|e| result_files.iter().take(k).any(|r| paths_match(r, e)))
        .count()
}

/// `scoreResults` (score.ts:56-111) — JS float arithmetic order
/// preserved for byte-identical `toFixed` output.
pub(crate) fn score_results(
    result_files: &[String],
    expected_files: &[String],
    top_k: usize,
) -> ScoreMetrics {
    let hits_at_k = hits_within(result_files, expected_files, top_k);

    let mut matched_files = Vec::new();
    let mut unmatched_expected_files = Vec::new();
    for expected in expected_files {
        if result_files.iter().any(|r| paths_match(r, expected)) {
            matched_files.push(expected.clone());
        } else {
            unmatched_expected_files.push(expected.clone());
        }
    }

    let mut mrr = 0.0f64;
    for (i, r) in result_files.iter().enumerate() {
        if expected_files.iter().any(|e| paths_match(r, e)) {
            mrr = 1.0 / (i + 1) as f64;
            break;
        }
    }

    let denominator = top_k.min(expected_files.len());
    let precision_at_k = if denominator > 0 {
        hits_at_k as f64 / denominator as f64
    } else {
        0.0
    };
    let n = expected_files.len();
    let recall = if n > 0 {
        matched_files.len() as f64 / n as f64
    } else {
        0.0
    };
    let recall_at_1 = if n > 0 {
        hits_within(result_files, expected_files, 1) as f64 / n as f64
    } else {
        0.0
    };
    let recall_at_3 = if n > 0 {
        hits_within(result_files, expected_files, 3) as f64 / n as f64
    } else {
        0.0
    };
    let recall_at_5 = if n > 0 {
        hits_within(result_files, expected_files, 5) as f64 / n as f64
    } else {
        0.0
    };
    let f1 = if precision_at_k + recall > 0.0 {
        2.0 * (precision_at_k * recall) / (precision_at_k + recall)
    } else {
        0.0
    };

    ScoreMetrics {
        precision_at_k,
        recall,
        recall_at_1,
        recall_at_3,
        recall_at_5,
        mrr,
        f1,
        hits_at_k,
        matched_files,
        unmatched_expected_files,
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::missing_assert_message,
    reason = "test-only module in a feature-gated crate + idiomatic test patterns"
)]
mod tests {
    //! Ports upstream `test/bench-score.test.ts` verbatim plus the
    //! `score` section of `tests/fixtures/bench_golden.json`.
    use super::*;

    fn sv(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn normalize_path_cases() {
        // bench-score.test.ts normalizePath describe (:9-26).
        assert_eq!(
            normalize_path("Resources/Concepts/Context Engineering.md"),
            "resources/concepts/context engineering.md"
        );
        assert_eq!(
            normalize_path("qmd://collection/docs/readme.md"),
            "docs/readme.md"
        );
        assert_eq!(normalize_path("/docs/readme.md/"), "docs/readme.md");
        assert_eq!(normalize_path("readme.md"), "readme.md");
    }

    #[test]
    fn paths_match_cases() {
        // bench-score.test.ts pathsMatch describe (:29-51).
        assert!(paths_match("docs/readme.md", "docs/readme.md"));
        assert!(paths_match("Docs/README.md", "docs/readme.md"));
        assert!(paths_match("/full/path/docs/readme.md", "docs/readme.md"));
        assert!(paths_match("readme.md", "docs/readme.md"));
        assert!(paths_match("qmd://col/docs/readme.md", "docs/readme.md"));
        assert!(!paths_match("docs/readme.md", "docs/other.md"));
    }

    #[test]
    fn score_perfect_all_in_topk() {
        let r = score_results(&sv(&["docs/a.md", "docs/b.md"]), &sv(&["a.md", "b.md"]), 2);
        assert_eq!(r.precision_at_k, 1.0);
        assert_eq!(r.recall, 1.0);
        assert_eq!(r.mrr, 1.0);
        assert_eq!(r.f1, 1.0);
        assert_eq!(r.hits_at_k, 2);
    }

    #[test]
    fn score_zero_none_found() {
        let r = score_results(&sv(&["docs/x.md", "docs/y.md"]), &sv(&["a.md", "b.md"]), 2);
        assert_eq!(r.precision_at_k, 0.0);
        assert_eq!(r.recall, 0.0);
        assert_eq!(r.mrr, 0.0);
        assert_eq!(r.f1, 0.0);
        assert_eq!(r.hits_at_k, 0);
    }

    #[test]
    fn score_found_outside_topk() {
        let r = score_results(&sv(&["x.md", "y.md", "a.md"]), &sv(&["a.md"]), 1);
        assert_eq!(r.precision_at_k, 0.0);
        assert_eq!(r.recall, 1.0);
        assert!((r.mrr - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(r.hits_at_k, 0);
    }

    #[test]
    fn score_mrr_rank2() {
        let r = score_results(&sv(&["x.md", "a.md"]), &sv(&["a.md"]), 2);
        assert!((r.mrr - 0.5).abs() < 1e-9);
    }

    #[test]
    fn score_recall_at_n_and_matched() {
        let r = score_results(
            &sv(&["x.md", "concepts/a.md", "y.md", "b.md", "z.md"]),
            &sv(&["concepts/a.md", "b.md"]),
            5,
        );
        assert_eq!(r.recall_at_1, 0.0);
        assert_eq!(r.recall_at_3, 0.5);
        assert_eq!(r.recall_at_5, 1.0);
        assert_eq!(r.mrr, 0.5);
        assert_eq!(r.matched_files, sv(&["concepts/a.md", "b.md"]));
    }

    /// Golden replay — every `score.results` case in the fixture must
    /// reproduce upstream's `scoreResults` output verbatim.
    #[test]
    fn golden_score_table() {
        #[derive(serde::Deserialize)]
        struct C {
            results: Vec<String>,
            expected: Vec<String>,
            #[serde(rename = "topK")]
            top_k: usize,
            out: Out,
        }
        #[derive(serde::Deserialize)]
        struct Out {
            precision_at_k: f64,
            recall: f64,
            recall_at_1: f64,
            recall_at_3: f64,
            recall_at_5: f64,
            mrr: f64,
            f1: f64,
            hits_at_k: usize,
            matched_files: Vec<String>,
            unmatched_expected_files: Vec<String>,
        }
        #[derive(serde::Deserialize)]
        struct F {
            score: S,
        }
        #[derive(serde::Deserialize)]
        struct S {
            results: Vec<C>,
        }
        let f: F = serde_json::from_str(include_str!("../../tests/fixtures/bench_golden.json"))
            .expect("fixture parses");
        for c in f.score.results {
            let got = score_results(&c.results, &c.expected, c.top_k);
            assert_eq!(
                got.precision_at_k, c.out.precision_at_k,
                "case {:?}",
                c.results
            );
            assert_eq!(got.recall, c.out.recall);
            assert_eq!(got.recall_at_1, c.out.recall_at_1);
            assert_eq!(got.recall_at_3, c.out.recall_at_3);
            assert_eq!(got.recall_at_5, c.out.recall_at_5);
            assert_eq!(got.mrr, c.out.mrr);
            assert_eq!(got.f1, c.out.f1);
            assert_eq!(got.hits_at_k, c.out.hits_at_k);
            assert_eq!(got.matched_files, c.out.matched_files);
            assert_eq!(got.unmatched_expected_files, c.out.unmatched_expected_files);
        }
    }
}
