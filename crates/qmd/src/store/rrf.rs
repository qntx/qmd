//! Reciprocal Rank Fusion — upstream `reciprocalRankFusion`,
//! `buildRrfTrace`, `getHybridRrfWeights` (store.ts:4657-4765,
//! 5521-5523).

use indexmap::IndexMap;

use super::search::SearchSource;
use crate::query::QueryKind;

/// RRF smoothing constant — upstream `k = 60` default.
pub(crate) const RRF_K: usize = 60;

/// One ranked hit inside a fusion input list — upstream `RankedResult`
/// (store.ts:5451-5456) plus `docid`/`context`, which upstream carries
/// through the side-channel `docidMap` and a `getContextForFile` lookup
/// at the end; keeping them on the entry is equivalent.
#[derive(Debug, Clone)]
pub(crate) struct RankedResult {
    /// `qmd://collection/path` URI.
    pub file: String,
    /// `collection/path` for display.
    pub display_path: String,
    /// Document title.
    pub title: String,
    /// Full body.
    pub body: String,
    /// Backend score (bm25-normalized or `1 - distance`).
    pub score: f64,
    /// Short docid (first 6 hash chars).
    pub docid: String,
    /// Folder context.
    pub context: Option<String>,
}

/// Provenance of one fusion input list — upstream `RankedListMeta`
/// (store.ts:5504-5508).
#[derive(Debug, Clone)]
pub(crate) struct RankedListMeta {
    /// `fts` or `vec`.
    pub source: SearchSource,
    /// `original` or the expansion kind that produced the list.
    pub query_type: QueryKind,
    /// The query text that produced the list.
    pub query: String,
}

/// Upstream `getHybridRrfWeights` (store.ts:5521-5523): original-query
/// lists get 2× weight; expansion-derived lists stay at 1×.
pub(crate) fn get_hybrid_rrf_weights(ranked_list_meta: &[RankedListMeta]) -> Vec<f64> {
    ranked_list_meta
        .iter()
        .map(|m| {
            if m.query_type == QueryKind::Original {
                2.0
            } else {
                1.0
            }
        })
        .collect()
}

struct FusionEntry {
    result: RankedResult,
    rrf_score: f64,
    top_rank: usize,
}

/// Upstream `reciprocalRankFusion` (store.ts:4657-4699): per-list
/// `weight / (k + rank + 1)` accumulation with a top-rank bonus
/// (+0.05 rank 0, +0.02 ranks 1-2), then a stable descending sort —
/// `IndexMap` preserves first-seen order for tied scores like the
/// upstream `Map` + stable sort.
pub(crate) fn reciprocal_rank_fusion(
    result_lists: &[Vec<RankedResult>],
    weights: &[f64],
    k: usize,
) -> Vec<RankedResult> {
    let mut scores: IndexMap<String, FusionEntry> = IndexMap::new();
    for (list_idx, list) in result_lists.iter().enumerate() {
        let weight = weights.get(list_idx).copied().unwrap_or(1.0);
        for (rank, result) in list.iter().enumerate() {
            let contribution = weight / (k + rank + 1) as f64;
            match scores.get_mut(&result.file) {
                Some(entry) => {
                    entry.rrf_score += contribution;
                    entry.top_rank = entry.top_rank.min(rank);
                }
                None => {
                    scores.insert(
                        result.file.clone(),
                        FusionEntry {
                            result: result.clone(),
                            rrf_score: contribution,
                            top_rank: rank,
                        },
                    );
                }
            }
        }
    }
    for entry in scores.values_mut() {
        if entry.top_rank == 0 {
            entry.rrf_score += 0.05;
        } else if entry.top_rank <= 2 {
            entry.rrf_score += 0.02;
        }
    }
    let mut fused: Vec<RankedResult> = scores
        .into_values()
        .map(|e| RankedResult {
            score: e.rrf_score,
            ..e.result
        })
        .collect();
    fused.sort_by(|a, b| b.score.total_cmp(&a.score));
    fused
}

/// One list's contribution to a fused score — upstream
/// `RRFContributionTrace`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RrfContribution {
    /// Index of the input list.
    pub list_index: usize,
    /// `fts` or `vec`.
    pub source: SearchSource,
    /// `original`/`lex`/`vec`/`hyde`.
    pub query_type: QueryKind,
    /// The query text that produced the list.
    pub query: String,
    /// 1-indexed rank inside the list.
    pub rank: usize,
    /// List weight.
    pub weight: f64,
    /// The backend's pre-fusion score for this hit.
    pub backend_score: f64,
    /// `weight / (k + rank)`.
    pub contribution: f64,
}

/// Per-file fusion trace — upstream `RRFScoreTrace` (the map value of
/// `buildRrfTrace`).
#[derive(Debug)]
pub(crate) struct FileRrfTrace {
    /// All list contributions.
    pub contributions: Vec<RrfContribution>,
    /// Sum of `contribution` before the bonus.
    pub base_score: f64,
    /// Best 1-indexed rank across lists.
    pub top_rank: usize,
    /// 0.05 (rank 1) / 0.02 (ranks 2-3) / 0.
    pub top_rank_bonus: f64,
    /// `base_score + top_rank_bonus`.
    pub total_score: f64,
}

/// Upstream `buildRrfTrace` (store.ts:4705-4765): same accumulation as
/// fusion but with 1-indexed ranks and per-list contribution records.
pub(crate) fn build_rrf_trace(
    result_lists: &[Vec<RankedResult>],
    weights: &[f64],
    list_meta: &[RankedListMeta],
    k: usize,
) -> IndexMap<String, FileRrfTrace> {
    let mut traces: IndexMap<String, FileRrfTrace> = IndexMap::new();
    for (list_idx, list) in result_lists.iter().enumerate() {
        let weight = weights.get(list_idx).copied().unwrap_or(1.0);
        let meta = list_meta.get(list_idx);
        for (rank0, result) in list.iter().enumerate() {
            let rank = rank0 + 1; // 1-indexed for explain output
            let contribution = weight / (k + rank) as f64;
            let detail = RrfContribution {
                list_index: list_idx,
                source: meta.map_or(SearchSource::Fts, |m| m.source),
                query_type: meta.map_or(QueryKind::Original, |m| m.query_type),
                query: meta.map_or_else(String::new, |m| m.query.clone()),
                rank,
                weight,
                backend_score: result.score,
                contribution,
            };
            match traces.get_mut(&result.file) {
                Some(t) => {
                    t.base_score += contribution;
                    t.top_rank = t.top_rank.min(rank);
                    t.contributions.push(detail);
                }
                None => {
                    traces.insert(
                        result.file.clone(),
                        FileRrfTrace {
                            contributions: vec![detail],
                            base_score: contribution,
                            top_rank: rank,
                            top_rank_bonus: 0.0,
                            total_score: 0.0,
                        },
                    );
                }
            }
        }
    }
    for trace in traces.values_mut() {
        let bonus = if trace.top_rank == 1 {
            0.05
        } else if trace.top_rank <= 3 {
            0.02
        } else {
            0.0
        };
        trace.top_rank_bonus = bonus;
        trace.total_score = trace.base_score + bonus;
    }
    traces
}

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::panic,
    clippy::unwrap_used,
    clippy::missing_assert_message,
    reason = "idiomatic test-code patterns"
)]
mod tests {
    use super::*;

    fn result(file: &str, score: f64) -> RankedResult {
        RankedResult {
            file: file.to_owned(),
            display_path: file.to_owned(),
            title: file.to_owned(),
            body: "body".to_owned(),
            score,
            docid: String::new(),
            context: None,
        }
    }

    fn meta(source: SearchSource, query_type: QueryKind, query: &str) -> RankedListMeta {
        RankedListMeta {
            source,
            query_type,
            query: query.to_owned(),
        }
    }

    // ── rrf-trace.test.ts ──

    #[test]
    fn trace_totals_match_fused_scores() {
        let list1 = vec![
            result("qmd://docs/a.md", 0.92),
            result("qmd://docs/b.md", 0.81),
        ];
        let list2 = vec![
            result("qmd://docs/b.md", 0.77),
            result("qmd://docs/a.md", 0.65),
        ];
        let weights = [2.0, 1.0];
        let metas = [
            meta(SearchSource::Fts, QueryKind::Lex, "lex query"),
            meta(SearchSource::Vec, QueryKind::Vec, "vec query"),
        ];
        let traces = build_rrf_trace(&[list1.clone(), list2.clone()], &weights, &metas, RRF_K);
        let fused = reciprocal_rank_fusion(&[list1, list2], &weights, RRF_K);

        for r in &fused {
            let t = traces.get(&r.file).unwrap();
            assert!(
                (t.total_score - r.score).abs() < 1e-10,
                "{}: {} vs {}",
                r.file,
                t.total_score,
                r.score
            );
        }
        let a = traces.get("qmd://docs/a.md").unwrap();
        assert_eq!(a.contributions.len(), 2);
        assert_eq!(a.contributions[0].source, SearchSource::Fts);
        assert_eq!(a.contributions[1].source, SearchSource::Vec);
        assert_eq!(a.top_rank, 1);
        assert!((a.top_rank_bonus - 0.05).abs() < 1e-10);
    }

    #[test]
    fn trace_top_rank_bonus_thresholds() {
        let list = vec![
            result("qmd://docs/r1.md", 0.9),
            result("qmd://docs/r2.md", 0.8),
            result("qmd://docs/r3.md", 0.7),
            result("qmd://docs/r4.md", 0.6),
        ];
        let metas = [meta(SearchSource::Fts, QueryKind::Lex, "rank")];
        let traces = build_rrf_trace(&[list], &[1.0], &metas, RRF_K);
        assert!((traces["qmd://docs/r1.md"].top_rank_bonus - 0.05).abs() < 1e-10);
        assert!((traces["qmd://docs/r2.md"].top_rank_bonus - 0.02).abs() < 1e-10);
        assert!((traces["qmd://docs/r3.md"].top_rank_bonus - 0.02).abs() < 1e-10);
        assert!((traces["qmd://docs/r4.md"].top_rank_bonus).abs() < 1e-10);
    }

    // ── store.test.ts Reciprocal Rank Fusion ──

    #[test]
    fn single_list_order_preserved() {
        let list = vec![
            result("doc1", 0.9),
            result("doc2", 0.8),
            result("doc3", 0.7),
        ];
        let fused = reciprocal_rank_fusion(&[list], &[], RRF_K);
        assert_eq!(
            fused.iter().map(|r| r.file.as_str()).collect::<Vec<_>>(),
            vec!["doc1", "doc2", "doc3"]
        );
    }

    #[test]
    fn merges_documents_from_multiple_lists() {
        let list1 = vec![result("doc1", 0.9), result("doc2", 0.8)];
        let list2 = vec![result("doc2", 0.95), result("doc3", 0.85)];
        let fused = reciprocal_rank_fusion(&[list1, list2], &[], RRF_K);
        for f in ["doc1", "doc2", "doc3"] {
            assert!(fused.iter().any(|r| r.file == f));
        }
    }

    #[test]
    fn respects_weights() {
        let list1 = vec![result("doc1", 0.9)];
        let list2 = vec![result("doc2", 0.9)];
        let fused = reciprocal_rank_fusion(&[list1, list2], &[2.0, 1.0], RRF_K);
        assert_eq!(fused[0].file, "doc1");
    }

    #[test]
    fn hybrid_weights_boost_original_over_expansion() {
        // Mirrors hybridQuery's common list order when a lex expansion
        // exists: original FTS, lex expansion FTS, original vector.
        let ranked_lists = vec![
            vec![result("original-fts-only.md", 0.95)],
            vec![result("lex-expansion-only.md", 0.95)],
            vec![result("original-vector.md", 0.95)],
        ];
        let metas = [
            meta(SearchSource::Fts, QueryKind::Original, "user query"),
            meta(SearchSource::Fts, QueryKind::Lex, "lex expansion"),
            meta(SearchSource::Vec, QueryKind::Original, "user query"),
        ];
        // Position-based [2,1,1] is the upstream bug being fixed.
        let buggy = reciprocal_rank_fusion(&ranked_lists, &[2.0, 1.0, 1.0], RRF_K);
        let pos_of =
            |f: &[RankedResult], name: &str| f.iter().position(|r| r.file == name).unwrap();
        assert!(pos_of(&buggy, "lex-expansion-only.md") < pos_of(&buggy, "original-vector.md"));

        let weights = get_hybrid_rrf_weights(&metas);
        assert_eq!(weights, vec![2.0, 1.0, 2.0]);
        let fixed = reciprocal_rank_fusion(&ranked_lists, &weights, RRF_K);
        assert!(pos_of(&fixed, "original-vector.md") < pos_of(&fixed, "lex-expansion-only.md"));
    }

    #[test]
    fn top_rank_bonus_applied() {
        let list1 = vec![result("doc1", 0.9), result("doc2", 0.8)];
        let list2 = vec![result("doc3", 0.85)];
        let fused = reciprocal_rank_fusion(&[list1, list2], &[], RRF_K);
        let s = |name: &str| fused.iter().find(|r| r.file == name).unwrap().score;
        assert!(s("doc1") > s("doc2"));
    }

    #[test]
    fn empty_lists() {
        assert!(reciprocal_rank_fusion(&[vec![], vec![]], &[], RRF_K).is_empty());
    }

    #[test]
    fn k_parameter_changes_scores() {
        let list = vec![result("doc1", 0.9)];
        let fused60 = reciprocal_rank_fusion(std::slice::from_ref(&list), &[], 60);
        let fused30 = reciprocal_rank_fusion(&[list], &[], 30);
        assert!(fused30[0].score > fused60[0].score);
    }

    #[test]
    fn trace_default_meta() {
        // Missing meta defaults to {fts, original, ""} upstream.
        let list = vec![result("doc1", 0.9)];
        let traces = build_rrf_trace(&[list], &[], &[], RRF_K);
        let t = &traces["doc1"];
        assert_eq!(t.contributions[0].source, SearchSource::Fts);
        assert_eq!(t.contributions[0].query_type, QueryKind::Original);
        assert_eq!(t.contributions[0].query, "");
        assert_eq!(t.contributions[0].rank, 1);
        assert!((t.contributions[0].weight - 1.0).abs() < 1e-10);
    }

    // ── AC5 golden (fixtures/hybrid_golden.json) ──

    fn golden() -> serde_json::Value {
        serde_json::from_str(include_str!("../../tests/fixtures/hybrid_golden.json")).unwrap()
    }

    fn lists_from(v: &serde_json::Value) -> Vec<Vec<RankedResult>> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|l| {
                l.as_array()
                    .unwrap()
                    .iter()
                    .map(|r| RankedResult {
                        file: r["file"].as_str().unwrap().to_owned(),
                        display_path: r["displayPath"].as_str().unwrap().to_owned(),
                        title: r["title"].as_str().unwrap().to_owned(),
                        body: r["body"].as_str().unwrap().to_owned(),
                        score: r["score"].as_f64().unwrap(),
                        docid: String::new(),
                        context: None,
                    })
                    .collect()
            })
            .collect()
    }

    fn metas_from(v: &serde_json::Value) -> Vec<RankedListMeta> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|m| RankedListMeta {
                source: match m["source"].as_str().unwrap() {
                    "vec" => SearchSource::Vec,
                    _ => SearchSource::Fts,
                },
                query_type: match m["queryType"].as_str().unwrap() {
                    "lex" => QueryKind::Lex,
                    "vec" => QueryKind::Vec,
                    "hyde" => QueryKind::Hyde,
                    _ => QueryKind::Original,
                },
                query: m["query"].as_str().unwrap().to_owned(),
            })
            .collect()
    }

    fn weights_from(v: &serde_json::Value) -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|w| w.as_f64().unwrap())
            .collect()
    }

    const EPS: f64 = 1e-12;

    #[test]
    fn fusion_matches_upstream_golden() {
        let g = golden();
        for case in g["cases"]["rrf"].as_array().unwrap() {
            let lists = lists_from(&case["lists"]);
            let weights = weights_from(&case["weights"]);
            let k = case["k"].as_u64().unwrap() as usize;
            let fused = reciprocal_rank_fusion(&lists, &weights, k);
            let want = case["fused"].as_array().unwrap();
            assert_eq!(fused.len(), want.len(), "{}", case["name"]);
            for (got, w) in fused.iter().zip(want) {
                assert_eq!(got.file, w["file"].as_str().unwrap(), "{}", case["name"]);
                assert!(
                    (got.score - w["score"].as_f64().unwrap()).abs() < EPS,
                    "{}: {} vs {}",
                    case["name"],
                    got.score,
                    w["score"]
                );
            }
        }
    }

    #[test]
    fn trace_matches_upstream_golden() {
        let g = golden();
        for case in g["cases"]["rrf_trace"].as_array().unwrap() {
            let lists = lists_from(&case["lists"]);
            let weights = weights_from(&case["weights"]);
            let metas = metas_from(&case["metas"]);
            let k = case["k"].as_u64().unwrap() as usize;
            let traces = build_rrf_trace(&lists, &weights, &metas, k);
            let want = case["trace"].as_object().unwrap();
            assert_eq!(traces.len(), want.len(), "{}", case["name"]);
            for (file, wt) in want {
                assert_trace_matches(
                    file,
                    traces
                        .get(file.as_str())
                        .unwrap_or_else(|| panic!("{}: missing trace for {file}", case["name"])),
                    wt,
                );
            }
        }
    }

    fn assert_trace_matches(file: &str, t: &FileRrfTrace, wt: &serde_json::Value) {
        let close = |a: f64, b: &serde_json::Value| (a - b.as_f64().unwrap()).abs() < EPS;
        assert!(close(t.base_score, &wt["baseScore"]), "{file} baseScore");
        assert_eq!(t.top_rank, wt["topRank"].as_u64().unwrap() as usize);
        assert!(
            close(t.top_rank_bonus, &wt["topRankBonus"]),
            "{file} topRankBonus"
        );
        assert!(close(t.total_score, &wt["totalScore"]), "{file} totalScore");
        let wcs = wt["contributions"].as_array().unwrap();
        assert_eq!(t.contributions.len(), wcs.len(), "{file} contributions");
        for (c, wc) in t.contributions.iter().zip(wcs) {
            assert_eq!(c.list_index, wc["listIndex"].as_u64().unwrap() as usize);
            assert_eq!(
                c.query_type.as_str(),
                wc["queryType"].as_str().unwrap(),
                "{file} queryType"
            );
            assert_eq!(c.query, wc["query"].as_str().unwrap());
            assert_eq!(c.rank, wc["rank"].as_u64().unwrap() as usize);
            assert!(close(c.weight, &wc["weight"]), "{file} weight");
            assert!(
                close(c.backend_score, &wc["backendScore"]),
                "{file} backendScore"
            );
            assert!(
                close(c.contribution, &wc["rrfContribution"]),
                "{file} rrfContribution"
            );
        }
    }

    #[test]
    fn hybrid_weights_match_upstream_golden() {
        let g = golden();
        for case in g["cases"]["hybrid_weights"].as_array().unwrap() {
            let metas = metas_from(&case["metas"]);
            assert_eq!(
                get_hybrid_rrf_weights(&metas),
                weights_from(&case["weights"]),
                "{case}"
            );
        }
    }
}
