//! Integration tests for document retrieval (P1.4): `Qmd::get`,
//! `document_body`, `multi_get`, `list_documents`,
//! `resolve_virtual_path`, docid lookup, similar-file suggestions and
//! ignore-rule errors.
//!
//! Ported upstream coverage: `test/store.test.ts` Document Retrieval /
//! Fuzzy Matching groups and the `qmd ls` listing query
//! (cli/qmd.ts:1712-1736).

#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::shadow_unrelated,
    clippy::field_reassign_with_default,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::missing_assert_message,
    clippy::panic,
    reason = "idiomatic test-code patterns"
)]

use std::fs;
use std::path::Path;

use qmd::{
    CollectionSpec, Environment, GetOptions, LineRange, MultiGetEntry, MultiGetOptions, Qmd,
    UpdateOptions,
};

fn write(dir: &Path, rel: &str, content: &str) {
    let path = dir.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
}

fn open(db: &Path) -> Qmd {
    Qmd::builder(db)
        .environment(Environment::default())
        .build()
        .unwrap()
}

fn add_docs(qmd: &Qmd, name: &str, docs: &Path, ignore: &[&str]) {
    qmd.add_collection(
        name,
        &CollectionSpec {
            path: docs.to_string_lossy().into_owned(),
            ignore: ignore.iter().map(|s| (*s).to_owned()).collect(),
            ..CollectionSpec::default()
        },
    )
    .unwrap();
}

fn indexed_db(tmp: &Path) -> (Qmd, std::path::PathBuf) {
    let docs = tmp.join("docs");
    write(
        &docs,
        "hello.md",
        "# Hello\nline two\nline three\nline four\n",
    );
    write(&docs, "sub/b.md", "# B\nbody b\n");
    write(&docs, "other.txt", "txt body\n"); // outside the default mask
    let qmd = open(&tmp.join("i.sqlite"));
    add_docs(&qmd, "docs", &docs, &[]);
    qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap();
    (qmd, docs)
}

// --- get: lookup forms ------------------------------------------------------

#[test]
fn get_by_virtual_path() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _docs) = indexed_db(tmp.path());
    let doc = qmd
        .get("qmd://docs/hello.md", GetOptions::default())
        .unwrap();
    assert_eq!(doc.virtual_path, "qmd://docs/hello.md");
    assert_eq!(doc.display_path, "docs/hello.md");
    assert_eq!(doc.title, "Hello");
    assert_eq!(doc.collection, "docs");
    assert_eq!(doc.docid, doc.hash.get(..6).unwrap());
    assert_eq!(doc.body, None);
    assert!(doc.body_length > 0);
    assert!(!doc.modified_at.is_empty());
}

#[test]
fn get_by_collection_prefixed_and_bare_path() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _docs) = indexed_db(tmp.path());
    // `docs/hello.md` substring-matches the virtual path.
    let a = qmd.get("docs/hello.md", GetOptions::default()).unwrap();
    // `hello.md` resolves as a collection-relative filesystem path.
    let b = qmd.get("hello.md", GetOptions::default()).unwrap();
    assert_eq!(a.virtual_path, b.virtual_path);
    assert_eq!(b.docid, a.docid);
}

#[test]
fn get_by_absolute_path() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, docs) = indexed_db(tmp.path());
    let abs = docs.join("hello.md");
    let doc = qmd
        .get(abs.to_str().unwrap(), GetOptions::default())
        .unwrap();
    assert_eq!(doc.virtual_path, "qmd://docs/hello.md");
}

#[test]
fn get_by_docid_forms() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _docs) = indexed_db(tmp.path());
    let doc = qmd
        .get("qmd://docs/hello.md", GetOptions::default())
        .unwrap();
    for form in [
        format!("#{}", doc.docid),
        doc.docid.clone(),
        format!("\"{}\"", doc.docid),
        format!("'{}'", doc.docid),
    ] {
        let got = qmd.get(&form, GetOptions::default()).unwrap();
        assert_eq!(got.virtual_path, "qmd://docs/hello.md", "form {form}");
    }
}

#[test]
fn get_include_body() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _docs) = indexed_db(tmp.path());
    let doc = qmd
        .get("qmd://docs/hello.md", GetOptions { include_body: true })
        .unwrap();
    assert_eq!(
        doc.body.as_deref(),
        Some("# Hello\nline two\nline three\nline four\n")
    );
}

#[test]
fn get_strips_line_suffix_for_lookup() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _docs) = indexed_db(tmp.path());
    // Upstream `findDocument` strips `/:(\d+)$/` before lookup.
    let doc = qmd.get("hello.md:2", GetOptions::default()).unwrap();
    assert_eq!(doc.virtual_path, "qmd://docs/hello.md");
}

#[test]
fn get_unknown_docid_is_not_found() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _docs) = indexed_db(tmp.path());
    let err = qmd.get("#000000", GetOptions::default()).unwrap_err();
    match err {
        qmd::Error::DocumentNotFound {
            query,
            similar_files,
        } => {
            assert_eq!(query, "#000000");
            assert!(similar_files.is_empty());
        }
        e => panic!("expected DocumentNotFound, got {e:?}"),
    }
}

#[test]
fn get_miss_suggests_similar_files() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _docs) = indexed_db(tmp.path());
    let err = qmd.get("helo.md", GetOptions::default()).unwrap_err();
    match err {
        qmd::Error::DocumentNotFound { similar_files, .. } => {
            assert!(similar_files.iter().any(|s| s == "hello.md"));
        }
        e => panic!("expected DocumentNotFound, got {e:?}"),
    }
}

#[test]
fn get_ignored_path_reports_exclusion() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    write(&docs, "a.md", "# A\n");
    write(&docs, "secret/x.md", "# Secret\n");
    let qmd = open(&tmp.path().join("i.sqlite"));
    add_docs(&qmd, "docs", &docs, &["secret/**"]);
    qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap();

    // The file exists on disk but the ignore rule kept it out.
    let err = qmd
        .get("docs/secret/x.md", GetOptions::default())
        .unwrap_err();
    match err {
        qmd::Error::ExcludedByIgnore {
            collection, path, ..
        } => {
            assert_eq!(collection, "docs");
            assert_eq!(path, "secret/x.md");
        }
        e => panic!("expected ExcludedByIgnore, got {e:?}"),
    }
}

// --- document_body -----------------------------------------------------------

#[test]
fn document_body_line_windows() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _docs) = indexed_db(tmp.path());
    let q = "qmd://docs/hello.md";

    assert_eq!(
        qmd.document_body(q, LineRange::ALL).unwrap().unwrap(),
        "# Hello\nline two\nline three\nline four\n"
    );
    assert_eq!(
        qmd.document_body(
            q,
            LineRange {
                from_line: Some(2),
                max_lines: Some(2)
            }
        )
        .unwrap()
        .unwrap(),
        "line two\nline three"
    );
    // `from` past the end yields "" like upstream `slice`.
    assert_eq!(
        qmd.document_body(
            q,
            LineRange {
                from_line: Some(99),
                max_lines: None
            }
        )
        .unwrap()
        .unwrap(),
        ""
    );
}

#[test]
fn document_body_missing_is_none() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _docs) = indexed_db(tmp.path());
    assert_eq!(qmd.document_body("nope.md", LineRange::ALL).unwrap(), None);
    assert_eq!(qmd.document_body("#000000", LineRange::ALL).unwrap(), None);
}

// --- multi_get ----------------------------------------------------------------

fn hit_paths(mg: &qmd::MultiGet) -> Vec<&str> {
    mg.docs
        .iter()
        .filter_map(|e| match e {
            MultiGetEntry::Hit(d) => Some(d.virtual_path.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn multi_get_comma_list_and_docid() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _docs) = indexed_db(tmp.path());
    let doc = qmd
        .get("qmd://docs/hello.md", GetOptions::default())
        .unwrap();

    let mg = qmd
        .multi_get("hello.md, sub/b.md", &MultiGetOptions::default())
        .unwrap();
    assert_eq!(mg.errors, Vec::<String>::new());
    assert_eq!(
        hit_paths(&mg),
        vec!["qmd://docs/hello.md", "qmd://docs/sub/b.md"]
    );

    // A docid inside a comma list resolves too.
    let mg = qmd
        .multi_get(
            &format!("#{}, sub/b.md", doc.docid),
            &MultiGetOptions::default(),
        )
        .unwrap();
    assert_eq!(hit_paths(&mg).len(), 2);
}

#[test]
fn multi_get_glob() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _docs) = indexed_db(tmp.path());
    let mg = qmd
        .multi_get("**/*.md", &MultiGetOptions::default())
        .unwrap();
    assert_eq!(mg.errors, Vec::<String>::new());
    assert_eq!(mg.docs.len(), 2);
}

#[test]
fn multi_get_unmatched_pattern_reports_error() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _docs) = indexed_db(tmp.path());
    let mg = qmd
        .multi_get("**/*.xyz", &MultiGetOptions::default())
        .unwrap();
    assert_eq!(mg.docs.len(), 0);
    assert_eq!(mg.errors, vec!["No files matched pattern: **/*.xyz"]);
}

#[test]
fn multi_get_ambiguous_name_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let docs_a = tmp.path().join("a");
    let docs_b = tmp.path().join("b");
    write(&docs_a, "same.md", "# A\n");
    write(&docs_b, "same.md", "# B\n");
    let qmd = open(&tmp.path().join("i.sqlite"));
    add_docs(&qmd, "ca", &docs_a, &[]);
    add_docs(&qmd, "cb", &docs_b, &[]);
    qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap();

    // `same.md` exists in both collections — comma-list resolution must
    // report the ambiguity instead of picking one.
    let mg = qmd
        .multi_get("same.md, ca/same.md", &MultiGetOptions::default())
        .unwrap();
    assert!(mg.errors.iter().any(|e| e.contains("Ambiguous path")));
    // The qualified entry still resolves.
    assert_eq!(hit_paths(&mg), vec!["qmd://ca/same.md"]);
}

#[test]
fn multi_get_max_bytes_skips() {
    let tmp = tempfile::tempdir().unwrap();
    let docs = tmp.path().join("docs");
    write(&docs, "small.md", "# S\n");
    write(&docs, "big.md", &"x".repeat(2048));
    let qmd = open(&tmp.path().join("i.sqlite"));
    add_docs(&qmd, "docs", &docs, &[]);
    qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap();

    let mg = qmd
        .multi_get(
            "**/*.md",
            &MultiGetOptions {
                include_body: false,
                max_bytes: 100,
            },
        )
        .unwrap();
    let skipped = mg
        .docs
        .iter()
        .filter_map(|e| match e {
            MultiGetEntry::Skipped {
                virtual_path,
                reason,
                ..
            } => Some((virtual_path.as_str(), reason.as_str())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(skipped.len(), 1);
    assert_eq!(skipped[0].0, "qmd://docs/big.md");
    assert!(skipped[0].1.contains("File too large"));
}

// --- list_documents -----------------------------------------------------------

#[test]
fn list_documents_sorted_and_prefixed() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _docs) = indexed_db(tmp.path());
    let all = qmd.list_documents("docs", None).unwrap();
    assert_eq!(
        all.iter().map(|d| d.path.as_str()).collect::<Vec<_>>(),
        vec!["hello.md", "sub/b.md"]
    );
    assert!(all.iter().all(|d| d.size > 0));

    let sub = qmd.list_documents("docs", Some("sub/")).unwrap();
    assert_eq!(
        sub.iter().map(|d| d.path.as_str()).collect::<Vec<_>>(),
        vec!["sub/b.md"]
    );
}

#[test]
fn list_documents_unknown_collection() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, _docs) = indexed_db(tmp.path());
    match qmd.list_documents("nope", None) {
        Err(qmd::Error::CollectionNotFound { name }) => assert_eq!(name, "nope"),
        other => panic!("expected CollectionNotFound, got {other:?}"),
    }
}

// --- resolve_virtual_path -----------------------------------------------------

#[test]
fn resolve_virtual_path_cases() {
    let tmp = tempfile::tempdir().unwrap();
    let (qmd, docs) = indexed_db(tmp.path());
    assert_eq!(
        qmd.resolve_virtual_path("qmd://docs/hello.md").unwrap(),
        Some(docs.join("hello.md"))
    );
    // `..` escapes are refused by the containment check.
    assert_eq!(
        qmd.resolve_virtual_path("qmd://docs/../escape.md").unwrap(),
        None
    );
    assert_eq!(qmd.resolve_virtual_path("qmd://nope/a.md").unwrap(), None);
    assert_eq!(qmd.resolve_virtual_path("/abs/a.md").unwrap(), None);
}

// --- golden corpus: matches `scripts/gen_vpath_golden.ts` ----------------------

fn golden_corpus(tmp: &Path) -> Qmd {
    let docs = tmp.join("docs");
    let blog = tmp.join("blog");
    write(&docs, "hello.md", "# Hello\n");
    write(&docs, "sub/b.md", "# B\n");
    write(&docs, "sub/deep/c.md", "# C\n");
    write(&docs, "SYNTAX.md", "# Syntax\n");
    write(&docs, "same.md", "# Same docs\n");
    write(&blog, "same.md", "# Same blog\n");
    let qmd = open(&tmp.join("i.sqlite"));
    add_docs(&qmd, "docs", &docs, &[]);
    add_docs(&qmd, "blog", &blog, &[]);
    qmd.update(&UpdateOptions::default(), &mut |_| {}).unwrap();
    qmd
}

fn vpath_golden() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/vpath_golden.json")).unwrap()
}

#[test]
fn glob_golden() {
    let tmp = tempfile::tempdir().unwrap();
    let qmd = golden_corpus(tmp.path());
    for case in vpath_golden()["glob"].as_array().unwrap() {
        let pattern = case[0].as_str().unwrap();
        let expected: Vec<String> = case[1]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect();
        let mg = qmd.multi_get(pattern, &MultiGetOptions::default()).unwrap();
        let mut got: Vec<String> = hit_paths(&mg).iter().map(ToString::to_string).collect();
        let mut expected_sorted = expected;
        got.sort();
        expected_sorted.sort();
        assert_eq!(got, expected_sorted, "glob {pattern:?}");
    }
}

#[test]
fn comma_resolve_golden() {
    let tmp = tempfile::tempdir().unwrap();
    let qmd = golden_corpus(tmp.path());
    for case in vpath_golden()["comma_resolve"].as_array().unwrap() {
        let name = case[0].as_str().unwrap();
        // A trailing non-resolving name turns the input into the
        // comma-list branch; its own error is ignored.
        let pattern = format!("{name},zzz999zzz");
        let mg = qmd
            .multi_get(&pattern, &MultiGetOptions::default())
            .unwrap();
        if let Some(ok) = case[1].get("ok") {
            let vp = ok.as_str().unwrap();
            assert!(
                hit_paths(&mg).contains(&vp),
                "comma {name:?} should resolve to {vp}, got {mg:?}"
            );
        } else {
            let err = case[1]["err"].as_str().unwrap();
            assert!(
                mg.errors.iter().any(|e| e == err),
                "comma {name:?} should error {err:?}, got {:?}",
                mg.errors
            );
        }
    }
}

#[test]
fn docid_matches_upstream_hash() {
    // The fixture's `#90f8ec` was produced by upstream `hashContent`
    // (sha256) over `# Hello\n` — our `get` must resolve the same docid.
    let tmp = tempfile::tempdir().unwrap();
    let qmd = golden_corpus(tmp.path());
    let doc = qmd.get("#90f8ec", GetOptions::default()).unwrap();
    assert_eq!(doc.virtual_path, "qmd://docs/hello.md");
}
