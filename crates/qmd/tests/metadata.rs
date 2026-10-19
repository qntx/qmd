//! Frontmatter metadata extraction and normalization tests — port of
//! upstream `test/metadata.test.ts`. Golden byte-for-byte coverage lives
//! in `metadata_golden.rs`.

#![allow(
    unused_crate_dependencies,
    reason = "each test binary only uses a subset of the package deps"
)]
#![allow(
    clippy::tests_outside_test_module,
    clippy::unwrap_used,
    clippy::missing_assert_message,
    clippy::panic,
    clippy::shadow_unrelated,
    reason = "idiomatic test-code patterns"
)]

use qmd::{
    DocumentMetadata, METADATA_EXTRACTION_VERSION, METADATA_LIMITS, MetadataValue,
    extract_document_metadata, utf16_len,
};

fn build_doc(frontmatter_yaml: &str) -> String {
    build_doc_body(frontmatter_yaml, "# Title\n\nBody text.\n")
}

fn build_doc_body(frontmatter_yaml: &str, body: &str) -> String {
    format!("---\n{frontmatter_yaml}---\n\n{body}")
}

fn metadata_json(content: &str, path: &str) -> String {
    extract_document_metadata(content, path).metadata.to_json()
}

fn is_empty(metadata: &DocumentMetadata) -> bool {
    metadata.is_empty()
}

// =============================================================================
// extractDocumentMetadata
// =============================================================================

#[test]
fn document_without_frontmatter_yields_empty_metadata() {
    let extraction = extract_document_metadata("# Title\n\nBody.\n", "doc.md");
    assert!(is_empty(&extraction.metadata));
    assert!(extraction.error.is_none());
    assert_eq!(extraction.extraction_version, METADATA_EXTRACTION_VERSION);
}

#[test]
fn frontmatter_without_qmd_namespace_yields_empty_metadata() {
    let extraction =
        extract_document_metadata(build_doc("title: Hello\ntags: [a, b]\n").as_str(), "doc.md");
    assert!(is_empty(&extraction.metadata));
    assert!(extraction.error.is_none());
}

#[test]
fn qmd_namespace_without_metadata_yields_empty_metadata() {
    let extraction =
        extract_document_metadata(build_doc("qmd:\n  other: true\n").as_str(), "doc.md");
    assert!(is_empty(&extraction.metadata));
    assert!(extraction.error.is_none());
}

#[test]
fn empty_qmd_metadata_mapping_yields_empty_metadata() {
    let extraction =
        extract_document_metadata(build_doc("qmd:\n  metadata: {}\n").as_str(), "doc.md");
    assert!(is_empty(&extraction.metadata));
    assert!(extraction.error.is_none());
}

#[test]
fn extracts_string_number_and_boolean_scalars() {
    let extraction = extract_document_metadata(
        build_doc(
            "qmd:\n  metadata:\n    status: published\n    priority: 3\n    reviewed: true\n",
        )
        .as_str(),
        "doc.md",
    );
    assert_eq!(
        extraction.metadata.to_json(),
        r#"{"status":"published","priority":3,"reviewed":true}"#
    );
    assert!(extraction.error.is_none());
}

#[test]
fn extracts_homogeneous_arrays_of_every_scalar_type() {
    let yaml = [
        "qmd:",
        "  metadata:",
        "    topics: [typescript, programming]",
        "    scores: [1, 2.5, 3]",
        "    flags: [true, false]",
        "",
    ]
    .join("\n");
    let extraction = extract_document_metadata(build_doc(&yaml).as_str(), "doc.md");
    assert_eq!(
        extraction.metadata.to_json(),
        r#"{"topics":["typescript","programming"],"scores":[1,2.5,3],"flags":[true,false]}"#
    );
}

#[test]
fn de_duplicates_array_values_preserving_first_seen_order() {
    let extraction = extract_document_metadata(
        build_doc("qmd:\n  metadata:\n    topics: [b, a, b, c, a]\n").as_str(),
        "doc.md",
    );
    assert_eq!(
        extraction.metadata.get("topics"),
        Some(&MetadataValue::Strings(vec![
            "b".to_owned(),
            "a".to_owned(),
            "c".to_owned()
        ]))
    );
}

#[test]
fn treats_prototype_shaped_metadata_keys_as_ordinary_data() {
    let extraction = extract_document_metadata(
        build_doc("qmd:\n  metadata:\n    __proto__: inherited\n    constructor: built\n").as_str(),
        "doc.md",
    );
    let keys: Vec<&str> = extraction.metadata.iter().map(|(k, _)| k).collect();
    assert_eq!(keys, ["__proto__", "constructor"]);
    assert_eq!(
        extraction.metadata.get("__proto__"),
        Some(&MetadataValue::String("inherited".to_owned()))
    );
    assert_eq!(
        extraction.metadata.get("constructor"),
        Some(&MetadataValue::String("built".to_owned()))
    );
    assert_eq!(
        extraction.metadata.to_json(),
        r#"{"__proto__":"inherited","constructor":"built"}"#
    );
}

#[test]
fn tolerates_bom_crlf_and_dots_closing_marker() {
    let bom_doc = "\u{feff}---\nqmd:\n  metadata:\n    status: ok\n---\nBody\n";
    assert_eq!(metadata_json(bom_doc, "doc.md"), r#"{"status":"ok"}"#);

    let crlf_doc = "---\r\nqmd:\r\n  metadata:\r\n    status: ok\r\n---\r\nBody\r\n";
    assert_eq!(metadata_json(crlf_doc, "doc.md"), r#"{"status":"ok"}"#);

    let dots_doc = "---\nqmd:\n  metadata:\n    status: ok\n...\nBody\n";
    assert_eq!(metadata_json(dots_doc, "doc.md"), r#"{"status":"ok"}"#);
}

#[test]
fn missing_closing_delimiter_is_treated_as_no_frontmatter() {
    let extraction =
        extract_document_metadata("---\nqmd:\n  metadata:\n    status: ok\n", "doc.md");
    assert!(is_empty(&extraction.metadata));
    assert!(extraction.error.is_none());
    assert_eq!(extraction.extraction_version, METADATA_EXTRACTION_VERSION);
}

#[test]
fn non_markdown_extensions_are_not_parsed_as_frontmatter() {
    let content = build_doc("qmd:\n  metadata:\n    status: ok\n");
    assert_eq!(metadata_json(&content, "script.ts"), "{}");
    assert_eq!(metadata_json(&content, "noext"), "{}");
    assert_eq!(
        metadata_json(&content, "doc.markdown"),
        r#"{"status":"ok"}"#
    );
    assert_eq!(metadata_json(&content, "doc.mdx"), r#"{"status":"ok"}"#);
}

#[test]
fn malformed_yaml_records_an_extraction_error() {
    let extraction = extract_document_metadata(build_doc("qmd: [unclosed\n").as_str(), "doc.md");
    assert!(is_empty(&extraction.metadata));
    assert!(
        extraction
            .error
            .as_deref()
            .is_some_and(|e| e.contains("invalid frontmatter YAML"))
    );
}

#[test]
fn non_mapping_qmd_or_qmd_metadata_records_an_extraction_error() {
    let qmd_scalar = extract_document_metadata(build_doc("qmd: hello\n").as_str(), "doc.md");
    assert!(
        qmd_scalar
            .error
            .as_deref()
            .is_some_and(|e| e.contains("'qmd' must be a mapping"))
    );

    let metadata_scalar =
        extract_document_metadata(build_doc("qmd:\n  metadata: hello\n").as_str(), "doc.md");
    assert!(
        metadata_scalar
            .error
            .as_deref()
            .is_some_and(|e| e.contains("'qmd.metadata' must be a mapping"))
    );
}

#[test]
fn rejects_nested_objects_null_empty_nested_and_mixed_arrays() {
    let cases: &[(&str, &str)] = &[
        (
            "qmd:\n  metadata:\n    nested:\n      a: 1\n",
            "unsupported value type",
        ),
        (
            "qmd:\n  metadata:\n    empty: null\n",
            "null is not supported",
        ),
        (
            "qmd:\n  metadata:\n    empty: []\n",
            "empty arrays are not supported",
        ),
        (
            "qmd:\n  metadata:\n    nested: [[1, 2]]\n",
            "nested arrays are not supported",
        ),
        (
            "qmd:\n  metadata:\n    mixed: [1, two]\n",
            "mixed-type arrays are not supported",
        ),
    ];
    for (yaml, want) in cases {
        let extraction = extract_document_metadata(build_doc(yaml).as_str(), "doc.md");
        assert!(is_empty(&extraction.metadata));
        assert!(
            extraction
                .error
                .as_deref()
                .is_some_and(|e| e.contains(want)),
            "{yaml:?} -> {:?}",
            extraction.error
        );
    }
}

#[test]
fn rejects_non_finite_numbers() {
    let extraction = extract_document_metadata(
        build_doc("qmd:\n  metadata:\n    bad: .inf\n").as_str(),
        "doc.md",
    );
    assert!(is_empty(&extraction.metadata));
    assert!(
        extraction
            .error
            .as_deref()
            .is_some_and(|e| e.contains("finite"))
    );
}

#[test]
fn rejects_oversized_keys_strings_arrays_and_key_counts() {
    let long_key = "k".repeat(METADATA_LIMITS.max_key_bytes + 1);
    let doc = build_doc(&format!("qmd:\n  metadata:\n    {long_key}: 1\n"));
    assert!(
        extract_document_metadata(&doc, "doc.md")
            .error
            .as_deref()
            .is_some_and(|e| e.contains("key exceeds"))
    );

    let long_string = "v".repeat(METADATA_LIMITS.max_string_length + 1);
    let doc = build_doc(&format!("qmd:\n  metadata:\n    long: \"{long_string}\"\n"));
    assert!(
        extract_document_metadata(&doc, "doc.md")
            .error
            .as_deref()
            .is_some_and(|e| e.contains("string exceeds"))
    );

    let big_array = format!(
        "[{}]",
        (0..=METADATA_LIMITS.max_array_length)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let doc = build_doc(&format!("qmd:\n  metadata:\n    big: {big_array}\n"));
    assert!(
        extract_document_metadata(&doc, "doc.md")
            .error
            .as_deref()
            .is_some_and(|e| e.contains("array exceeds"))
    );

    let many_keys = (0..=METADATA_LIMITS.max_keys)
        .map(|i| format!("    key{i}: 1"))
        .collect::<Vec<_>>()
        .join("\n");
    let doc = build_doc(&format!("qmd:\n  metadata:\n{many_keys}\n"));
    assert!(
        extract_document_metadata(&doc, "doc.md")
            .error
            .as_deref()
            .is_some_and(|e| e.contains("keys (max"))
    );
}

#[test]
fn rejects_oversized_frontmatter_blocks() {
    let filler = format!(
        "filler: \"{}\"\n",
        "x".repeat(METADATA_LIMITS.max_frontmatter_bytes)
    );
    let doc = build_doc(&format!("{filler}qmd:\n  metadata:\n    status: ok\n"));
    let extraction = extract_document_metadata(&doc, "doc.md");
    assert!(is_empty(&extraction.metadata));
    assert!(
        extraction
            .error
            .as_deref()
            .is_some_and(|e| e.contains("frontmatter exceeds"))
    );
}

#[test]
fn bounds_yaml_alias_expansion() {
    let alias_bomb = [
        "a: &a [x, x, x, x, x, x, x, x, x, x]",
        "b: &b [*a, *a, *a, *a, *a, *a, *a, *a, *a, *a]",
        "c: &c [*b, *b, *b, *b, *b, *b, *b, *b, *b, *b]",
        "qmd:",
        "  metadata:",
        "    status: ok",
        "",
    ]
    .join("\n");
    let extraction = extract_document_metadata(build_doc(&alias_bomb).as_str(), "doc.md");
    assert!(is_empty(&extraction.metadata));
    assert!(
        extraction
            .error
            .as_deref()
            .is_some_and(|e| e.contains("invalid frontmatter YAML"))
    );
}

#[test]
fn bounds_extraction_error_message_length() {
    let long_key = "k".repeat(600);
    let doc = build_doc(&format!(
        "qmd:\n  metadata:\n    valid: 1\n    \"{long_key}x\": 1\n"
    ));
    let extraction = extract_document_metadata(&doc, "doc.md");
    let error = extraction.error.as_deref().unwrap_or_default();
    assert!(!error.is_empty());
    assert!(utf16_len(error) <= METADATA_LIMITS.max_error_length);
}

#[test]
fn unquoted_dates_stay_strings() {
    let extraction = extract_document_metadata(
        build_doc("qmd:\n  metadata:\n    published: 2024-01-15\n").as_str(),
        "doc.md",
    );
    assert_eq!(
        extraction.metadata.get("published"),
        Some(&MetadataValue::String("2024-01-15".to_owned()))
    );
    assert_eq!(
        extraction.metadata.to_json(),
        r#"{"published":"2024-01-15"}"#
    );
}
