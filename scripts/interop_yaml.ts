#!/usr/bin/env bun
/**
 * AC4 YAML interop helper: load a config file through upstream
 * `loadConfig` and re-serialize it with upstream `saveConfig`.
 *
 * Usage:
 *   QMD_UPSTREAM=/path/to/tobi/qmd bun scripts/interop_yaml.ts <in.yaml> <out.yaml>
 *
 * Prints the upstream-parsed config as JSON on stdout. Driven by the
 * `config_interop` test (`cargo test -- --ignored config_interop`).
 */

const [input, output] = process.argv.slice(2);
const upstream = process.env.QMD_UPSTREAM;
if (!upstream || !input || !output) {
  console.error("QMD_UPSTREAM=<upstream checkout> and <in.yaml> <out.yaml> are required");
  process.exit(1);
}

const { setConfigSource, loadConfig, saveConfig } = await import(
  `${upstream}/src/collections.ts`
);

setConfigSource({ configPath: input });
const config = loadConfig();
console.log(JSON.stringify(config));

setConfigSource({ configPath: output });
saveConfig(config);
