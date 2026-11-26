# model-packs Worker

Serves browser model packs from the `austegard-media` R2 bucket at
`https://model-packs.austegard.workers.dev/<key>`. The SemanticHistory Chrome extension
(`oaustegard/browser-extensions`, `semantic-history/`) downloads the int8 SparseUp v2 pack from
`https://model-packs.austegard.workers.dev/sparseup-v2/int8-v1/`.

The Worker serves only keys under the prefixes in `PREFIXES` (`worker.js`) and answers 404 for the rest of
the bucket. It handles GET, HEAD, Range (206) and If-None-Match (304), sends CORS for any origin, and marks
responses immutable for a year, so add a prefix only for a pack that is never rewritten in place.

Deploy (needs `CF_API_TOKEN` with Workers and R2 access):

```bash
CLOUDFLARE_API_TOKEN="$CF_API_TOKEN" CLOUDFLARE_ACCOUNT_ID="$CF_ACCOUNT_ID" npx wrangler@4.158.0 deploy
```

First deployed 2026-11-26, version `2b6b8804-38aa-4e21-a654-b33ce51164fc`. On that day all four weight
parts, `manifest.json` and `tokenizer.json` matched the r2.dev copies by sha256.
