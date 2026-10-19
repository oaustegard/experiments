# eg2-query-embedder

EmbeddingGemma 2 query embedding as a hosted endpoint, for pages that embed
their corpus in the browser (see `embeddinggemma2-webgpu-tail/`) but should not
make a casual visitor download the 174 MB model to run one query. Results and
reasoning: [`RESULTS.md`](RESULTS.md).

Live: `POST https://eg2-query-embedder.austegard.workers.dev/embed` with
`{"text": "...", "dims": 768}` from an allowed origin. `POST /warm` starts the
container without waiting; call it when the search box gets focus.

## Layout

```
server/   Node server (server.cjs), Dockerfile, correctness gate, cold-start bench
worker/   Cloudflare Worker + Container config, live check
results/  bench and live-check outputs cited by RESULTS.md
```

## Check

```bash
cd server
ONNXRUNTIME_NODE_INSTALL_CUDA=skip npm ci
mkdir -p model && for f in onnx/model_q4.onnx onnx/model_q4.onnx_data tokenizer.json tokenizer_config.json config_sentence_transformers.json; do
  curl -sSfL -o model/$(basename $f) https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/daa72c51243991dfcaf9f9137d2c573d8f7790c0/$f; done
node test_parity.mjs                       # model card scores; exit 1 on a miss
node bench_cold.mjs --reps 3 --drop-caches # host cold start (drop-caches needs root)
docker build --platform linux/amd64 -t eg2-query-embedder:v1 .
node bench_cold.mjs --reps 3 --drop-caches --docker eg2-query-embedder:v1 --cpus 1 --memory 3g --threads 1
cd ../worker && node live_check.mjs        # against the deployed Worker
```

## Deploy

```bash
cd worker && npm ci && npx wrangler deploy
```

`wrangler.jsonc` builds the image from `server/Dockerfile`. In a Claude Code on
the Web session npm cannot reach the registry from inside `docker build`, so
the live deploy used an image built locally with the proxy and CA passed in,
pushed with `wrangler containers push eg2-query-embedder:v1`, and a copy of the
config whose `image` points at `registry.cloudflare.com/<account>/eg2-query-embedder:v1`
(`wrangler.deploy.json`, gitignored).
