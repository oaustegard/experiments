# ARTIFACTS: large files not in git

Built by the scripts in `run/` (see RESULTS.md, Reproduce). Published as GitHub release
[`ettin-native-20261021`](https://github.com/oaustegard/experiments/releases/tag/ettin-native-20261021):
each ONNX build and checkpoint directory below is one zip (named in the release notes), the
teacher vectors are `teacher-arctic-m-v1.5.f16.npy` and `teacher-train.jsonl.gz`, and
`SHA256SUMS` covers the release files. The per-file sha256 below are of the unzipped files.

## ONNX builds (browser-ready: model.onnx + tokenizer.json + meta.json)

| path | bytes | sha256 |
|---|---|---|
| `out/models/e17c/t4-w8L63/meta.json` | 202 | `c808d964c73c7de3…` |
| `out/models/e17c/t4-w8L63/model.onnx` | 11,590,733 | `c651ae63ce5da707…` |
| `out/models/e17c/t4-w8L63/tokenizer.json` | 1,774,483 | `b6c3b82c35eb6265…` |
| `out/models/e17c/t3-w8L63/meta.json` | 202 | `c808d964c73c7de3…` |
| `out/models/e17c/t3-w8L63/model.onnx` | 10,382,479 | `24549c472b678527…` |
| `out/models/e17c/t3-w8L63/tokenizer.json` | 1,774,483 | `b6c3b82c35eb6265…` |
| `out/models/e17c/t2-w8L63/meta.json` | 202 | `c808d964c73c7de3…` |
| `out/models/e17c/t2-w8L63/model.onnx` | 8,770,699 | `7877a0ccccd3a890…` |
| `out/models/e17c/t2-w8L63/tokenizer.json` | 1,774,483 | `b6c3b82c35eb6265…` |
| `out/models/e17c/t8-w8/meta.json` | 202 | `c808d964c73c7de3…` |
| `out/models/e17c/t8-w8/model.onnx` | 17,432,322 | `ddae25b1b1535e13…` |
| `out/models/e17c/t8-w8/tokenizer.json` | 1,774,483 | `b6c3b82c35eb6265…` |
| `out/models/leafirc/t4-w8L63/meta.json` | 195 | `69dd263c33ddea31…` |
| `out/models/leafirc/t4-w8L63/model.onnx` | 19,507,578 | `ea069228e4ac9c6e…` |
| `out/models/leafirc/t4-w8L63/tokenizer.json` | 527,497 | `ce5ec5092f69344c…` |
| `out/models/leafirc/t8-w8/meta.json` | 195 | `69dd263c33ddea31…` |
| `out/models/leafirc/t8-w8/model.onnx` | 24,756,249 | `e3130ead581226ee…` |
| `out/models/leafirc/t8-w8/tokenizer.json` | 527,497 | `ce5ec5092f69344c…` |

## SentenceTransformer checkpoints (fp32)

| path | bytes | sha256 |
|---|---|---|
| `out/contrast/e17-cap-c/epoch1/1_Pooling/config.json` | 90 | `e27b2b16abb46773…` |
| `out/contrast/e17-cap-c/epoch1/2_Dense/config.json` | 228 | `5fa10e9f7d399884…` |
| `out/contrast/e17-cap-c/epoch1/2_Dense/model.safetensors` | 786,520 | `a74e81b71b3681cb…` |
| `out/contrast/e17-cap-c/epoch1/README.md` | 5,308 | `93271ecd40e14b04…` |
| `out/contrast/e17-cap-c/epoch1/config.json` | 1,580 | `6c4dc9edf966e303…` |
| `out/contrast/e17-cap-c/epoch1/config_sentence_transformers.json` | 335 | `9cb1fe8cbccc19ec…` |
| `out/contrast/e17-cap-c/epoch1/model.safetensors` | 67,193,928 | `106a0f9eefae577b…` |
| `out/contrast/e17-cap-c/epoch1/modules.json` | 401 | `8662aac5f6652b27…` |
| `out/contrast/e17-cap-c/epoch1/sentence_bert_config.json` | 241 | `3084164002c0bca0…` |
| `out/contrast/e17-cap-c/epoch1/tokenizer.json` | 3,583,485 | `218484396a9d0829…` |
| `out/contrast/e17-cap-c/epoch1/tokenizer_config.json` | 569 | `d2f408ae312df2a6…` |
| `out/contrast/leafir-c/epoch2/1_Pooling/config.json` | 90 | `4fef0a7e8c8ee36e…` |
| `out/contrast/leafir-c/epoch2/2_Dense/config.json` | 227 | `8d9708677c7333b3…` |
| `out/contrast/leafir-c/epoch2/2_Dense/model.safetensors` | 1,182,880 | `a3766a238b6316e5…` |
| `out/contrast/leafir-c/epoch2/3_Normalize/config.json` | 97 | `cdb09dfca347a56a…` |
| `out/contrast/leafir-c/epoch2/README.md` | 13,975 | `44385c5469cd6761…` |
| `out/contrast/leafir-c/epoch2/config.json` | 958 | `445c461a51c99a51…` |
| `out/contrast/leafir-c/epoch2/config_sentence_transformers.json` | 335 | `9cb1fe8cbccc19ec…` |
| `out/contrast/leafir-c/epoch2/model.safetensors` | 90,272,656 | `62ac71cffd83bc1a…` |
| `out/contrast/leafir-c/epoch2/modules.json` | 537 | `47f1c99c16e43861…` |
| `out/contrast/leafir-c/epoch2/sentence_bert_config.json` | 241 | `3084164002c0bca0…` |
| `out/contrast/leafir-c/epoch2/tokenizer.json` | 711,649 | `2fc687b11de0bc1b…` |
| `out/contrast/leafir-c/epoch2/tokenizer_config.json` | 635 | `9405b311958c7c5e…` |
| `out/students/e17-cap/epoch3/1_Pooling/config.json` | 90 | `e27b2b16abb46773…` |
| `out/students/e17-cap/epoch3/2_Dense/config.json` | 228 | `5fa10e9f7d399884…` |
| `out/students/e17-cap/epoch3/2_Dense/model.safetensors` | 786,520 | `0438272af94195aa…` |
| `out/students/e17-cap/epoch3/README.md` | 5,308 | `93271ecd40e14b04…` |
| `out/students/e17-cap/epoch3/config.json` | 1,580 | `6c4dc9edf966e303…` |
| `out/students/e17-cap/epoch3/config_sentence_transformers.json` | 335 | `9cb1fe8cbccc19ec…` |
| `out/students/e17-cap/epoch3/model.safetensors` | 67,193,928 | `d6e8a7cc30688aef…` |
| `out/students/e17-cap/epoch3/modules.json` | 401 | `8662aac5f6652b27…` |
| `out/students/e17-cap/epoch3/sentence_bert_config.json` | 241 | `3084164002c0bca0…` |
| `out/students/e17-cap/epoch3/tokenizer.json` | 3,583,485 | `218484396a9d0829…` |
| `out/students/e17-cap/epoch3/tokenizer_config.json` | 570 | `8d8165669734e04b…` |

## Teacher targets

| path | bytes | sha256 |
|---|---|---|
| `out/data/snowflake-arctic-embed-m-v1.5.f16.npy` | 533,753,984 | `b6508e5b121f4f3e…` |
| `out/data/train.jsonl` | 66,866,193 | `a049e5ce306e9d6f…` |
