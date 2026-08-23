# nl2sh

Search shell documentation in plain words. Attach a model if you want one to
write the command.

```console
$ nl2sh search "give everyone read write and execute permission on key.txt"
1. chmod   0.612
   Change the access permissions of a file or directory.
   $ chmod 777 path/to/file

2. chown   0.488
   Change user and group ownership of a file or directory.
   $ chown user:group path/to/file
```

With no model configured that is the whole program: ranked documentation, no
network, no API key, no weights on disk. Point it at a model and the same
retrieved pages become the context for a generated command.

```console
$ nl2sh --backend ollama --model qwen2.5-coder:1.5b \
        "give everyone read write and execute permission on key.txt"
chmod 777 key.txt
# ollama:qwen2.5-coder:1.5b  0.31s, 8 tokens
```

`nl2sh` prints commands. It never runs them.

## Install

```bash
pip install nl2sh                 # search, plus every remote backend
pip install 'nl2sh[llamacpp]'     # add in-process GGUF
pip install 'nl2sh[transformers]' # add in-process Hugging Face
pip install 'nl2sh[mlx]'          # add Apple Silicon
```

Remote backends and local daemons need no extra: both are HTTP, and `nl2sh`
speaks it with the standard library. The extras exist only for runtimes that
load weights inside this process.

## Backends

`nl2sh doctor` probes all of them and says what each one needs.

```console
$ nl2sh doctor
backends    (* = the one configured)
  * ok  none           no model; ranked documentation only
    --  ollama         http://localhost:11434/api/tags unreachable: Connection refused — is `ollama serve` running?
    --  llamacpp       no GGUF given; --model is a path or repo:file
    --  openai         set OPENAI_API_KEY (or pass --api-key)
```

| backend | runs | `--model` is |
|---|---|---|
| `none` | nothing | — |
| `ollama` | local daemon | a pulled tag, `qwen2.5-coder:1.5b` |
| `llamacpp` | in-process GGUF | a path, or `org/repo:file.gguf` |
| `llama-server` | local llama.cpp server | whatever it serves |
| `lmstudio` | LM Studio's server | whatever it serves |
| `mlx` | Apple Silicon | an `mlx-community` repo id |
| `transformers` | in-process PyTorch | a Hugging Face repo id |
| `openai` | any `/v1/chat/completions` | that server's model id |
| `anthropic` | Claude | `claude-haiku-4-5-20251001` |
| `gemini` | Gemini | `gemini-3.5-flash-lite` |

`openai` covers vLLM, Together, Groq, OpenRouter and anything else serving that
route. Set `--base-url` and the key.

## Configuration

Flags beat environment beats `~/.config/nl2sh/config.toml` beats defaults.
`nl2sh config` prints the resolved values and where each one came from.

```toml
[nl2sh]
backend = "ollama"
model = "qwen2.5-coder:1.5b"
k = 3
```

## Accuracy by model size

The retrieval tier surfaces the right utility 0.506 of the time on a
132-utility evaluation whose constant-utility prior is 0.012, and it answers in
milliseconds on a CPU. 0.506 rather than the 0.555 the research quotes, because
the query adapter that separates them is not shipped — see Limits. A generator on top costs weights, memory and seconds per
request, and below about a billion parameters it does not pay for them.

Measured on the same 164 requests, oracle sources, greedy decoding
(`oaustegard/experiments`, `nl2sh-instantiate/`):

| model | routing | usable | exact | tok/s |
|---|---|---|---|---|
| Gemma 3 270M, zero-shot | 0.427 | 0.427 | 0.079 | 17.0 |
| Gemma 3 270M, fine-tuned | 0.610 | 0.470 | 0.030 | 20.3 |
| Gemma 3 1B, zero-shot | 0.799 | 0.793 | 0.195 | 1.5 |
| Gemma 3 4B, zero-shot | 0.957 | 0.957 | 0.402 | 0.4 |

*usable* = names the right utility and is not a token-repeat loop.

Two things in that table set the defaults here. A 1B instruction model with no
training beats a fine-tuned 270M on every column, so `nl2sh` ships no
fine-tuned weights and no training path. And routing overstates what you can
run: scored by execution in a fixture rather than by its leading token, the 1B
lands at 0.104 against its 0.799 routing.

## Prompt shapes

`nl2sh` hands the model retrieved examples and asks it to substitute your
values into one, rather than to write a command from scratch. At 1B that beats
free generation on every column — routing 0.848 against 0.799, literal
reproduction 0.688 against 0.542, exact 0.280 against 0.195.

The same prompt is wrong below about 1B. A 270M model answers in the shape of
the example lines it was shown on 77% of requests, which drops its routing to
0.146 against 0.500 for `generate_anchored`. `nl2sh` warns when a model whose
name looks small is paired with the default and suggests the other one; it does
not switch silently, because the size of a model behind an HTTP endpoint is not
something this program can know.

## Reasoning models

A model that thinks before answering needs a token budget for the thinking.
Nemotron 3 Nano scored 0.000 under the default 64 tokens — every response
truncated mid-thought — and answered correctly on 3 of 4 probe requests at 200,
using 107 new tokens on average to produce a command of about 15. Raise
`--max-tokens` for those models, and read the latency before deciding they
belong on a laptop.

`nl2sh` strips a closed `<think>` block before parsing. An unterminated one
returns nothing rather than the last line of the model's scratchpad.

## Provenance

The retrieval tier, the prompts and the evaluation come from the `nl2sh-*`
directories in [`oaustegard/experiments`](https://github.com/oaustegard/experiments):
`nl2sh-retrieval` built the hybrid index, `nl2sh-dense` the enriched corpus and
the dense arm, `nl2sh-instantiate` the prompt comparison and the model bake-off.

`nl2sh/vendor/` carries `prompts.py` and `extract_params.py` copied verbatim
from those directories with their hashes pinned, because every number quoted
above was produced by those exact bytes. `vendor.sh` re-copies and re-pins them.

## Limits

- The query-side adapter measured in `nl2sh-retrieval` is not included, which
  is the whole difference between 0.506 here and the 0.555 that directory
  reports. It gained +0.184 on the 207 utilities it trained on and **lost**
  0.039 on utilities it had not seen, so shipping it would trade a headline
  number for worse behaviour on exactly the commands a user is most likely to
  be looking up.
- The evaluation corpus is auto-generated and carries noise. One request reads
  "Scan all ports on 192." against a gold command of `npm -p- 192.168.130.0`,
  which is not a port scan.
- Execution scoring decides 0.22 of the evaluation. The rest names tooling the
  scoring container does not have, or commands the sandbox refuses.
- Search latency: about 4-10 ms per query once warm, ~3 s for the first query
  in a process, and 2-3 minutes the very first time ever, when it encodes all
  6,397 pages. The encoder download is ~26 MB.
- `nl2sh search` currently needs the `oaustegard/experiments` checkout beside
  it for the corpus and the retrieval modules; a bare `pip install` gets the
  backends and the CLI but not a working index. `nl2sh doctor` says which.
  Packaging the corpus is the open item.
- The accuracy table's tok/s came from runs pinned to 2 of a 4-vCPU box with
  another model decoding on the other two, so they understate. Re-measured
  alone at 4 threads, decode is roughly double: 20.5 tok/s for 270M float32,
  18.6 bfloat16, 4.2 for 1B, 1.0 for 4B. Time to first token is the number
  that decides whether this feels usable — 178 ms, 446 ms, 2.3 s and 10.5 s
  respectively. No accelerator was involved anywhere; read every latency as a
  floor for real hardware.
