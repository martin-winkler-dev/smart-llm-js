# @smart-llm/gemini

[![Test](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml)
[![Package](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml)
[![Bun](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fgemini%2Fpackage.json&query=%24.engines.bun&label=Bun&logo=bun&color=blue)](https://bun.sh)
[![Node.js](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fgemini%2Fpackage.json&query=%24.engines.node&label=Node.js&logo=nodedotjs&color=blue)](https://nodejs.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/martin-winkler-dev/smart-llm-js/blob/main/LICENSE)

Gemini provider adapter (`createGeminiProvider`, `GEMINI_SUPPORT`) for [smart-llm](https://github.com/martin-winkler-dev/smart-llm-js#readme), supporting the Gemini Developer API (AI Studio) and Vertex AI via [smart-genai](https://www.npmjs.com/package/smart-genai). No classes; post-initialization methods never throw.

## Install

```bash
bun add @smart-llm/gemini @smart-llm/core
# or
npm install @smart-llm/gemini @smart-llm/core
```

ESM package with TypeScript types for Node.js >= 22.12 (`require()` supported) and Bun >= 1.3.

## Quickstart

```ts
import { createLlm } from '@smart-llm/core';
import { createGeminiProvider } from '@smart-llm/gemini';

const llm = createLlm({
    providers: {
        gemini: {
            adapter: createGeminiProvider(),              // env GEMINI_API_KEY, or createGeminiProvider({ vertex: true })
            models: {
                flash: {
                    model: 'gemini-3.8-flash',
                    capabilities: {
                        images: true,
                        tools: true,
                        json: true,
                        webSearch: { supported: true },
                        reasoning: { supported: true, default: 'low' },
                    },
                    modelLimits: { rpm: 1_000, tpm: 1_000_000 },
                },
                vec: {
                    model: 'gemini-embedding-2',
                    capabilities: { embeddings: true },
                    modelLimits: { rpm: 100 },
                },
            },
        },
    },
});

const answer = await llm.flash('Summarize quantum tunneling in two sentences.', { web_search: true });
const vectors = await llm.embed({ model: 'vec', input: ['First paragraph', 'Second paragraph'], dimensions: 768 });
```

## Options

`createGeminiProvider(clientOrOptions?)` accepts an existing `GenaiClient` instance from `smart-genai` or `Omit<GenaiClientOptions, 'defaults'>`:

| Option | Default | Description |
|---|---|---|
| `apiKey` | env `GEMINI_API_KEY` | Developer API key (Developer API only) |
| `vertex` | `undefined` | `true` or `{ project?, location?, bucket?, credentialsFile?, authClient?, storageUrl? }` for Vertex AI |
| `baseUrl` | official endpoint | Custom API endpoint URL |
| `env` | `process.env` | Environment map used for credential and project lookup |
| `timeoutMs` | `120_000` | HTTP request timeout per call in milliseconds (2 minutes) |
| `batchWaitSec` | `10` | Aggregation window in seconds for batch calls (`1` to `600`) |

Passing `defaults` to `createGeminiProvider` throws a `TypeError`; configure request defaults in `createLlm`.

## Supported Capabilities (`GEMINI_SUPPORT`)

```ts
const GEMINI_SUPPORT = {
    images: true,
    tools: true,
    webSearch: true,
    reasoning: true,
    batch: true,
    json: 'native',
    embeddings: true,
};
```

- **Structured JSON (`native` mode)**: Forwards cleaned JSON Schemas natively as Gemini `responseJsonSchema` and validates responses against the schema.
- **Reasoning**: Maps `reasoning_effort` (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`) via `smart-genai` to thinking budgets on Gemini 2.x (`0` for `none`, `128` minimum on 2.5 Pro) or thinking levels on Gemini 3.x (`none` maps to `minimal` as Gemini 3 cannot disable thinking; `xhigh` and `max` map to `high`). Thought summaries return in `data.reasoning` (`includeThoughts: true` when `reasoning_effort` is set; suppress via `extra: { config: { thinkingConfig: { includeThoughts: false } } }`). Assistant `reasoning` in message history is stripped before sending.
- **Tools & Thought Signatures**: Preserves Gemini 3 thought signatures in `toolCall.extra_content.google.thought_signature` across conversation turns (preserved automatically by `assistantMessage`). Missing call IDs are assigned `call_<uuid>`. Combining `tools` with `web_search: true` is supported on Gemini 3+ and rejected on Gemini 2.x.
- **Web Search Grounding**: `web_search: true` enables Google Search grounding; citations return in `data.sources` (`{ url, title? }`).
- **Batch Calls**: `batch: true` aggregates calls arriving within `batchWaitSec` into Gemini Batch API jobs (50% cost reduction; requires `vertex.bucket` / env `VERTEX_BATCH_BUCKET` on Vertex AI). Executes through the model's `batchQueue` bounded only by `modelLimits.concurrency`.
- **Embeddings**: Supports `task` (`'query'`, `'document'`, `'similarity'`, `'classification'`, `'clustering'`), `title` (with `task: 'document'`), and `dimensions`. Embeds up to `100` texts per call (`maxEmbedInputs: 100`) on the Developer API and for `gemini-embedding-001` on Vertex AI. On Vertex AI, `gemini-embedding-2` accepts `1` text per call; configure `capabilities: { embeddings: { supported: true, maxInputs: 1 } }` so `llm.embed` splits batches into single-text calls automatically. `usage.inputTokens` is `null` (unreported by the Gemini embedding API).
- **Extra Configuration (`extra.config`)**: Validates and deep-merges `extra: { config }` (`GenaiConfig`) across provider, model, and call layers. Other keys in `extra` are rejected.
- **Model Listing**: `llm.listModels('gemini')` returns `{ id, displayName?, inputTokenLimit?, outputTokenLimit? }[]`.

Full documentation of configuration, requests, rate limits, and error handling is available in the [smart-llm README](https://github.com/martin-winkler-dev/smart-llm-js#readme).
