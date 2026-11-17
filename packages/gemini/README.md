# @smart-llm/gemini

[![Test](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml)
[![Package](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml)
[![Bun](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fgemini%2Fpackage.json&query=%24.engines.bun&label=Bun&logo=bun&color=blue)](https://bun.sh)
[![Node.js](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fgemini%2Fpackage.json&query=%24.engines.node&label=Node.js&logo=nodedotjs&color=blue)](https://nodejs.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Gemini adapter for [smart-llm](https://github.com/martin-winkler-dev/smart-llm-js#readme): connects the Gemini Developer API (AI Studio) and Vertex AI through [smart-genai](https://www.npmjs.com/package/smart-genai).

## Install

```bash
bun add @smart-llm/gemini @smart-llm/core
# or
npm install @smart-llm/gemini @smart-llm/core
```

ESM package with TypeScript types for Node.js >= 22.12 (`require()` supported) and Bun >= 1.3.

## Usage

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

`createGeminiProvider(clientOrOptions?: GenaiClient | GeminiOptions)` accepts an existing `smart-genai` client or `GeminiOptions` (`Omit<GenaiClientOptions, 'defaults'>`). Passing `defaults` or setting both `apiKey` and `vertex` throws `TypeError`; set request defaults in `createLlm`. A passed client created with `defaults` keeps applying them.

| Option | Type | Default | Description |
|---|---|---|---|
| `apiKey` | `string` | env `GEMINI_API_KEY` | Developer API key (`GEMINI_API_KEY_ENV`); Developer API only |
| `vertex` | `boolean` \| `GenaiVertexOptions` | `undefined` | `true` or `{ project?, location?, bucket?, credentialsFile?, authClient?, storageUrl? }` for Vertex AI |
| `baseUrl` | `string` | official endpoint | Custom API endpoint URL |
| `env` | `Readonly<Record<string, string \| undefined>>` | `process.env` | Environment map for the API key and the Vertex AI project, location, bucket, and credentials file |
| `timeoutMs` | `number` | `120_000` | HTTP timeout per call in milliseconds (2 minutes) |
| `batchWaitSec` | `number` | `10` | Collection window in seconds for batch calls (`1` to `600`) |

Vertex AI also reads `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` (default `'global'`), `VERTEX_BATCH_BUCKET`, and `GOOGLE_APPLICATION_CREDENTIALS` (else gcloud ADC). Missing or invalid credentials throw `TypeError`.

## Extra Options

`extra` (provider, model, or call layer) accepts only `GeminiExtra`:

| Key | Type | Description |
|---|---|---|
| `config` | `GenerateContentConfig` | Gemini request configuration passed to smart-genai |

- **Merge**: `config` merges across provider, model, and call layers per top-level field (later layers win; nested objects such as `toolConfig` are replaced whole). `thinkingConfig` merges per member; `thinkingBudget` and `thinkingLevel` count as one setting, so the last layer that sets either replaces both from earlier layers.
- **Validation**: other `extra` keys, a non-object `config`, own `__proto__` keys, a `config.tools` that is neither an array nor `null`, and `abortSignal` / `httpOptions` / `automaticContinuation` inside `config` are rejected (`TypeError` at setup, `invalid_request` per call).

## Capabilities

`GEMINI_SUPPORT`: `{ images: true, tools: true, webSearch: true, reasoning: true, batch: true, json: 'native', embeddings: true }`.

- **Messages**: `stop` takes up to 5 sequences and `seed` must be a 32-bit signed integer (`invalid_request` otherwise).
- **Images**: base64 data URLs; `detail` is not sent.
- **Tools**: missing call IDs become `call_` plus 32 hex characters. Thought signatures are kept in `toolCall.extra_content.google.thought_signature` and replayed on the next turn (`assistantMessage` keeps them). `strict` on tool schemas is not sent; `tool_choice` is stripped by `@smart-llm/core` when `tools` is omitted (even if `extra.config.tools` declares functions). `tools` and `web_search: true` append after `extra.config.tools` and fail with `invalid_request` on `gemini-2.*` models.
- **JSON** (`native`): sends the cleaned JSON Schema as `responseJsonSchema`. On Vertex AI batch, `required` is stripped and `responseSchema` only supports the typed-schema subset (unsupported keywords fail with `invalid_request`). `@smart-llm/core` validates the parsed answer.
- **Reasoning**: maps `reasoning_effort` (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`):
  - `gemini-2.*` models: `thinkingBudget` (`0` for `none`; at least `128` on `gemini-2.5-pro`)
  - Gemma models: `thinkingLevel` `MINIMAL` for `none` to `low`, `HIGH` for `medium` to `max`
  - other models: `thinkingLevel`; `none` and `minimal` send `MINIMAL` on Gemini 3.0 to 3.6 Flash and Flash-Lite and `LOW` on others; `xhigh` and `max` send `HIGH`
  - thought summaries return in `data.reasoning`: `includeThoughts: true` is sent whenever `reasoning_effort` is set, except when `none` sends `thinkingBudget: 0`; turn off with `extra: { config: { thinkingConfig: { includeThoughts: false } } }`
  - assistant `reasoning` in the history is not sent
- **Web search**: `web_search: true` enables Google Search grounding; citations return in `data.sources` (`{ url, title? }`).
- **Batch**: `batch: true` groups calls that arrive within `batchWaitSec` and share `model`, `config`, `response_format`, and `tools` into one Gemini Batch API job; Vertex AI needs `vertex.bucket` or env `VERTEX_BATCH_BUCKET`. Jobs are polled every 30 s for up to 72 hours, then fail with non-retryable `timeout`. A group above the payload limit is split into several jobs; a single request above it fails with `invalid_request`.
- **Embeddings**: `task` (`'query'`, `'document'`, `'similarity'`, `'classification'`, `'clustering'`) maps to Gemini task types; per-input `title` requires `task: 'document'`; `dimensions` is supported. `usage.inputTokens` is always `null`.
  - up to `100` texts per call (`maxEmbedInputs: 100`) on the Developer API, and on Vertex AI for `gemini-embedding-001` and non-Gemini (non-MaaS) models
  - Vertex AI with other Gemini models (including `gemini-embedding-2`) and MaaS models: more than `1` text per call returns `unsupported`; set `capabilities: { embeddings: { supported: true, maxInputs: 1 } }` so `llm.embed` sends one text per call
  - models other than `gemini-embedding-2` take one `title` per call; different titles fail with `invalid_request`
- **Model listing**: `llm.listModels('gemini')` returns `{ id, displayName?, inputTokenLimit?, outputTokenLimit? }[]` (Vertex AI publisher models populate `id` only).

## Errors

| Code | Retryable | When |
|---|---|---|
| `rate_limited` | yes | HTTP 429 (`retry-after` honored) |
| `http` | yes / no | HTTP errors (`status` and `retryAfterMs` preserved) |
| `network` | yes | Connection failure |
| `timeout` | yes | Request timed out |
| `timeout` | no | Batch job not finished after 72 hours |
| `aborted` | no | Caller `signal` aborted |
| `blocked` / `truncated` | no | Safety block / token limit; carries `error.output` |
| `invalid_request` | no | Invalid `extra.config`, request limits under Capabilities, `tools` with `web_search: true` on `gemini-2.*`, oversized batch request, or unsupported Vertex batch schema keywords |
| `unsupported` | no | Multi-text `embed` on Vertex AI for models limited to one text, or `batch: true` on Vertex AI without `vertex.bucket` / `VERTEX_BATCH_BUCKET` |
| `unexpected` | yes | Malformed function call (`MALFORMED_FUNCTION_CALL`) |
| `unexpected` | no | Missing candidate, failed batch job, other tool finish errors, or non-`TypeError` exceptions |

An `invalid_json` answer is parsed and validated by `@smart-llm/core` (or forwarded as `invalid_json` if no output was returned).

## Exports

| Export | Kind | Description |
|---|---|---|
| `createGeminiProvider` | function | Creates the adapter |
| `GEMINI_SUPPORT` | constant | `LlmProviderSupport` of the adapter |
| `GEMINI_API_KEY_ENV` | constant | `'GEMINI_API_KEY'` (re-exported from smart-genai) |
| `GeminiOptions` | type | Options of `createGeminiProvider` |
| `GeminiExtra` | type | `extra` accepted by the adapter |

Configuration, requests, rate limits, and error handling of `@smart-llm/core` are documented in the [smart-llm README](https://github.com/martin-winkler-dev/smart-llm-js#readme).

## License

Apache-2.0, see [LICENSE](LICENSE).
