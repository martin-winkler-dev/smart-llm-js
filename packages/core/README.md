# @smart-llm/core

[![Test](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml)
[![Package](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml)
[![Bun](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fcore%2Fpackage.json&query=%24.engines.bun&label=Bun&logo=bun&color=blue)](https://bun.sh)
[![Node.js](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fcore%2Fpackage.json&query=%24.engines.node&label=Node.js&logo=nodedotjs&color=blue)](https://nodejs.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Core coordinator for [smart-llm](https://github.com/martin-winkler-dev/smart-llm-js#readme): routes OpenAI-form requests to provider adapters by configured model name, with capability validation, JSON schema handling, and [smart-limit](https://www.npmjs.com/package/smart-limit) rate limits.

## Install

```bash
bun add @smart-llm/core
# or
npm install @smart-llm/core
```

ESM package with TypeScript types for Node.js >= 22.12 (`require()` supported) and Bun >= 1.3. Add one or more adapters: `@smart-llm/anthropic`, `@smart-llm/gemini`, `@smart-llm/deepseek`, `@smart-llm/groq`, `@smart-llm/ollama`, or `@smart-llm/openai-compatible`.

## Usage

```ts
import { createLlm, imageFromFile } from '@smart-llm/core';
import { createGeminiProvider } from '@smart-llm/gemini';
import { createOllamaProvider } from '@smart-llm/ollama';

const llm = createLlm({
    defaults: { temperature: 0.2 },
    limiterOptions: { maxRetries: 2 },
    providers: {
        gemini: {
            adapter: createGeminiProvider(),
            models: {
                flash: {
                    model: 'gemini-3.8-flash',
                    capabilities: { images: true, json: true, reasoning: { supported: true } },
                    modelLimits: { rpm: 1_000, tpm: 1_000_000 },
                },
                'flash-think': {
                    model: 'gemini-3.8-flash',
                    capabilities: { images: true, json: true, reasoning: { supported: true, default: 'high' } },
                },
            },
        },
        ollama: {
            adapter: createOllamaProvider(),
            sharedLimits: { concurrency: 1 },
            charsPerToken: 3.5,
            models: {
                qwen: {
                    model: 'qwen2.5:7b',
                    capabilities: { tools: true, json: true },
                },
            },
        },
    },
});

const result = await llm.generate({ model: 'flash', messages: [{ role: 'user', content: 'Hi' }] });
const same = await llm.qwen('Hi', { temperature: 0 });
const photo = await imageFromFile('photo.jpg');
const vision = await llm.flash([{ type: 'text', text: 'What is this?' }, photo], { signal: AbortSignal.timeout(60_000) });
```

- **Model routing**: model names (`flash`, `qwen`) must be unique across all providers; `generate` and `embed` accept only configured names.
- **Model functions**: each model is exposed as `llm.<name>(input, options?)`, bound without `this` (e.g. `run(llm.flash)`).
  - `input`: a `string`, `LlmContentPart[]` (text and images as one `user` message), or `LlmMessage[]`
  - `options`: any request field except `model` and `messages`, plus `signal` and `extra`; `undefined` keeps defaults
- **Reserved names**: `generate`, `embed`, `models`, `listModels`, `then`, `toJSON`, `__proto__`, and `Object.prototype` keys cannot be model names (`createLlm` throws `TypeError`). Non-identifier names work with bracket notation (`llm['flash-think']`).
- **Results**: `createLlm`, `imagePart`, and `imageFromFile` throw `TypeError` on invalid input (`imageFromFile` also rejects on filesystem errors). `generate`, `embed`, `listModels`, and `llm.<name>()` never throw or reject; they return `{ ok: true, data }` or `{ ok: false, error }`.
- **Strict validation**: unknown keys in config, requests, messages (including OpenAI's `name` and `refusal` or DeepSeek's `prefix`), content parts, `image_url`, `tools`, `response_format`, `embed` requests/inputs/options, `listModels` options, and circular references fail (`TypeError` in `createLlm`, `invalid_request` per call). `extra` objects reject own `'__proto__'` properties (`TypeError` in `createLlm`, `invalid_request` per call). Send provider-specific fields through the adapter (e.g. `extra` or `mapBody` in `@smart-llm/openai-compatible`).

## Configuration

### Defaults and Extra

Sampling defaults (`temperature`, `top_p`, `max_tokens` / `max_completion_tokens`, `seed`, `stop`) resolve in order:

`Global defaults -> Provider defaults -> Model defaults -> Call options`

`max_tokens` and `max_completion_tokens` normalize to `max_tokens` (integer `>= 1`). Both in the same `defaults` layer throw `TypeError`; both in one call return `invalid_request`. Non-sampling fields (`web_search`, `batch`, `reasoning_effort`, `response_format`, `tools`, `tool_choice`) in `defaults` throw `TypeError`.

`extra` merges across `Provider -> Model -> Call` (shallow merge, or `adapter.mergeExtra` when defined). `undefined` keys keep earlier values; `null` is forwarded. Adapters with `validateExtra` check configured `extra` at setup (`TypeError`) and call `extra` per call (`invalid_request`).

### Limits and Queuing

`sharedLimits` apply across all models of a provider; `modelLimits` apply to one model; with both, a request must satisfy both. Each model has separate `generate`, `embed`, and batch queues; `modelLimits.concurrency` applies to each queue, `sharedLimits` apply to the `generate` and `embed` queues. Unknown limit keys throw `TypeError`.

| Limit | Description |
|---|---|
| `concurrency` | Maximum concurrent active requests (integer `>= 1`) |
| `rps`, `rpm`, `rpd` | Requests per second, minute, or day, rolling windows (integer `>= 1`) |
| `tpm`, `tpd` | Tokens per minute or day, rolling windows (integer `>= 1`) |
| `intervalMs` | Minimum milliseconds between call starts (integer `>= 1`) |
| `buckets` | Custom `smart-limit` buckets |

`limiterOptions` (integers `>= 0`):

| Option | Default | Description |
|---|---|---|
| `maxRetries` | `4` | Retry attempts for timeouts and isolated retryable errors |
| `retryDelayMs` | `2_000` | Initial exponential backoff delay (ms) |
| `maxRetryDelayMs` | `300_000` | Maximum backoff delay (ms) |
| `maxWaitMs` | `900_000` | Maximum queue wait before failing (ms) |

- **Retries**: only errors with `retryable: true` are retried. Retryable `rate_limited`, `network`, and `http` 5xx errors lock the queue until `retry-after` (or backoff, at least `1_000` ms) elapses, then send one probe request before queued calls resume; they retry until `maxWaitMs` regardless of `maxRetries`. Other retryable errors (e.g. `timeout`) count against `maxRetries`.
- **Wait limit**: calls exceeding `maxWaitMs` fail with `limiter` (`limiterCode: 'MAX_WAIT_EXCEEDED'`), or with the last provider error if already attempted.
- **Token estimate**: `charsPerToken` (global, provider, or model; default `4`) sets the booked estimate. `generate` books `Math.ceil(characters / charsPerToken) + (images * 1_000) + (max_tokens ?? 1_000)` and settles to `inputTokens + outputTokens + reasoningTokens` when usage is reported (`null` counts as `0`); `rate_limited` errors without usage refund to `0`, other errors without usage keep the estimate. `embed` books `Math.ceil(characters / charsPerToken)` per chunk and settles to `usage.inputTokens`.
- **Batch queue**: `batch: true` calls are bounded only by `modelLimits.concurrency` and `limiterOptions`, skipping token booking, `sharedLimits`, non-concurrency `modelLimits`, and queue locks; every retryable error counts against `maxRetries`.

### Capabilities

| Capability | Values | Description |
|---|---|---|
| `images` | `boolean` | Image parts in `user` messages |
| `tools` | `boolean` | Function tools, tool calls, and tool messages |
| `json` | `boolean` \| `LlmJsonMode` | `true` (adapter default), `'native'`, `'strict'`, `'grammar'`, `'prompt'`, or `false` |
| `webSearch` | `{ supported, default? }` | Web search grounding (`default` defaults to `false`) |
| `batch` | `{ supported, default? }` | Asynchronous batch execution (`default` defaults to `false`) |
| `reasoning` | `{ supported, default?, levels? }` | Reasoning effort; `default` and `levels` require `supported: true`, `default` must be in `levels`, and `levels` must contain at least one effort other than `'none'` |
| `embeddings` | `boolean` \| `{ supported, maxInputs? }` | Vector embeddings (`maxInputs` integer `>= 1`, at most `adapter.maxEmbedInputs` when set; default `adapter.maxEmbedInputs` or `64`) |

- **Undeclared**: capabilities default to `false` (plain text only). Declaring a capability or `json` mode unsupported by `adapter.supports` throws `TypeError`.
- **Unsupported requests**: requesting a capability the model lacks returns `unsupported`; disabling one (`web_search: false`, `batch: false`, and `reasoning_effort: 'none'` on models without `reasoning`) is stripped.
- **Effort mapping**: with `reasoning.levels`, other efforts map to the nearest declared level along `none < minimal < low < medium < high < xhigh < max` (ties round up; non-`none` never maps to `none`; `none` maps to the lowest level when `none` is not declared).
- **Discovery**: `llm.models` lists configured models (`readonly LlmModel[]`) with resolved capabilities. `llm.listModels(provider, options?)` queries the configured `provider` key (`options?: { signal?: AbortSignal }`) and returns `Promise<Result<LlmModelInfo[], LlmError>>`.

## Requests

`llm.generate(request, options?)` takes call options `{ signal?, extra? }` (`LlmCallOptions`) and a request with these fields:

| Field | Description |
|---|---|
| `model` | Configured model name |
| `messages` | `system`, `developer`, `user`, `assistant`, `tool`; at least one `user` message; content is a string or a non-empty array of parts; `tool` content is a string or non-empty `LlmTextPart[]`; assistant `content` may be `null` with non-empty `tool_calls` (`arguments` `''` or JSON object text); `tool_call_id` must match a tool call of an earlier assistant message; images only in `user` messages as `data:image/...;base64,...` URLs (optional `detail?: 'auto' \| 'low' \| 'high'`) |
| `response_format` | `{ type: 'text' }`, `{ type: 'json_object' }`, or `{ type: 'json_schema', json_schema: { name, schema?, description?, strict? } }`; `name` matches `^[A-Za-z0-9_-]{1,64}$` |
| `temperature`, `top_p`, `seed` | Finite numbers; ranges are checked by the provider |
| `max_tokens` / `max_completion_tokens` | Output token limit |
| `stop` | Stop sequence(s) |
| `reasoning_effort` | `'none'`, `'minimal'`, `'low'`, `'medium'`, `'high'`, `'xhigh'`, `'max'` |
| `web_search`, `batch` | Booleans |
| `tools` | Non-empty, unique names matching `^[A-Za-z0-9_-]{1,64}$` |
| `tool_choice` | `'auto'`, `'none'`, `'required'`, or `{ type: 'function', function: { name } }` (must match a declared tool); dropped without `tools` |

`llm.embed(request, options?)` takes `{ model, input, task?, dimensions? }` (`input`: a non-empty string or a non-empty `(string | { text, title? })[]` with non-empty texts; `task`: `'query' | 'document' | 'similarity' | 'classification' | 'clustering'`; `title` requires `task: 'document'`; `dimensions`: integer `>= 1`) and `options?: { signal?: AbortSignal }`. Inputs are split into chunks of `maxInputs`; when one chunk fails, the other chunks of the request are aborted.

### JSON Modes

| Mode | Request |
|---|---|
| `native` | Sends `cleanSchema(schema)` in `response_format` |
| `strict` | Sends `toStrictSchema(schema)` with `strict: true`; removes optional `null`s with `removeOptionalNulls` |
| `grammar` | Inserts `schemaInstruction(cleanSchema(schema))` as a `system` message after leading `system`/`developer` messages and sends `cleanSchema(schema)` |
| `prompt` | Inserts `schemaInstruction(cleanSchema(schema))` as a `system` message after leading `system`/`developer` messages and sends `{ type: 'json_object' }` |

`{ type: 'json_object' }` and `json_schema` without `schema` send `{ type: 'json_object' }` in every mode (`grammar` and `prompt` also insert `schemaInstruction(undefined)`) and require a JSON object; arrays or primitives fail with `invalid_json`. With returned `toolCalls`, JSON parsing and validation are skipped. Invalid JSON or schema mismatches fail with `invalid_json` without retry and carry `error.output.text` (and `error.output.json` when the text parsed).

## Responses

`generate` returns `LlmOutput`: `{ text, json?, reasoning?, sources?, toolCalls?, name, provider, model, usage: { inputTokens, outputTokens, reasoningTokens }, finishReason, ms, settings, attempts }`. `outputTokens` never includes reasoning tokens.

`embed` returns `{ embeddings, name, provider, model, usage: { inputTokens: number | null }, ms, calls, attempts }`.

## Errors

`LlmError`: `{ code, message, retryable, output?, settings?, attempts?, status?, retryAfterMs?, limiterCode?, text?, usage?, model?, ms?, name?, provider? }`. `settings` is absent on `invalid_request`, `unsupported`, `embed`, `listModels`, and pre-settings `unexpected` errors; `attempts` is absent on `invalid_request`, `unsupported`, `listModels`, and pre-queue `embed` errors.

| Code | Retryable | When |
|---|---|---|
| `rate_limited` | yes | HTTP 429; locks the queue until `retry-after` or backoff |
| `timeout` | per adapter | Request timed out |
| `network` | yes | Connection failure; locks the queue |
| `http` | per adapter | HTTP error; 5xx lock the queue when retryable |
| `aborted` | no | Caller `signal` aborted |
| `blocked` | no | Output blocked by a content filter; carries `output` |
| `truncated` | no | Output token limit reached; carries `output` |
| `invalid_json` | no | Answer is not valid JSON or violates the schema; carries `output` |
| `unsupported` | no | Capability or `listModels` not available |
| `invalid_request` | no | Invalid request, options, model or provider name, or `extra` |
| `limiter` | no | smart-limit rejected the call before dispatch (`limiterCode`: `COST_EXCEEDS_LIMIT`, `MAX_WAIT_EXCEEDED`, `STOPPED`, or `BUCKET_FAILED`) |
| `unexpected` | per adapter | Malformed response, adapter exception, or adapter result that breaks the contract |

## Helpers

| Export | Description |
|---|---|
| `assistantMessage(output)` | Builds an `assistant` message from `LlmOutput` or `LlmAnswer`; keeps `tool_calls` and `reasoning`; `content` is `null` when `text === ''` and `toolCalls` is non-empty |
| `toolMessage(call, result)` | Builds a `tool` message; strings as-is, other values JSON-serialized (`BigInt` as decimal string, top-level `undefined` / function / symbol as `'null'`, circular refs as `"[Circular]"`); falls back to `String(result)`, then `[unserializable tool result]`; never throws |
| `imageFromFile(path, mimeType?)` | Reads `.png`, `.jpg`, `.jpeg`, `.webp`, or `.gif` (any case) into `LlmImagePart`; other extensions reject with `TypeError` unless `mimeType` is given |
| `imagePart(data, mimeType, detail?)` | Encodes a non-empty `Uint8Array` or `ArrayBuffer` with an `image/<subtype>` MIME type (lowercased) and optional `detail` (`'auto' \| 'low' \| 'high'`); throws `TypeError` on invalid input |
| `cleanSchema(schema)` | Deep copy with `$schema` and `$id` removed from all sub-schemas |
| `toStrictSchema(schema)` | Cleans `schema`; object nodes with `properties` (under `properties`, `items`, `anyOf`, `$defs`, `definitions`) get all properties `required`, `additionalProperties: false`, and nullable optional fields; `allOf`, `oneOf`, `not`, `prefixItems`, tuple `items`, and object `additionalProperties` subtrees are only cleaned |
| `removeOptionalNulls(value, schema)` | Removes `null` values of declared, non-required object properties (following `items`, matching `anyOf` branches, and local `$ref`s) |
| `findSchemaMismatch(value, schema, path?)` | Validates `type`, `enum`, `const`, `properties`, `required`, `additionalProperties: false`, `items`, `anyOf`, and local `$ref`s; returns a mismatch path string or `undefined` |
| `parseJsonText(text)` | Parses JSON, stripping ```` ``` ```` / ```` ```json ```` fences; returns `{ value }` or `undefined` |
| `schemaInstruction(schema)` | System prompt text for `prompt` and `grammar` modes (`schema: LlmJsonSchema \| undefined`) |

## Adapter Contract

Custom adapters implement `LlmProvider`:

| Member | Description |
|---|---|
| `name` | Provider name |
| `supports` | `LlmProviderSupport` (`images`, `tools`, `webSearch`, `reasoning`, `batch`, `json`, `embeddings`); boolean flags are `true` or omitted (`false` throws `TypeError`), `json` is `LlmJsonMode` or omitted |
| `generate(request, { signal, extra })` | Returns `Promise<Result<LlmProviderOutput, LlmProviderError>>`; must not throw or reject (throws map to `unexpected`). An output needs string `text` and `model`, a `usage` object, and numeric `ms`; an error needs string `code` and `message` and boolean `retryable`. Every `usage` field must be a finite number `>= 0` or `null`; otherwise the call fails as `unexpected` |
| `listModels?(options?)` | Returns `Promise<Result<LlmModelInfo[], LlmError>>` |
| `validateExtra?(extra, option)` | Validates `extra`; throws `TypeError` on invalid options |
| `mergeExtra?(layers)` | Custom merge of provider, model, and call `extra` |
| `embed?(request, { signal })` | Required when `supports.embeddings` is `true`; receives `{ model, input: { text, title? }[], task?, dimensions? }` and returns `Promise<Result<LlmProviderEmbedOutput, LlmProviderError>>` with one vector per input; `usage.inputTokens` must be a finite number `>= 0` or `null` |
| `maxEmbedInputs?` | Maximum texts per `embed` call (positive integer; default `64`) |

## Exported Types

- Configuration: `LlmConfig`, `LlmProviderSpec`, `LlmModelSpec`, `LlmDefaults`, `LlmLimits`, `LlmLimiterOptions`, `LlmCapabilities`, `LlmModelCapabilities`, `LlmJsonMode`, `LlmReasoningEffort`.
- Client: `Llm`, `LlmModel`, `LlmModelNames`, `LlmModelCall`, `LlmModelInput`, `LlmModelOptions`, `LlmCallOptions`, `LlmReservedName`, `LlmModelInfo`.
- Requests: `LlmRequest`, `LlmMessage`, `LlmRole`, `LlmContentPart`, `LlmTextPart`, `LlmImagePart`, `LlmTool`, `LlmToolCall`, `LlmToolChoice`, `LlmResponseFormat`, `LlmJsonSchema`, `LlmEmbedRequest`, `LlmEmbedInput`, `LlmEmbedTask`.
- Results: `Result`, `LlmOutput`, `LlmAnswer`, `LlmUsage`, `LlmSettings`, `LlmSource`, `LlmEmbedOutput`, `LlmError`, `LlmErrorCode`.
- Adapter contract: `LlmProvider`, `LlmProviderSupport`, `LlmProviderRequest`, `LlmProviderOutput`, `LlmProviderError`, `LlmProviderEmbedRequest`, `LlmProviderEmbedOutput`.

## License

Apache-2.0, see [LICENSE](LICENSE).
