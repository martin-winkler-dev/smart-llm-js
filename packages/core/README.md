# @smart-llm/core

[![Test](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml)
[![Package](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml)
[![Bun](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fcore%2Fpackage.json&query=%24.engines.bun&label=Bun&logo=bun&color=blue)](https://bun.sh)
[![Node.js](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fcore%2Fpackage.json&query=%24.engines.node&label=Node.js&logo=nodedotjs&color=blue)](https://nodejs.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/martin-winkler-dev/smart-llm-js/blob/main/LICENSE)

Core coordinator for [smart-llm](https://github.com/martin-winkler-dev/smart-llm-js#readme): unified OpenAI-form request routing across Gemini, DeepSeek, Groq, local Ollama models, and any OpenAI-compatible API, with named models, capability validation, JSON schema handling, and [smart-limit](https://www.npmjs.com/package/smart-limit) rate limits. No classes; post-initialization methods never throw.

| Package | Description |
|---|---|
| `@smart-llm/core` | Core coordinator (`createLlm`), message & tool helpers (`assistantMessage`, `toolMessage`), image helpers (`imagePart`, `imageFromFile`), JSON schema utilities (`cleanSchema`, `toStrictSchema`, `removeOptionalNulls`, `findSchemaMismatch`, `parseJsonText`, `schemaInstruction`), and TypeScript types |
| `@smart-llm/gemini` | Gemini adapter (`createGeminiProvider`, `GEMINI_SUPPORT`) for Developer API and Vertex AI via [smart-genai](https://www.npmjs.com/package/smart-genai) |
| `@smart-llm/deepseek` | DeepSeek adapter (`createDeepseekProvider`, `DEEPSEEK_SUPPORT`, `DEEPSEEK_API_KEY_ENV`) |
| `@smart-llm/groq` | Groq adapter (`createGroqProvider`, `GROQ_SUPPORT`, `GROQ_API_KEY_ENV`) |
| `@smart-llm/ollama` | Local Ollama adapter (`createOllamaProvider`, `fetchOllamaCapabilities`, `OLLAMA_SUPPORT`) |
| `@smart-llm/openai-compatible` | Generic OpenAI-compatible adapter (`createOpenAiCompatibleProvider`, `fetchJson`, `statusError`) for any `/chat/completions` and `/embeddings` endpoint |

```bash
bun add @smart-llm/core
# or
npm install @smart-llm/core
```

ESM package with TypeScript types for Node.js >= 22.12 (`require()` supported) and Bun >= 1.3.

## Configuration

```ts
import { createLlm, imageFromFile } from '@smart-llm/core';
import { createDeepseekProvider } from '@smart-llm/deepseek';
import { createGeminiProvider } from '@smart-llm/gemini';
import { createOllamaProvider } from '@smart-llm/ollama';

const llm = createLlm({
    defaults: { temperature: 0.2 },
    limiterOptions: { maxRetries: 2 },
    providers: {
        gemini: {
            adapter: createGeminiProvider(),              // env GEMINI_API_KEY, or createGeminiProvider({ vertex: true })
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
        deepseek: {
            adapter: createDeepseekProvider(),            // env DEEPSEEK_API_KEY
            sharedLimits: { concurrency: 10 },            // shared pool across all DeepSeek models
            models: { 'ds-chat': 'deepseek-chat' },       // shorthand string: provider model id, plain text only
        },
        ollama: {
            adapter: createOllamaProvider(),              // http://localhost:11434/v1
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

// Standard generate call
const result = await llm.generate({
    model: 'flash',
    messages: [{ role: 'user', content: 'Hi' }],
});

// Model member function: llm.<name>(input, options)
const same = await llm.qwen('Hi', { temperature: 0 });

// Content parts (text and image) with call options
const photo = await imageFromFile('photo.jpg');
const vision = await llm.flash([{ type: 'text', text: 'What is this?' }, photo], { signal });

// Bracket notation for non-identifier names
const bracket = await llm['ds-chat']('Explain gravity in one sentence');
```

### Core Architecture & Guarantees

- **Typed Model Routing**: Model names (`flash`, `ds-chat`, `qwen`) must be unique across all providers. `generate` and `embed` accept only configured model names.
- **Model Member Functions**: Each configured model is attached directly to the `llm` instance as a callable function: `llm.<name>(input, options)`.
  - `input`: A string or content parts array (`LlmContentPart[]`, text and images) as one `user` message, or an array of messages (`LlmMessage[]`) representing the conversation history.
  - `options`: Any request field except `model` and `messages` (`temperature`, `response_format`, `tools`, `reasoning_effort`, `batch`, etc.), plus call options `signal` and `extra`. Options set to `undefined` preserve configured defaults; `{}` changes nothing.
  - **Detached Execution**: The model function does not rely on `this` and can be passed directly as a callback: `run(llm.flash)` or `const classify = (input) => llm.flash(input, { response_format: word })`.
  - **Validation Errors**: An invalid input (e.g. empty or mixed array), or passing `model` or `messages` inside `options`, fails with `{ ok: false, error: { code: 'invalid_request' } }`.
- **Reserved Names**: The names `generate`, `embed`, `models`, `listModels`, `then`, `toJSON`, `__proto__`, and all properties of `Object.prototype` (`constructor`, `toString`, `valueOf`, etc.) cannot be used as model names (`createLlm` throws a `TypeError`). Model names with hyphens or non-identifier characters can be called via bracket notation (`llm['ds-chat']`).
- **Multiple Aliases**: The same provider model id can be registered under multiple names with different defaults, capabilities, or limits (e.g. `flash` for fast responses and `flash-think` for high reasoning).
- **Capability Isolation**: Adapters do not hardcode model capabilities. Each model declares what it supports in `capabilities`. An undeclared capability defaults to `false`, restricting the model to plain text.
- **Immutable Config & Settings**: Configuration objects, defaults, `extra`, and per-call `settings` are deep-cloned so external mutations never affect active or subsequent API calls.
- **Error Safety**: Only factory functions (`createLlm`, `create*Provider`) throw exceptions (a `TypeError` on invalid configurations or missing credentials). All post-initialization methods (`generate`, `embed`, `listModels`, and `llm.<name>()`) never throw or reject. Every invocation returns `{ ok: true, data }` or `{ ok: false, error }`.

### Layered Defaults & Extra

Request defaults (`defaults`) include sampling parameters and output limits: `temperature`, `top_p`, `max_tokens` / `max_completion_tokens`, `seed`, and `stop`. Defaults resolve hierarchically, with later layers overriding earlier layers:

`Global defaults -> Provider defaults -> Model defaults -> Call options`

`max_tokens` and `max_completion_tokens` represent the same setting; specifying either in a higher layer overrides any value from a lower layer, and they are normalized to `max_tokens`. Setting both within the same layer throws a `TypeError`.

Fields that are not sampling defaults must not be placed in `defaults` (doing so throws a `TypeError`):

| Field | Where to configure |
|---|---|
| `web_search` | Model capability default (`capabilities.webSearch.default`) or per call |
| `batch` | Model capability default (`capabilities.batch.default`) or per call |
| `reasoning_effort` | Model capability default (`capabilities.reasoning.default`) or per call |
| `response_format` | Per-call options |
| `tools`, `tool_choice` | Per-call options |

`extra` holds provider-specific settings and merges in the same priority order: provider `extra`, model `extra`, then call `extra`. By default, keys are shallow-merged (later keys win), unless the provider defines a custom `mergeExtra`. Adapters with `validateExtra` validate the configuration in `createLlm` (throwing a `TypeError` on invalid options) and call-level `extra` in `generate` (returning an `invalid_request` error).

### Limits & Queuing

Limits are managed via [smart-limit](https://github.com/martin-winkler-dev/smart-limit-js). `sharedLimits` on a provider apply across all models of that provider (e.g. an account-wide RPM or single-concurrency local server). `modelLimits` apply only to that specific model. When both are configured, a request must satisfy both sets of limits. Without explicit limits, requests run unthrottled.

| Limit Field | Description |
|---|---|
| `concurrency` | Maximum concurrent active requests (integer `>= 1`) |
| `rps`, `rpm`, `rpd` | Requests per second (`1_000` ms), minute (`60_000` ms), or day (`86_400_000` ms) rolling sliding windows |
| `tpm`, `tpd` | Tokens per minute or day rolling sliding windows |
| `intervalMs` | Minimum milliseconds between call starts (e.g. `3_000` = at most 1 call every 3 s) |
| `buckets` | Custom smart-limit buckets, e.g. `createSlidingWindow({ tokens: 50_000, perMs: 10_000 })` |

`limiterOptions` (top-level configuration applied to every call):

| Option | Default | Description |
|---|---|---|
| `maxRetries` | `4` | Retry attempts for timeouts and transient per-call errors |
| `retryDelayMs` | `2_000` | Initial exponential backoff delay in milliseconds |
| `maxRetryDelayMs` | `300_000` | Maximum backoff delay cap (5 minutes) |
| `maxWaitMs` | `900_000` | Maximum queue wait time before failing (15 minutes) |

`maxRetries` bounds timeouts and isolated retryable errors (`retry_backoff` and `retry_immediate`). Rate limits (HTTP 429), server errors (5xx), and network errors (`backoff_global`) retry with backoff until `maxWaitMs` is reached, even when `maxRetries: 0`. Sleeping retries sit in an unbounded buffer and never block free `modelLimits.concurrency` slots from starting other calls.

#### Token Booking & `charsPerToken`

`charsPerToken` calibrates the pre-call token estimate booked against token rate limits (`tpm`, `tpd`). It can be set globally, per provider, or per model (default: `4`, positive finite decimals allowed).

- **Generation calls (`generate`)**:
  `Math.ceil(characters / charsPerToken) + (images * 1_000) + (max_tokens ?? 1_000)`
  Characters are counted across message texts, assistant `reasoning`, `JSON.stringify(tools)`, assistant tool call names and arguments, and tool message contents. When the provider returns actual usage, booked tokens settle to `inputTokens + outputTokens + reasoningTokens`. On HTTP 429 errors without reported usage, booked tokens are refunded (`tokens: 0`).
- **Embedding calls (`embed`)**:
  Each batch chunk books `Math.ceil(characters / charsPerToken)` across input `text` and `title` strings, settling to `usage.inputTokens` when reported.

### Capabilities

| Capability | Values | Description |
|---|---|---|
| `images` | `boolean` | Image parts in `user` messages |
| `tools` | `boolean` | Function tools, tool calls, and tool messages in history |
| `json` | `boolean` \| `LlmJsonMode` | `true` (adapter default), `'native'`, `'strict'`, `'grammar'`, `'prompt'`, or `false` |
| `webSearch` | `{ supported, default? }` | Web search grounding; `default` (default `false`) applies when unset in call |
| `batch` | `{ supported, default? }` | Asynchronous batch execution; `default` (default `false`) applies when unset in call |
| `reasoning` | `{ supported, default?, levels? }` | Reasoning effort; `default` applies when unset; `levels` restricts supported effort levels |
| `embeddings` | `boolean` \| `{ supported, maxInputs? }` | Vector embeddings; `maxInputs` sets maximum texts per provider call |

When `reasoning.levels` is declared, requested efforts outside the list map to the nearest declared level in the order:

`none` < `minimal` < `low` < `medium` < `high` < `xhigh` < `max`

An effort other than `none` never maps to `none`. Ties resolve to the higher level. If `levels` does not include `none`, reasoning cannot be disabled, and `none` maps to the lowest declared level.

- `llm.models`: `readonly LlmModel[]` listing all configured models (`{ name, provider, model, capabilities }`) with resolved capabilities.
- `llm.listModels(provider, options?)`: Queries available models from the configured provider adapter (`options?: { signal?: AbortSignal }`), returning `Promise<Result<LlmModelInfo[], LlmError>>`.

## Exported Helpers & Utilities

### Message & Tool Helpers

- `assistantMessage(output)`: Formats `LlmOutput` or `LlmAnswer` as an `assistant` message, preserving `text` (or `null` when only `toolCalls` are present), `tool_calls` (including `extra_content`), and `reasoning`.
- `toolMessage(call, result)`: Formats a tool execution result as `{ role: 'tool', tool_call_id: call.id, content }` (strings passed as-is; other values serialized to JSON).

### Multimodal Image Helpers

- `imageFromFile(path, mimeType?)`: Reads an image file from disk (auto-detecting `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif` when `mimeType` is omitted) and returns an `LlmImagePart` (`{ type: 'image_url', image_url: { url } }`).
- `imagePart(data, mimeType, detail?)`: Encodes a `Uint8Array` or `ArrayBuffer` as a base64 `data:image/...;base64,...` `LlmImagePart` with optional `detail` (`'auto' | 'low' | 'high'`).

### JSON Schema Utilities

- `cleanSchema(schema)`: Returns a deep copy of `schema` with `$schema` and `$id` removed.
- `toStrictSchema(schema)`: Cleans `schema`, marks all object properties `required`, sets `additionalProperties: false`, and makes optional fields nullable.
- `removeOptionalNulls(value, schema)`: Recursively removes `null` values from optional object properties (resolving matching `anyOf` branches).
- `findSchemaMismatch(value, schema, path?)`: Validates `value` against `schema` (`type`, `enum`, `const`, `properties`, `required`, `additionalProperties`, `items`, `anyOf`); returns a path mismatch string or `undefined`.
- `parseJsonText(text)`: Parses JSON string (stripping surrounding markdown code fences if present); returns `{ value }` or `undefined`.
- `schemaInstruction(schema?)`: Builds the system prompt instruction text for `prompt` and `grammar` JSON modes.

### Custom Adapter Contract (`LlmProvider`)

- `name`: Provider name string.
- `supports`: `LlmProviderSupport` declaring supported features (`images`, `tools`, `webSearch`, `reasoning`, `batch`, `json`, `embeddings`).
- `generate(request, { signal, extra })`: Dispatches a single API request and maps responses to `Result<LlmProviderOutput, LlmProviderError>`. Must never throw or reject.
- `listModels?(options?)`: Queries available models (`Result<LlmModelInfo[], LlmError>`).
- `validateExtra?(extra, option)`: Validates provider-specific `extra` options; throws `TypeError` on invalid options.
- `mergeExtra?(layers)`: Custom merge logic for provider, model, and call `extra` objects.
- `embed?(request, { signal })`: Dispatches a single vector embedding request for up to `maxEmbedInputs` texts (`Result<LlmProviderEmbedOutput, LlmProviderError>`).
- `maxEmbedInputs?`: Maximum texts permitted in a single embedding call (positive integer; defaults to `64` when omitted).

Full documentation of requests, tool calls, batch execution, embeddings, structured JSON modes, responses, and error codes is available in the [smart-llm README](https://github.com/martin-winkler-dev/smart-llm-js#readme).
