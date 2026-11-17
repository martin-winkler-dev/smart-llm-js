# @smart-llm/ollama

[![Test](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml)
[![Package](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml)
[![Bun](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Follama%2Fpackage.json&query=%24.engines.bun&label=Bun&logo=bun&color=blue)](https://bun.sh)
[![Node.js](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Follama%2Fpackage.json&query=%24.engines.node&label=Node.js&logo=nodedotjs&color=blue)](https://nodejs.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Ollama adapter for [smart-llm](https://github.com/martin-winkler-dev/smart-llm-js#readme): connects a local or remote Ollama server through its OpenAI-compatible endpoint and reads model capabilities from `/api/tags`.

## Install

```bash
bun add @smart-llm/ollama @smart-llm/core
# or
npm install @smart-llm/ollama @smart-llm/core
```

ESM package with TypeScript types for Node.js >= 22.12 (`require()` supported) and Bun >= 1.3.

## Usage

```ts
import { createLlm } from '@smart-llm/core';
import { createOllamaProvider, fetchOllamaCapabilities } from '@smart-llm/ollama';

const tags = await fetchOllamaCapabilities();
const qwenCaps = tags.ok ? (tags.data['qwen2.5:7b'] ?? { tools: true }) : { tools: true };

const llm = createLlm({
    providers: {
        ollama: {
            adapter: createOllamaProvider(),              // http://localhost:11434/v1
            sharedLimits: { concurrency: 1 },
            charsPerToken: 3.5,
            models: {
                qwen: {
                    model: 'qwen2.5:7b',
                    capabilities: { ...qwenCaps, json: true },
                },
                bge: {
                    model: 'bge-m3',
                    capabilities: { embeddings: true },
                },
            },
        },
    },
});

const answer = await llm.qwen('Explain TCP three-way handshake.');
const vectors = await llm.embed({ model: 'bge', input: ['First text', 'Second text'] });
```

## Options

`createOllamaProvider(options?: OllamaOptions)` returns an `LlmProvider`; options and `OLLAMA_SUPPORT` are snapshotted at creation.

| Option | Type | Default | Description |
|---|---|---|---|
| `baseUrl` | `string` | `'http://localhost:11434/v1'` | OpenAI-compatible Ollama endpoint URL |
| `apiKey` | `string` | `undefined` | Bearer token for authenticated reverse proxies |
| `timeoutMs` | `number` | `600_000` | Timeout per `generate`, `embed`, and `listModels` call in milliseconds (10 minutes) |
| `fetch` | `typeof fetch` | `globalThis.fetch` | Custom fetch implementation |

## Extra Options

`extra` (provider, model, or call layer) is spread onto the `/chat/completions` body. The adapter does not validate it.

## Capabilities

`OLLAMA_SUPPORT`: `{ images: true, tools: true, reasoning: true, json: 'grammar', embeddings: true }`.

- **Messages**: `developer` messages are sent as `system`; `max_tokens` / `max_completion_tokens` are sent as `max_tokens`.
- **Images**: supported; image parts are sent unchanged.
- **Tools**: supported. `tool_choice` is never sent: `tool_choice: 'none'` removes `tools` from the body, other values are dropped.
- **JSON** (`grammar`): sends the cleaned JSON Schema in `response_format` (`json_object` without a schema); `@smart-llm/core` inserts a `system` JSON instruction (with field `description`s when a schema is given) after leading `system`/`developer` messages and validates the answer (`invalid_json` without retry).
- **Reasoning**: supported; `reasoning_effort` is sent as-is (`'none'` disables thinking on models that support it). `message.reasoning` is returned in `data.reasoning` (`reasoning_content` is not read); if `reasoning_tokens` is reported, it is deducted from `completion_tokens` (clamped at `0`) into `usage.outputTokens` and `usage.reasoningTokens`. Assistant `reasoning` in the history is not sent.
- **Web search**: not supported; `web_search: true` returns `unsupported`.
- **Batch**: not supported; `batch: true` returns `unsupported`.
- **Embeddings**: supported. Sends `POST <baseUrl>/embeddings` with `dimensions` when set; `@smart-llm/core` splits inputs into chunks of `maxInputs` (default `64`). `task` and `title` return `unsupported`.
- **Model listing**: `llm.listModels('ollama')` queries `GET <baseUrl>/models` and returns `{ id }[]`.

## Errors

| Code | Retryable | When |
|---|---|---|
| `rate_limited` | yes | HTTP 429; `retry-after` (seconds or HTTP date) sets `retryAfterMs` |
| `http` | yes | HTTP 5xx, 408, 409, 425, or 498 |
| `http` | no | Other non-2xx statuses |
| `timeout` | yes | `timeoutMs` elapsed |
| `aborted` | no | Caller `signal` aborted |
| `network` | yes | `fetch` failed, e.g. the server is not running |
| `truncated` | no | Finish reason `length`; carries `error.output` |
| `blocked` | no | Finish reason `content_filter`; carries `error.output` |
| `unexpected` | yes | Finish reason `insufficient_system_resource`, finish reason `tool_calls` without a usable call, or tool call arguments that are not a JSON object; carries `error.output` |
| `unexpected` | no | Malformed response or unknown finish reason (unknown finish reasons carry `error.output`); from `fetchOllamaCapabilities`, a `/api/tags` body that is not a model list |
| `unsupported` | no | `web_search: true`, `batch: true`, or embed `task` / `title` |

## Exports

| Export | Kind | Description |
|---|---|---|
| `createOllamaProvider` | function | Creates the adapter |
| `fetchOllamaCapabilities` | function | Reads model capabilities from the server (see below) |
| `OLLAMA_SUPPORT` | constant | `LlmProviderSupport` of the adapter |
| `OllamaOptions` | type | Options of `createOllamaProvider` |
| `OllamaCapabilitiesOptions` | type | Options of `fetchOllamaCapabilities` |

### `fetchOllamaCapabilities(options?: OllamaCapabilitiesOptions)`

Queries `GET /api/tags` (a trailing `/v1` and trailing slashes on `baseUrl` are stripped, so the `createOllamaProvider` URL works) and returns `Promise<Result<Record<string, LlmModelCapabilities>, LlmError>>`. Never throws.

- capability tags: `vision` -> `images: true`, `tools` -> `tools: true`, `thinking` -> `reasoning: { supported: true }`, `embedding` -> `embeddings: true`; `json` is never set (model keys include tags such as `bge-m3:latest`)
- `null` body or missing/`null` `models`: `{}`
- other non-array `models` or an entry without a string `name`: `unexpected`
- entry without a `capabilities` array: that model's entry maps to `{}`

| Option | Type | Default | Description |
|---|---|---|---|
| `baseUrl` | `string` | `'http://localhost:11434/v1'` | Server endpoint URL |
| `apiKey` | `string` | `undefined` | Bearer token for authenticated reverse proxies |
| `timeoutMs` | `number` | `30_000` | Timeout in milliseconds (30 seconds); expiry returns retryable `timeout` |
| `signal` | `AbortSignal` | `undefined` | Abort signal; aborting returns `aborted` |
| `fetch` | `typeof fetch` | `globalThis.fetch` | Custom fetch implementation |

Configuration, requests, rate limits, and error handling of `@smart-llm/core` are documented in the [smart-llm README](https://github.com/martin-winkler-dev/smart-llm-js#readme).

## License

Apache-2.0, see [LICENSE](LICENSE).
