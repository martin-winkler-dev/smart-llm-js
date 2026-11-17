# @smart-llm/groq

[![Test](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml)
[![Package](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml)
[![Bun](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fgroq%2Fpackage.json&query=%24.engines.bun&label=Bun&logo=bun&color=blue)](https://bun.sh)
[![Node.js](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fgroq%2Fpackage.json&query=%24.engines.node&label=Node.js&logo=nodedotjs&color=blue)](https://nodejs.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Groq adapter for [smart-llm](https://github.com/martin-winkler-dev/smart-llm-js#readme): connects the Groq chat completions API.

## Install

```bash
bun add @smart-llm/groq @smart-llm/core
# or
npm install @smart-llm/groq @smart-llm/core
```

ESM package with TypeScript types for Node.js >= 22.12 (`require()` supported) and Bun >= 1.3.

## Usage

```ts
import { createLlm } from '@smart-llm/core';
import { createGroqProvider } from '@smart-llm/groq';

const llm = createLlm({
    providers: {
        groq: {
            adapter: createGroqProvider(),                // uses env GROQ_API_KEY
            models: {
                oss: {
                    model: 'openai/gpt-oss-120b',
                    capabilities: {
                        tools: true,
                        json: true,
                        reasoning: { supported: true, levels: ['low', 'medium', 'high'], default: 'medium' },
                    },
                    modelLimits: { rpm: 30, tpm: 200_000 },
                },
            },
        },
    },
});

const result = await llm.oss('Compare B-trees and LSM-trees in three bullet points.');
```

## Options

`createGroqProvider(options?: GroqOptions)` returns an `LlmProvider`; options and `GROQ_SUPPORT` are snapshotted at creation.

| Option | Type | Default | Description |
|---|---|---|---|
| `apiKey` | `string` | env `GROQ_API_KEY` | API key (`GROQ_API_KEY_ENV`); throws `TypeError` if missing or `''` |
| `env` | `Readonly<Record<string, string \| undefined>>` | `process.env` | Environment map for the API key lookup |
| `baseUrl` | `string` | `'https://api.groq.com/openai/v1'` | API base URL |
| `timeoutMs` | `number` | `120_000` | Timeout per `generate` and `listModels` call in milliseconds (2 minutes) |
| `fetch` | `typeof fetch` | `globalThis.fetch` | Custom fetch implementation |

## Extra Options

`extra` (provider, model, or call layer) is spread onto the `/chat/completions` body after request fields (e.g. `extra: { reasoning_format: 'parsed' }` to return parsed reasoning on models that default to raw/hidden). The adapter does not validate it.

## Capabilities

`GROQ_SUPPORT`: `{ images: true, tools: true, reasoning: true, json: 'strict' }`.

- **Messages**: `developer` messages are sent as `system`; `max_tokens` / `max_completion_tokens` are sent as `max_completion_tokens`.
- **Images**: supported; image parts are sent unchanged.
- **Tools**: supported; `tool_choice` is sent as given.
- **JSON** (`strict`): for `json_schema` with a `schema`, converts it with `toStrictSchema` from `@smart-llm/core`, sends `strict: true`, and removes `null` values of optional properties (`removeOptionalNulls`); `json_object` and schemaless `json_schema` send `response_format: { type: 'json_object' }`. `@smart-llm/core` validates the answer (`invalid_json` without retry). For models without strict schema support, declare `capabilities: { json: 'prompt' }`.
- **Reasoning**: supported; `reasoning_effort` is sent as-is. `message.reasoning` is returned in `data.reasoning` (`reasoning_content` is ignored); `reasoning_tokens` is deducted from `completion_tokens` (clamped at `0`) to compute `usage.outputTokens` and reported as `usage.reasoningTokens`. Assistant `reasoning` in the history is not sent. Declare the efforts a model accepts in `capabilities.reasoning.levels`.
- **Web search**: not supported; `web_search: true` returns `unsupported`.
- **Batch**: not supported; `batch: true` returns `unsupported`.
- **Embeddings**: not supported; `embed` returns `unsupported`.
- **Model listing**: `llm.listModels('groq')` queries `GET /models` and returns `{ id }[]`.

## Errors

| Code | Retryable | When |
|---|---|---|
| `rate_limited` | yes | HTTP 429; `retry-after` (seconds or HTTP date) sets `retryAfterMs` |
| `http` | yes | HTTP 5xx, 408, 409, 425, or 498; error body `error.code === 'tool_use_failed'` |
| `http` | no | Other non-2xx statuses |
| `invalid_json` | no | Error body `error.code === 'json_validate_failed'`; when `error.failed_generation` is a string, it is set on `error.text` and `error.output.text` |
| `timeout` | yes | `timeoutMs` elapsed |
| `aborted` | no | Caller `signal` aborted |
| `network` | yes | `fetch` failed |
| `truncated` | no | Finish reason `length`; carries `error.output` |
| `blocked` | no | Finish reason `content_filter`; carries `error.output` |
| `unexpected` | yes | Finish reason `insufficient_system_resource`, finish reason `tool_calls` without a usable call, or tool call arguments that are not a JSON object; carries `error.output` |
| `unexpected` | no | Malformed response or unknown finish reason (unknown finish reasons carry `error.output`) |
| `unsupported` | no | `web_search: true`, `batch: true`, or `embed` |

## Exports

| Export | Kind | Description |
|---|---|---|
| `createGroqProvider` | function | Creates the adapter |
| `GROQ_SUPPORT` | constant | `LlmProviderSupport` of the adapter |
| `GROQ_API_KEY_ENV` | constant | `'GROQ_API_KEY'` |
| `GroqOptions` | type | Options of `createGroqProvider` |

Configuration, requests, rate limits, and error handling of `@smart-llm/core` are documented in the [smart-llm README](https://github.com/martin-winkler-dev/smart-llm-js#readme).

## License

Apache-2.0, see [LICENSE](LICENSE).
