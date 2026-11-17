# @smart-llm/openai-compatible

[![Test](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml)
[![Package](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml)
[![Bun](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fopenai-compatible%2Fpackage.json&query=%24.engines.bun&label=Bun&logo=bun&color=blue)](https://bun.sh)
[![Node.js](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fopenai-compatible%2Fpackage.json&query=%24.engines.node&label=Node.js&logo=nodedotjs&color=blue)](https://nodejs.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Generic OpenAI-compatible adapter for [smart-llm](https://github.com/martin-winkler-dev/smart-llm-js#readme): connects any API with `/chat/completions`, `/embeddings`, and `/models` endpoints.

## Install

```bash
bun add @smart-llm/openai-compatible @smart-llm/core
# or
npm install @smart-llm/openai-compatible @smart-llm/core
```

ESM package with TypeScript types for Node.js >= 22.12 (`require()` supported) and Bun >= 1.3.

## Usage

```ts
import { createLlm } from '@smart-llm/core';
import { createOpenAiCompatibleProvider } from '@smart-llm/openai-compatible';

const apiKey = process.env.TOGETHER_API_KEY;
if (!apiKey) throw new Error('TOGETHER_API_KEY is not set');

const together = createOpenAiCompatibleProvider({
    name: 'together',
    baseUrl: 'https://api.together.xyz/v1',
    apiKey,
    supports: { images: true, tools: true, json: 'prompt', embeddings: true },
});

const llm = createLlm({
    providers: {
        together: {
            adapter: together,
            sharedLimits: { rpm: 60 },
            models: {
                llama: {
                    model: 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
                    capabilities: { tools: true, json: true },
                },
            },
        },
    },
});

const result = await llm.llama('List three benefits of immutable data structures.');
```

## Options

`createOpenAiCompatibleProvider(options: OpenAiCompatibleOptions)` snapshots `options` at creation and returns an `LlmProvider`. Options are not validated; unknown keys are ignored.

| Option | Type | Default | Description |
|---|---|---|---|
| `name` | `string` | *required* | Provider name |
| `baseUrl` | `string` | *required* | API base URL (e.g. `https://api.together.xyz/v1`); trailing slashes are stripped |
| `apiKey` | `string` | `undefined` | Sent as `authorization: Bearer <apiKey>`; when `undefined`, no `authorization` header is sent |
| `headers` | `Record<string, string>` | `undefined` | Headers for all endpoints; lowercase `content-type` / `authorization` keys override the defaults |
| `supports` | `Omit<LlmProviderSupport, 'webSearch' \| 'batch'>` | `{}` | Supported capabilities (`images`, `tools`, `reasoning`, `json`, `embeddings`) |
| `maxTokensField` | `'max_tokens'` \| `'max_completion_tokens'` | `'max_completion_tokens'` | Body field that receives the output token limit |
| `mapBody` | `(body, request) => body` | `undefined` | Transforms the `/chat/completions` body after `extra` is applied |
| `reasoningFields` | `readonly string[]` | `['reasoning_content', 'reasoning']` | Response message fields read as reasoning text (first non-empty string wins) |
| `reasoningMessageField` | `string` | `undefined` | Assistant message field that replays `reasoning` in the history; without it `reasoning` is not sent |
| `timeoutMs` | `number` | `120_000` | Timeout per `generate`, `embed`, and `listModels` call in milliseconds (2 minutes) |
| `fetch` | `typeof fetch` | `globalThis.fetch` | Custom fetch implementation |

## Extra Options

`extra` (provider, model, or call layer) is spread onto the `/chat/completions` body after the request fields and before `mapBody`. The adapter does not validate it.

## Capabilities

Capabilities come from `supports`; `webSearch` and `batch` are always removed.

- **Messages**: `developer` messages are sent as `system`; `max_tokens` / `max_completion_tokens` are sent as `maxTokensField`. `extra_content` and empty `tool_calls: []` are stripped from outgoing messages; empty tool call `arguments` are sent as `'{}'`; assistant `reasoning` is stripped unless `reasoningMessageField` is set.
- **Images**: via `supports.images`; image parts are sent unchanged.
- **Tools**: via `supports.tools`. Response tool calls without a function name are dropped; missing, `''`, or non-string `id` becomes `call_` plus 32 hex characters; missing, `null`, or `''` arguments become `'{}'`, non-string arguments are JSON-serialized. For `null`, `stop`, `end_turn`, or `tool_calls` finish reasons, arguments that are not a JSON object fail with retryable `unexpected`.
- **JSON**: via `supports.json` (`'native'`, `'strict'`, `'grammar'`, or `'prompt'`); `@smart-llm/core` builds `response_format` and validates the answer.
- **Reasoning**: via `supports.reasoning`; `reasoning_effort` is sent as-is. Reasoning text is read from `reasoningFields` (and replayed via `reasoningMessageField` when set). `completion_tokens_details.reasoning_tokens` is deducted from `completion_tokens` (clamped at `0`) into `usage.outputTokens` and reported as `usage.reasoningTokens`; negative, non-finite, or non-numeric counts become `null`.
- **Web search**: not supported; `web_search: true` returns `unsupported`.
- **Batch**: not supported; `batch: true` returns `unsupported`.
- **Embeddings**: via `supports.embeddings`. Sends `POST /embeddings` with `{ model, input: string[], dimensions? }` (chunked by `@smart-llm/core`, default `maxInputs: 64`), sorts vectors by `index` (falling back to array position), and reports `usage.prompt_tokens` as `usage.inputTokens` (`null` if missing/invalid). `task` and `title` return `unsupported`.
- **Model listing**: `listModels()` queries `GET /models` and returns `{ id }[]`.

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
| `unexpected` | yes | Finish reason `tool_calls` without a usable call, tool call arguments that are not a JSON object, or finish reason `insufficient_system_resource`; carries `error.output` |
| `unexpected` | no | 2xx body that is not JSON, response without a choice, other unknown finish reasons (carries `error.output`), an `/embeddings` response with a wrong vector count or structure, or an uncaught exception (e.g. throwing `mapBody` or cyclic `extra`) |
| `unsupported` | no | `web_search: true`, `batch: true`, `embed` without `supports.embeddings`, or embed `task` / `title` |

## Exports

| Export | Kind | Description |
|---|---|---|
| `createOpenAiCompatibleProvider` | function | Creates the adapter |
| `OpenAiCompatibleOptions` | type | Options of `createOpenAiCompatibleProvider` |
| `fetchJson` | function | `(fetcher, url, init) => Promise<Result<unknown, LlmError>>`; parses JSON on 2xx and maps HTTP, `retry-after`, `json_validate_failed` (`error.text` without `error.output`), `tool_use_failed`, `TimeoutError` (`timeout`), `AbortError` (`aborted`), and network errors; never throws |
| `statusError` | function | `(status, message, retryAfter) => LlmError`; maps HTTP status and `retry-after: string \| null` (`rate_limited` on 429; retryable `http` on 5xx, 408, 409, 425, 498; non-retryable `http` otherwise) |

Configuration, requests, rate limits, and error handling of `@smart-llm/core` are documented in the [smart-llm README](https://github.com/martin-winkler-dev/smart-llm-js#readme).

## License

Apache-2.0, see [LICENSE](LICENSE).
