# @smart-llm/deepseek

[![Test](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml)
[![Package](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml)
[![Bun](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fdeepseek%2Fpackage.json&query=%24.engines.bun&label=Bun&logo=bun&color=blue)](https://bun.sh)
[![Node.js](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fdeepseek%2Fpackage.json&query=%24.engines.node&label=Node.js&logo=nodedotjs&color=blue)](https://nodejs.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

DeepSeek adapter for [smart-llm](https://github.com/martin-winkler-dev/smart-llm-js#readme): connects the DeepSeek chat completions API.

## Install

```bash
bun add @smart-llm/deepseek @smart-llm/core
# or
npm install @smart-llm/deepseek @smart-llm/core
```

ESM package with TypeScript types for Node.js >= 22.12 (`require()` supported) and Bun >= 1.3.

## Usage

```ts
import { createLlm } from '@smart-llm/core';
import { createDeepseekProvider } from '@smart-llm/deepseek';

const llm = createLlm({
    providers: {
        deepseek: {
            adapter: createDeepseekProvider(),            // uses env DEEPSEEK_API_KEY
            sharedLimits: { concurrency: 10 },
            models: {
                flash: {
                    model: 'deepseek-flash',
                    capabilities: { tools: true, json: true, reasoning: { supported: true, default: 'none' } },
                },
                pro: {
                    model: 'deepseek-pro',
                    capabilities: { tools: true, json: true, reasoning: { supported: true } },
                },
            },
        },
    },
});

const result = await llm.flash('Explain database indexing in one paragraph.');
```

## Options

`createDeepseekProvider(options?: DeepseekOptions)` returns an `LlmProvider`; options and `DEEPSEEK_SUPPORT` are snapshotted at creation.

| Option | Type | Default | Description |
|---|---|---|---|
| `apiKey` | `string` | env `DEEPSEEK_API_KEY` | API key (`DEEPSEEK_API_KEY_ENV`); throws `TypeError` if missing or `''` |
| `env` | `Readonly<Record<string, string \| undefined>>` | `process.env` | Environment map for the API key lookup |
| `baseUrl` | `string` | `'https://api.deepseek.com'` | API base URL (trailing slashes are stripped) |
| `timeoutMs` | `number` | `600_000` | Timeout per `generate` and `listModels` call in milliseconds (10 minutes) |
| `fetch` | `typeof fetch` | `globalThis.fetch` | Custom fetch implementation |

## Extra Options

`extra` (provider, model, or call layer) is spread onto the `/chat/completions` body after request fields and can overwrite them. The adapter does not validate it.

## Capabilities

`DEEPSEEK_SUPPORT`: `{ images: true, tools: true, reasoning: true, json: 'prompt' }`.

- **Messages**: `developer` messages are sent as `system`; `max_tokens` / `max_completion_tokens` are sent as `max_tokens`.
- **Images**: supported; image parts are sent unchanged.
- **Tools**: supported; `tool_choice` is sent as given. Pass assistant turns through `assistantMessage(output)` from `@smart-llm/core` so their `reasoning` is replayed (see Reasoning).
- **JSON** (`prompt`): sends `response_format: { type: 'json_object' }`; `@smart-llm/core` inserts a `system` JSON instruction (with cleaned schema when given) after leading `system`/`developer` messages, skips validation when tool calls are returned, and validates the answer (`invalid_json` without retry).
- **Reasoning**: supported; `reasoning_effort` is sent as-is. `reasoning_content` is returned in `data.reasoning` (`message.reasoning` is ignored); `reasoning_tokens` is deducted from `completion_tokens` (clamped at `0`) to compute `usage.outputTokens` and reported as `usage.reasoningTokens`. Assistant `reasoning` in the history is sent back as `reasoning_content` on each assistant message that has it.
- **Web search**: not supported; `web_search: true` returns `unsupported`.
- **Batch**: not supported; `batch: true` returns `unsupported`.
- **Embeddings**: not supported; `embed` returns `unsupported`.
- **Model listing**: `llm.listModels('deepseek')` queries `GET /models` and returns `{ id }[]`.

## Errors

| Code | Retryable | When |
|---|---|---|
| `rate_limited` | yes | HTTP 429; `retry-after` (seconds or HTTP date) sets `retryAfterMs` |
| `http` | yes | HTTP 5xx, 408, 409, 425, or 498 |
| `http` | no | Other non-2xx statuses |
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
| `createDeepseekProvider` | function | Creates the adapter |
| `DEEPSEEK_SUPPORT` | constant | `LlmProviderSupport` of the adapter |
| `DEEPSEEK_API_KEY_ENV` | constant | `'DEEPSEEK_API_KEY'` |
| `DeepseekOptions` | type | Options of `createDeepseekProvider` |

Configuration, requests, rate limits, and error handling of `@smart-llm/core` are documented in the [smart-llm README](https://github.com/martin-winkler-dev/smart-llm-js#readme).

## License

Apache-2.0, see [LICENSE](LICENSE).
