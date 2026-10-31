# @smart-llm/openai-compatible

[![Test](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml)
[![Package](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml)
[![Bun](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fopenai-compatible%2Fpackage.json&query=%24.engines.bun&label=Bun&logo=bun&color=blue)](https://bun.sh)
[![Node.js](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fopenai-compatible%2Fpackage.json&query=%24.engines.node&label=Node.js&logo=nodedotjs&color=blue)](https://nodejs.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/martin-winkler-dev/smart-llm-js/blob/main/LICENSE)

Generic OpenAI-compatible provider adapter (`createOpenAiCompatibleProvider`, `fetchJson`, `statusError`) for [smart-llm](https://github.com/martin-winkler-dev/smart-llm-js#readme). Connects any `/chat/completions`, `/embeddings`, and `/models` API. No classes; post-initialization methods never throw.

## Install

```bash
bun add @smart-llm/openai-compatible @smart-llm/core
# or
npm install @smart-llm/openai-compatible @smart-llm/core
```

ESM package with TypeScript types for Node.js >= 22.12 (`require()` supported) and Bun >= 1.3.

## Quickstart

```ts
import { createLlm } from '@smart-llm/core';
import { createOpenAiCompatibleProvider } from '@smart-llm/openai-compatible';

const together = createOpenAiCompatibleProvider({
    name: 'together',
    baseUrl: 'https://api.together.xyz/v1',
    apiKey: process.env.TOGETHER_API_KEY,
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

## Options (`OpenAiCompatibleOptions`)

`createOpenAiCompatibleProvider(options)` creates an `LlmProvider` adapter:

| Option | Type | Default | Description |
|---|---|---|---|
| `name` | `string` | *required* | Provider name identifier |
| `baseUrl` | `string` | *required* | Base URL (e.g. `https://api.together.xyz/v1`; trailing slashes stripped) |
| `apiKey` | `string` | `undefined` | Optional Bearer token (`Authorization: Bearer <apiKey>`) |
| `headers` | `Record<string, string>` | `undefined` | Custom request headers merged over defaults |
| `supports` | `Omit<LlmProviderSupport, 'webSearch' \| 'batch'>` | `{}` | Supported capabilities (`images`, `tools`, `reasoning`, `json`, `embeddings`) |
| `maxTokensField` | `'max_tokens'` \| `'max_completion_tokens'` | `'max_completion_tokens'` | Request body field name for output token limits |
| `mapBody` | `(body, request) => body` | `undefined` | Optional hook to transform the outgoing `/chat/completions` body |
| `reasoningFields` | `readonly string[]` | `['reasoning_content', 'reasoning']` | Response message fields inspected for reasoning text |
| `reasoningMessageField` | `string` | `undefined` | Assistant message field used to send reasoning history back |
| `timeoutMs` | `number` | `120_000` | Request timeout in milliseconds (2 minutes) |
| `fetch` | `typeof fetch` | `globalThis.fetch` | Custom fetch implementation |

## Adapter Behavior & Exported Helpers

- **Message Normalization**: Converts `developer` role messages to `system`, strips `extra_content` from outgoing `tool_calls`, and forwards `extra` fields into the request body.
- **Tool Calls**: Assigns `call_<uuid>` IDs when omitted by the provider and validates that `function.arguments` is a JSON object (failing with a retryable `unexpected` error if malformed).
- **Usage & Finish Reasons**: Deducts `completion_tokens_details.reasoning_tokens` from `completion_tokens` so `usage.outputTokens` never includes reasoning tokens. Maps `finish_reason: 'length'` to `truncated` and `'content_filter'` to `blocked`.
- **Embeddings**: Sends `POST /embeddings` with `{ model, input: string[], dimensions? }`, sorts returned vectors by `index`, and reports `usage.prompt_tokens`. Rejects `task` and `title` with `unsupported`.
- **Model Listing**: `listModels()` queries `GET /models` and returns `{ id }[]`.
- **`fetchJson(fetcher, url, init)`**: Performs an HTTP request, parses JSON on 2xx responses, and maps HTTP status codes, `retry-after` headers (seconds or HTTP-date), `json_validate_failed` (attaching `failed_generation`), `tool_use_failed`, timeouts, aborts, and network errors to `Result<unknown, LlmError>`. Never throws.
- **`statusError(status, message, retryAfter)`**: Maps an HTTP status code and `retry-after` header value to an `LlmError` (`rate_limited` on 429; retryable `http` on 5xx, 408, 409, 425, 498; non-retryable `http` otherwise).

Full documentation of configuration, requests, rate limits, and error handling is available in the [smart-llm README](https://github.com/martin-winkler-dev/smart-llm-js#readme).
