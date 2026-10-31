# @smart-llm/ollama

[![Test](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml)
[![Package](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml)
[![Bun](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Follama%2Fpackage.json&query=%24.engines.bun&label=Bun&logo=bun&color=blue)](https://bun.sh)
[![Node.js](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Follama%2Fpackage.json&query=%24.engines.node&label=Node.js&logo=nodedotjs&color=blue)](https://nodejs.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/martin-winkler-dev/smart-llm-js/blob/main/LICENSE)

Local Ollama provider adapter (`createOllamaProvider`, `fetchOllamaCapabilities`, `OLLAMA_SUPPORT`) for [smart-llm](https://github.com/martin-winkler-dev/smart-llm-js#readme). No classes; post-initialization methods never throw.

## Install

```bash
bun add @smart-llm/ollama @smart-llm/core
# or
npm install @smart-llm/ollama @smart-llm/core
```

ESM package with TypeScript types for Node.js >= 22.12 (`require()` supported) and Bun >= 1.3.

## Quickstart

```ts
import { createLlm } from '@smart-llm/core';
import { createOllamaProvider, fetchOllamaCapabilities } from '@smart-llm/ollama';

const tags = await fetchOllamaCapabilities();
const qwenCaps = tags.ok ? tags.data['qwen2.5:7b'] : { tools: true };

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

### `createOllamaProvider(options?: OllamaOptions)`

| Option | Type | Default | Description |
|---|---|---|---|
| `baseUrl` | `string` | `'http://localhost:11434/v1'` | OpenAI-compatible Ollama endpoint URL |
| `apiKey` | `string` | `undefined` | Optional Bearer token for authenticated reverse proxies |
| `timeoutMs` | `number` | `600_000` | Request timeout in milliseconds (10 minutes) |
| `fetch` | `typeof fetch` | `globalThis.fetch` | Custom fetch implementation |

### `fetchOllamaCapabilities(options?: OllamaCapabilitiesOptions)`

Queries `GET /api/tags` on the Ollama server (stripping any trailing `/v1` from `baseUrl`) and maps reported model capability tags (`vision` -> `images: true`, `tools` -> `tools: true`, `thinking` -> `reasoning: { supported: true }`, `embedding` -> `embeddings: true`). Returns `Promise<Result<Record<string, LlmModelCapabilities>, LlmError>>` and never throws.

| Option | Type | Default | Description |
|---|---|---|---|
| `baseUrl` | `string` | `'http://localhost:11434/v1'` | Server endpoint URL (`/v1` suffix is stripped automatically) |
| `timeoutMs` | `number` | `30_000` | Request timeout in milliseconds (30 seconds) |
| `signal` | `AbortSignal` | `undefined` | Optional abort signal |
| `fetch` | `typeof fetch` | `globalThis.fetch` | Custom fetch implementation |

## Supported Capabilities (`OLLAMA_SUPPORT`)

```ts
const OLLAMA_SUPPORT = {
    images: true,
    tools: true,
    reasoning: true,
    json: 'grammar',
    embeddings: true,
};
```

- **Structured JSON (`grammar` mode)**: Passes the cleaned JSON Schema in `response_format` for grammar-constrained decoding and injects the schema with field `description`s into the system prompt. Automatically retries (`retry_immediate`) if output fails schema validation.
- **Tool Calls**: Ollama does not support `tool_choice`. `tool_choice: 'none'` removes `tools` from the request body; other `tool_choice` values are stripped.
- **Reasoning**: Extracts `message.reasoning` into `data.reasoning`. `reasoning_effort: 'none'` disables thinking (except on models like `gpt-oss` that cannot disable reasoning).
- **Embeddings**: Executes batched `/v1/embeddings` requests (default `maxInputs: 64`) and supports `dimensions`. `task` and `title` are not supported and return `unsupported`.
- **Model Listing**: `llm.listModels('ollama')` queries `GET /v1/models` and returns `{ id }[]`.

Full documentation of configuration, requests, rate limits, and error handling is available in the [smart-llm README](https://github.com/martin-winkler-dev/smart-llm-js#readme).
