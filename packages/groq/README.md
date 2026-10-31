# @smart-llm/groq

[![Test](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml)
[![Package](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml)
[![Bun](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fgroq%2Fpackage.json&query=%24.engines.bun&label=Bun&logo=bun&color=blue)](https://bun.sh)
[![Node.js](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fgroq%2Fpackage.json&query=%24.engines.node&label=Node.js&logo=nodedotjs&color=blue)](https://nodejs.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/martin-winkler-dev/smart-llm-js/blob/main/LICENSE)

Groq API provider adapter (`createGroqProvider`, `GROQ_SUPPORT`, `GROQ_API_KEY_ENV`) for [smart-llm](https://github.com/martin-winkler-dev/smart-llm-js#readme). No classes; post-initialization methods never throw.

## Install

```bash
bun add @smart-llm/groq @smart-llm/core
# or
npm install @smart-llm/groq @smart-llm/core
```

ESM package with TypeScript types for Node.js >= 22.12 (`require()` supported) and Bun >= 1.3.

## Quickstart

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

## Options (`GroqOptions`)

`createGroqProvider(options?)` configures the Groq adapter:

| Option | Type | Default | Description |
|---|---|---|---|
| `apiKey` | `string` | env `GROQ_API_KEY` | API key (`GROQ_API_KEY_ENV`); throws `TypeError` if missing |
| `env` | `Readonly<Record<string, string \| undefined>>` | `process.env` | Environment map used for API key lookup |
| `baseUrl` | `string` | `'https://api.groq.com/openai/v1'` | API base URL |
| `timeoutMs` | `number` | `120_000` | Request timeout in milliseconds (2 minutes) |
| `fetch` | `typeof fetch` | `globalThis.fetch` | Custom fetch implementation |

## Supported Capabilities (`GROQ_SUPPORT`)

```ts
const GROQ_SUPPORT = {
    images: true,
    tools: true,
    reasoning: true,
    json: 'strict',
};
```

- **Structured JSON (`strict` mode)**: Transforms schemas via `toStrictSchema` (marks all object properties `required`, makes optional properties nullable, sets `additionalProperties: false` and `strict: true`) and strips `null` values on optional properties from parsed output (`removeOptionalNulls`). Models that do not support strict schemas can declare `capabilities: { json: 'prompt' }`.
- **Reasoning**: Extracts `message.reasoning` into `data.reasoning` and deducts `reasoning_tokens` from `usage.outputTokens`. Configure model-specific `levels` in `capabilities.reasoning`:
  - `openai/gpt-oss-*`: `levels: ['low', 'medium', 'high']` (reasoning cannot be disabled).
  - `qwen/*`: `levels: ['none', 'low', 'medium', 'high']`; set `extra: { reasoning_format: 'parsed' }` to receive reasoning in `data.reasoning`.
  - `muse/*`: `levels: ['low', 'medium', 'high', 'xhigh']` (reasoning cannot be disabled).
- **Error Handling**: Treats HTTP 498 (Groq flex tier capacity) and `tool_use_failed` as retryable (`retry_backoff`). Maps `json_validate_failed` to `invalid_json` with `failed_generation` preserved in `error.output.text`.
- **Output Token Limit**: Maps `max_tokens` / `max_completion_tokens` to the `max_completion_tokens` request field.
- **Model Listing**: `llm.listModels('groq')` queries `GET /models` and returns `{ id }[]`.

Full documentation of configuration, requests, rate limits, and error handling is available in the [smart-llm README](https://github.com/martin-winkler-dev/smart-llm-js#readme).
