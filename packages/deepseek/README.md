# @smart-llm/deepseek

[![Test](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml)
[![Package](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml)
[![Bun](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fdeepseek%2Fpackage.json&query=%24.engines.bun&label=Bun&logo=bun&color=blue)](https://bun.sh)
[![Node.js](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fdeepseek%2Fpackage.json&query=%24.engines.node&label=Node.js&logo=nodedotjs&color=blue)](https://nodejs.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/martin-winkler-dev/smart-llm-js/blob/main/LICENSE)

DeepSeek API provider adapter (`createDeepseekProvider`, `DEEPSEEK_SUPPORT`, `DEEPSEEK_API_KEY_ENV`) for [smart-llm](https://github.com/martin-winkler-dev/smart-llm-js#readme). No classes; post-initialization methods never throw.

## Install

```bash
bun add @smart-llm/deepseek @smart-llm/core
# or
npm install @smart-llm/deepseek @smart-llm/core
```

ESM package with TypeScript types for Node.js >= 22.12 (`require()` supported) and Bun >= 1.3.

## Quickstart

```ts
import { createLlm } from '@smart-llm/core';
import { createDeepseekProvider } from '@smart-llm/deepseek';

const llm = createLlm({
    providers: {
        deepseek: {
            adapter: createDeepseekProvider(),            // uses env DEEPSEEK_API_KEY
            sharedLimits: { concurrency: 10 },
            models: {
                chat: {
                    model: 'deepseek-chat',
                    capabilities: { tools: true, json: true, reasoning: { supported: true, default: 'none' } },
                },
                reasoner: {
                    model: 'deepseek-reasoner',
                    capabilities: { tools: true, json: true, reasoning: { supported: true } },
                },
            },
        },
    },
});

const result = await llm.chat('Explain database indexing in one paragraph.');
```

## Options (`DeepseekOptions`)

`createDeepseekProvider(options?)` configures the DeepSeek adapter:

| Option | Type | Default | Description |
|---|---|---|---|
| `apiKey` | `string` | env `DEEPSEEK_API_KEY` | API key (`DEEPSEEK_API_KEY_ENV`); throws `TypeError` if missing |
| `env` | `Readonly<Record<string, string \| undefined>>` | `process.env` | Environment map used for API key lookup |
| `baseUrl` | `string` | `'https://api.deepseek.com'` | API base URL |
| `timeoutMs` | `number` | `600_000` | Request timeout in milliseconds (10 minutes) |
| `fetch` | `typeof fetch` | `globalThis.fetch` | Custom fetch implementation |

## Supported Capabilities (`DEEPSEEK_SUPPORT`)

```ts
const DEEPSEEK_SUPPORT = {
    images: true,
    tools: true,
    reasoning: true,
    json: 'prompt',
};
```

- **Reasoning**: Extracts `reasoning_content` into `data.reasoning` and deducts `reasoning_tokens` from `usage.outputTokens`. Preserves assistant `reasoning` as `reasoning_content` across multi-turn tool call histories (handled automatically by `assistantMessage`). Without `reasoning_effort`, DeepSeek thinking models default to `high`; `reasoning_effort: 'none'` disables thinking.
- **Tool Calls**: In thinking mode, DeepSeek requires `reasoning_content` on prior assistant tool turns and rejects `tool_choice: 'required'` or named function choices with HTTP 400.
- **Structured JSON (`prompt` mode)**: Sends `response_format: { type: 'json_object' }`, injects the JSON Schema instruction into the system prompt, validates the parsed response, and automatically retries (`retry_immediate`) if output fails JSON parsing or schema validation.
- **Output Token Limit**: Maps `max_tokens` / `max_completion_tokens` to the `max_tokens` request field.
- **Model Listing**: `llm.listModels('deepseek')` queries `GET /models` and returns `{ id }[]`.

Full documentation of configuration, requests, rate limits, and error handling is available in the [smart-llm README](https://github.com/martin-winkler-dev/smart-llm-js#readme).
