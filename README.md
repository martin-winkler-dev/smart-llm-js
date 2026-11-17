# smart-llm

[![Test](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml)
[![Package](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml)
[![Bun](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackage.json&query=%24.engines.bun&label=Bun&logo=bun&color=blue)](https://bun.sh)
[![Node.js](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackage.json&query=%24.engines.node&label=Node.js&logo=nodedotjs&color=blue)](https://nodejs.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Unified multi-provider client for Anthropic, Gemini, DeepSeek, Groq, local Ollama models, and any OpenAI-compatible API. Configures providers, models, capabilities, and [smart-limit](https://www.npmjs.com/package/smart-limit) rate limits in one place and routes calls by configured model name.

## Packages

| Package | Description |
|---|---|
| [`@smart-llm/core`](packages/core#readme) | Coordinator (`createLlm`), message and tool helpers (`assistantMessage`, `toolMessage`), image helpers (`imagePart`, `imageFromFile`), JSON schema utilities (`cleanSchema`, `toStrictSchema`, `removeOptionalNulls`, `findSchemaMismatch`, `parseJsonText`, `schemaInstruction`), and TypeScript types |
| [`@smart-llm/anthropic`](packages/anthropic#readme) | Anthropic Claude adapter (`createAnthropicProvider`, `ANTHROPIC_SUPPORT`, `ANTHROPIC_API_KEY_ENV`, `DEFAULT_MAX_TOKENS`) via `@anthropic-ai/sdk` |
| [`@smart-llm/gemini`](packages/gemini#readme) | Gemini adapter (`createGeminiProvider`, `GEMINI_SUPPORT`, `GEMINI_API_KEY_ENV`) for the Developer API and Vertex AI via [smart-genai](https://www.npmjs.com/package/smart-genai) |
| [`@smart-llm/deepseek`](packages/deepseek#readme) | DeepSeek adapter (`createDeepseekProvider`, `DEEPSEEK_SUPPORT`, `DEEPSEEK_API_KEY_ENV`) |
| [`@smart-llm/groq`](packages/groq#readme) | Groq adapter (`createGroqProvider`, `GROQ_SUPPORT`, `GROQ_API_KEY_ENV`) |
| [`@smart-llm/ollama`](packages/ollama#readme) | Ollama adapter (`createOllamaProvider`, `fetchOllamaCapabilities`, `OLLAMA_SUPPORT`) |
| [`@smart-llm/openai-compatible`](packages/openai-compatible#readme) | Generic OpenAI-compatible adapter (`createOpenAiCompatibleProvider`, `fetchJson`, `statusError`) for `/chat/completions`, `/embeddings`, and `/models` endpoints |

## Install

```bash
bun add @smart-llm/core @smart-llm/anthropic @smart-llm/gemini @smart-llm/deepseek @smart-llm/groq @smart-llm/ollama @smart-llm/openai-compatible
# or
npm install @smart-llm/core @smart-llm/anthropic @smart-llm/gemini @smart-llm/deepseek @smart-llm/groq @smart-llm/ollama @smart-llm/openai-compatible
```

Install `@smart-llm/core` plus the adapters you use. ESM packages with TypeScript types for Node.js >= 22.12 (`require()` supported) and Bun >= 1.3.

## Quickstart

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
            models: { 'ds-flash': 'deepseek-flash' },     // shorthand string: provider model id, plain text only
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

// Model function: llm.<name>(input, options)
const same = await llm.qwen('Hi', { temperature: 0 });

// Content parts (text and image) with call options
const photo = await imageFromFile('photo.jpg');
const vision = await llm.flash([{ type: 'text', text: 'What is this?' }, photo], { signal: AbortSignal.timeout(60_000) });

// Bracket notation for non-identifier names
const bracket = await llm['ds-flash']('Explain gravity in one sentence');
```

## Core Concepts

### Model Calls

- **Model routing**: model names (`flash`, `ds-flash`, `qwen`) must be unique across all providers. `generate` and `embed` accept only configured model names.
- **Model functions**: each configured model is a function on the `llm` instance: `llm.<name>(input, options)`.
  - `input`: a string or content parts array (`LlmContentPart[]`, text and images) as one `user` message, or a message array (`LlmMessage[]`) as the conversation history
  - `options`: any request field except `model` and `messages` (`temperature`, `response_format`, `tools`, `reasoning_effort`, `batch`, etc.), plus call options `signal` and `extra`; options set to `undefined` keep configured defaults, `{}` changes nothing
  - the function does not use `this` and can be passed as a callback: `run(llm.flash)` or `const classify = (input) => llm.flash(input, { response_format: word })`
  - an invalid input (e.g. empty or mixed array), or `model` / `messages` inside `options`, fails with `invalid_request`
- **Reserved names**: `generate`, `embed`, `models`, `listModels`, `then`, `toJSON`, `__proto__`, and all properties of `Object.prototype` (`constructor`, `toString`, `valueOf`, etc.) cannot be model names (`createLlm` throws `TypeError`). Names with hyphens or other non-identifier characters work with bracket notation (`llm['ds-flash']`).
- **Aliases**: the same provider model id can be registered under several names with different defaults, capabilities, or limits (e.g. `flash` and `flash-think`).
- **Immutable config and settings**: configuration objects, defaults, `extra`, and per-call `settings` are deep-cloned, so external mutations never affect running or later calls. Each call also deep-clones the request fields (`messages`, `tools`, `stop`, `response_format` with its `schema`) and the `extra` layers, so the caller can change its objects while a call waits in the queue, and an adapter that mutates its input cannot change stored defaults. Only plain objects and arrays are cloned; other values (strings such as image data URLs, class instances) are passed by reference.
- **Strict validation**: request, message, content part, and `image_url` keys are validated; unknown keys (including OpenAI's `name` and `refusal` or DeepSeek's `prefix` on messages) fail with `invalid_request`. Send provider-specific fields through the adapter (e.g. `extra` or `mapBody` for openai-compatible). `response_format.json_schema.name` must match `^[A-Za-z0-9_-]{1,64}$`. Request fields and `extra` with circular references fail with `invalid_request`.
- **Error safety**: `createLlm`, `create*Provider`, `imagePart`, and `imageFromFile` throw `TypeError` on invalid configuration, credentials, or image input (`imageFromFile` also rejects on filesystem errors). `generate`, `embed`, `listModels`, and `llm.<name>()` never throw or reject; every call returns `{ ok: true, data }` or `{ ok: false, error }`.

### Defaults and Extra

Request defaults (`defaults`) hold sampling parameters and output limits: `temperature`, `top_p`, `max_tokens` / `max_completion_tokens`, `seed`, and `stop`. Later layers override earlier ones:

`Global defaults -> Provider defaults -> Model defaults -> Call options`

`max_tokens` and `max_completion_tokens` are the same setting and normalize to `max_tokens`; either one in a later layer overrides both from earlier layers. Both in the same `defaults` layer throw `TypeError`; both in one call return `invalid_request`.

Other fields in `defaults` throw `TypeError`:

| Field | Where to configure |
|---|---|
| `web_search` | Model capability default (`capabilities.webSearch.default`) or per call |
| `batch` | Model capability default (`capabilities.batch.default`) or per call |
| `reasoning_effort` | Model capability default (`capabilities.reasoning.default`) or per call |
| `response_format` | Per call |
| `tools`, `tool_choice` | Per call |

`extra` holds provider-specific settings and merges in the same order: provider `extra`, model `extra`, then call `extra`. Keys are shallow-merged (later keys win) unless the adapter defines `mergeExtra`. A key set to `undefined` keeps the value of an earlier layer; with the default shallow merge, `null` is forwarded as JSON `null` (adapters with custom `mergeExtra`, such as Anthropic `extra.params`, may handle `null` differently). Adapters with `validateExtra` check configured `extra` in `createLlm` (`TypeError`) and call `extra` in `generate` (`invalid_request`). The `extra` keys of each adapter are listed under [Providers](#providers).

### Limits and Queuing

Limits use [smart-limit](https://github.com/martin-winkler-dev/smart-limit-js). `sharedLimits` on a provider apply across all its models (e.g. an account-wide RPM or a single-concurrency local server); `modelLimits` apply to one model. With both, a request must satisfy both. Without limits, requests run unthrottled.

```ts
import { createLlm } from '@smart-llm/core';
import { createDeepseekProvider } from '@smart-llm/deepseek';

const llm = createLlm({
    providers: {
        deepseek: {
            adapter: createDeepseekProvider(),
            sharedLimits: { concurrency: 10 },                                               // 10 concurrent requests across all models
            models: {
                'ds-flash': { model: 'deepseek-flash', modelLimits: { intervalMs: 3_000 } }, // at least 3 s between call starts
                'ds-pro': 'deepseek-pro',
            },
        },
    },
});
```

| Limit | Description |
|---|---|
| `concurrency` | Maximum concurrent active requests (integer `>= 1`) |
| `rps`, `rpm`, `rpd` | Requests per second (`1_000` ms), minute (`60_000` ms), or day (`86_400_000` ms), rolling sliding windows (integer `>= 1`) |
| `tpm`, `tpd` | Tokens per minute or day, rolling sliding windows (integer `>= 1`) |
| `intervalMs` | Minimum milliseconds between call starts (integer `>= 1`; e.g. `3_000` = at most 1 call every 3 s) |
| `buckets` | Custom smart-limit buckets, e.g. `createSlidingWindow({ tokens: 50_000, perMs: 10_000 })` |

Unknown limit keys throw `TypeError`. Each model has separate queues for `generate`, `embed`, and batch calls; `modelLimits.concurrency` applies to each queue separately. `sharedLimits` and `modelLimits` rate/token buckets (`rps`, `rpm`, `rpd`, `tpm`, `tpd`, `intervalMs`, `buckets`) are shared across a model's `generate` and `embed` queues, not applied to batch calls.

`limiterOptions` (top-level, applied to every call; integers `>= 0`):

| Option | Default | Description |
|---|---|---|
| `maxRetries` | `4` | Retry attempts for timeouts and other isolated retryable errors |
| `retryDelayMs` | `2_000` | Initial exponential backoff delay in milliseconds |
| `maxRetryDelayMs` | `300_000` | Maximum backoff delay (5 minutes) |
| `maxWaitMs` | `900_000` | Maximum queue wait before failing (15 minutes) |

Only errors with `retryable: true` are retried. `maxRetries` bounds timeouts and other isolated retryable errors (`retry_backoff`); these also stop once their cumulative backoff exceeds `maxWaitMs`. Rate limits (`rate_limited`), server errors (`http` 5xx), and network errors (`backoff_global`) retry with backoff until `maxWaitMs`, even with `maxRetries: 0`. Batch calls treat every retryable error as `retry_backoff`. Sleeping retries wait in an unbounded buffer and never block free `modelLimits.concurrency` slots.

#### Rate Limit Locks and Probing

Each model queue is FIFO. A retryable HTTP 429, 5xx, or network failure locks that queue (`backoff_global`); the other queues of the model stay open. When the `retry-after` duration (or backoff delay, at least `1_000` ms) elapses, the queue sends one probe request, and queued requests resume only after it succeeds. Other models, including those of the same provider, continue unless they fail themselves.

If the estimated wait for bucket capacity or a lock exceeds `maxWaitMs`, all waiting requests in that queue end at once: requests that already called the provider return their last provider error (e.g. `rate_limited` with `retryAfterMs`), unstarted requests return `limiter` with `limiterCode: 'MAX_WAIT_EXCEEDED'`. Calls started while the lock exceeds `maxWaitMs` fail without a network request.

#### Token Booking and `charsPerToken`

`charsPerToken` calibrates the pre-call token estimate booked against `tpm` and `tpd`. Set it globally, per provider, or per model (default `4`; positive finite decimals allowed).

- **`generate`**: books `Math.ceil(characters / charsPerToken) + (images * 1_000) + (max_tokens ?? 1_000)`. Characters count message texts (including the schema instruction of the `prompt` and `grammar` JSON modes), assistant `reasoning`, `JSON.stringify(tools)`, assistant tool call names and arguments, and tool message contents. When the provider reports usage (on a result or an error), the booking settles to `inputTokens + outputTokens + reasoningTokens` (`null` counts as `0`). `rate_limited` errors without usage refund the booking (`tokens: 0`); other errors without usage keep the estimate.
- **`embed`**: each chunk books `Math.ceil(characters / charsPerToken)` over input `text` and `title` strings and settles to `usage.inputTokens` when reported.

### Capabilities

Adapters declare the features their API supports (`adapter.supports`); models declare which of them they use in `capabilities`. Undeclared features default to `false`; a model without `capabilities` accepts and returns plain text only.

```ts
import type { LlmModelSpec } from '@smart-llm/core';

const models: Record<string, LlmModelSpec> = {
    vision: { model: 'my-finetune', capabilities: { images: true, json: true } },
    oss: {
        model: 'openai/gpt-oss-120b',
        capabilities: { tools: true, json: true, reasoning: { supported: true, levels: ['low', 'medium', 'high'] } },
    },
};
```

| Capability | Values | Description |
|---|---|---|
| `images` | `boolean` | Image parts in `user` messages |
| `tools` | `boolean` | Function tools, tool calls, and tool messages in the history |
| `json` | `boolean` \| `LlmJsonMode` | `true` (adapter default), `'native'`, `'strict'`, `'grammar'`, `'prompt'`, or `false` |
| `webSearch` | `{ supported, default? }` | Web search grounding; `default` (default `false`) applies when the call does not set it |
| `batch` | `{ supported, default? }` | Asynchronous batch execution; `default` (default `false`) applies when the call does not set it |
| `reasoning` | `{ supported, default?, levels? }` | Reasoning effort; `default` applies when unset; `levels` restricts the accepted efforts |
| `embeddings` | `boolean` \| `{ supported, maxInputs? }` | Vector embeddings; `maxInputs` sets the maximum texts per provider call |

#### Adapter Support

| Adapter | `images` | `tools` | `json` | `webSearch` | `batch` | `reasoning` | `embeddings` |
|---|---|---|---|---|---|---|---|
| Anthropic (`ANTHROPIC_SUPPORT`) | yes | yes | `native` | yes | yes | yes | no |
| Gemini (`GEMINI_SUPPORT`) | yes | yes | `native` | yes | yes | yes | yes (adapter `maxEmbedInputs: 100`) |
| DeepSeek (`DEEPSEEK_SUPPORT`) | yes | yes | `prompt` | no | no | yes | no |
| Groq (`GROQ_SUPPORT`) | yes | yes | `strict` | no | no | yes | no |
| Ollama (`OLLAMA_SUPPORT`) | yes | yes | `grammar` | no | no | yes | yes |
| OpenAI-compatible | via `supports` | via `supports` | via `supports` | never | never | via `supports` | via `supports` |

`createLlm` throws `TypeError` if a model declares a capability its adapter does not support, sets `default: true` on `webSearch` or `batch` without `supported: true`, sets a reasoning `default` or `levels` without `supported: true`, declares `levels` that are not distinct valid efforts with at least one effort other than `none`, sets a reasoning `default` missing from `levels`, or sets `embeddings.maxInputs` above `adapter.maxEmbedInputs`. An explicit `json` mode overrides the adapter's default mode and only requires the adapter to support JSON.

#### Request Validation and Effort Mapping

A request that needs a capability the model lacks fails with `unsupported` before any network call (e.g. an image without `images: true`, or `tools` without `tools: true`). Disabling an unsupported capability is always allowed: `web_search: false`, `batch: false`, and `reasoning_effort: 'none'` are stripped on models without those capabilities.

With `reasoning.levels`, requested efforts outside the list map to the nearest declared level in the order:

`none` < `minimal` < `low` < `medium` < `high` < `xhigh` < `max`

An effort other than `none` never maps to `none`; ties resolve to the higher level. Without `none` in `levels`, `none` maps to the lowest declared level. Without `levels`, the requested effort is passed as-is to the adapter; how each adapter sends it is listed under [Providers](#providers).

Reasoning text returns in `data.reasoning`. Reasoning tokens are reported in `usage.reasoningTokens` and excluded from `usage.outputTokens`.

#### Model Inspection and Discovery

- `llm.models`: `readonly LlmModel[]` of all configured models (`{ name, provider, model, capabilities }`) with resolved capabilities.
- `llm.listModels(provider, options?)`: queries the adapter (`options?: { signal?: AbortSignal }`) and returns `Promise<Result<LlmModelInfo[], LlmError>>` (`{ id, displayName?, inputTokenLimit?, outputTokenLimit? }[]`); an unknown `provider` returns `invalid_request`, an adapter without `listModels` returns `unsupported`.

### Requests

`llm.generate(request, options?)` takes a subset of the OpenAI Chat Completions payload and optional call options `{ signal?: AbortSignal, extra?: Record<string, unknown> }` (`LlmCallOptions`; unknown keys fail with `invalid_request`):

| Field | Type | Description |
|---|---|---|
| `model` | `string` | Configured model name |
| `messages` | `LlmMessage[]` | `system`, `developer`, `user`, `assistant`, `tool`; at least one `user` message. Content is a string or a non-empty array of parts; assistant `content` may be `null` together with `tool_calls` |
| `response_format` | `LlmResponseFormat` | `{ type: 'text' }`, `{ type: 'json_object' }`, or `{ type: 'json_schema', json_schema: { name, schema?, description?, strict? } }` |
| `temperature` | `number` | Sampling temperature (finite number; range checked by the provider) |
| `top_p` | `number` | Nucleus sampling probability (finite number) |
| `max_tokens` / `max_completion_tokens` | `number` | Maximum output tokens (positive integer; not both in one layer) |
| `seed` | `number` | Random seed (finite number) |
| `stop` | `string` \| `string[]` | Stop sequence(s) |
| `reasoning_effort` | `LlmReasoningEffort` | `'none'`, `'minimal'`, `'low'`, `'medium'`, `'high'`, `'xhigh'`, `'max'` |
| `web_search` | `boolean` | Web search grounding (Gemini, Anthropic) |
| `batch` | `boolean` | Asynchronous batch execution (Gemini, Anthropic) |
| `tools` | `LlmTool[]` | Function definitions available to the model |
| `tool_choice` | `LlmToolChoice` | `'auto'`, `'none'`, `'required'`, or `{ type: 'function', function: { name } }` |

Message rules (violations fail with `invalid_request`): `tool_calls` (non-empty, `type: 'function'`, non-empty `id`, `function.arguments` empty or JSON object text) and `reasoning` only on `assistant` messages; `tool_call_id` only on `tool` messages, matching a tool call id of an earlier assistant message; image parts only in `user` messages.

### Tool Calls

Define tools as function schemas. Returned calls appear in `data.toolCalls`. Continue the conversation by appending the assistant message and the tool results:

```ts
import { assistantMessage, createLlm, type LlmMessage, type LlmTool, toolMessage } from '@smart-llm/core';
import { createOllamaProvider } from '@smart-llm/ollama';

const llm = createLlm({
    providers: {
        ollama: {
            adapter: createOllamaProvider(),
            models: { qwen: { model: 'qwen2.5:7b', capabilities: { tools: true } } },
        },
    },
});

async function getWeather(city: string): Promise<{ city: string; celsius: number }> {
    return { city, celsius: 21 };
}

const tools: LlmTool[] = [{
    type: 'function',
    function: {
        name: 'get_weather',
        description: 'Current weather for a city',
        parameters: {
            type: 'object',
            properties: { city: { type: 'string' } },
            required: ['city'],
        },
    },
}];

const messages: LlmMessage[] = [{ role: 'user', content: 'Weather in Graz?' }];
const first = await llm.qwen(messages, { tools });

if (first.ok && first.data.toolCalls) {
    messages.push(assistantMessage(first.data));
    for (const call of first.data.toolCalls) {
        const args = JSON.parse(call.function.arguments);
        messages.push(toolMessage(call, await getWeather(args.city)));
    }
    const second = await llm.qwen(messages, { tools });
}
```

- `assistantMessage(output)`: builds an `assistant` message from `data` or `error.output`; keeps text, `tool_calls`, and `reasoning`. With `output.text === ''` and non-empty `output.toolCalls`, `content` is `null`.
- `toolMessage(call, result)`: builds a `tool` message. Strings are used as-is; other values are JSON-serialized (a top-level `undefined`, function, or symbol becomes `'null'`, `BigInt` a decimal string, a circular reference `"[Circular]"`). It never throws: when serialization fails (e.g. a throwing `toJSON`), `content` falls back to `String(result)`, then to `[unserializable tool result]`.
- `tools`: non-empty array of unique function declarations; names match `^[A-Za-z0-9_-]{1,64}$`. Invalid or duplicate names fail with `invalid_request`.
- `tool_choice`: dropped when `tools` is omitted. A named function choice must match a declared function.
- `parameters` and `json_schema.schema`: the root must be a JSON Schema object; sub-schemas must be objects or booleans, without circular references (`invalid_request` otherwise). Tool `strict` is a boolean or `null`.
- `response_format`: a tool call turn is not validated as JSON; validation applies to the final text answer.
- `extra_content` on tool calls holds adapter data for replay (Anthropic content blocks, Gemini thought signatures); keep it by building the assistant turn with `assistantMessage`. The OpenAI-compatible adapters strip `extra_content` before sending.

### Structured JSON Output

Pass a JSON Schema in `response_format`. The parsed and validated value is returned in `data.json`:

```ts
import { createLlm, type LlmModelInput, type LlmResponseFormat } from '@smart-llm/core';
import { createOllamaProvider } from '@smart-llm/ollama';

const llm = createLlm({
    providers: {
        ollama: {
            adapter: createOllamaProvider(),
            models: { qwen: { model: 'qwen2.5:7b', capabilities: { json: true } } },
        },
    },
});

const word: LlmResponseFormat = {
    type: 'json_schema',
    json_schema: {
        name: 'word',
        schema: {
            type: 'object',
            properties: {
                kind: { type: 'string', enum: ['noun', 'verb'], description: 'Word class' },
            },
            required: ['kind'],
        },
    },
};

const result = await llm.qwen('Classify "house".', { response_format: word });
if (result.ok) {
    console.log(result.data.json); // { kind: 'noun' }
} else if (result.error.code === 'invalid_json') {
    console.log('Raw text:', result.error.output?.text);
    console.log('Parsed JSON (if schema mismatch):', result.error.output?.json);
}

// Reusable model call with fixed schema
const classify = (input: LlmModelInput) => llm.qwen(input, { response_format: word });
```

- **Supported schema subset**: `type` (`object`, `array`, `string`, `number`, `integer`, `boolean`, `null`, or an array of types), `properties`, `required`, `additionalProperties` (enforced only when `false`), `items`, `enum`, `const`, `anyOf`, local `$ref`s (`#/$defs/...`, `#/definitions/...`), and `description`. Other keywords (`allOf`, `oneOf`, `not`, formats, bounds) are sent but not validated. `$schema` and `$id` are stripped.
- **Validation**: markdown code fences (```` ```json ... ``` ````) are stripped before parsing. Invalid JSON or a schema violation fails with `invalid_json` and `retryable: false` after one attempt; no JSON mode re-sends the request. The raw output is in `error.output.text` (and `error.text`); if the text parsed but failed validation, `error.output.json` holds the parsed value (after `removeOptionalNulls` in `strict` mode). With `{ type: 'json_object' }`, or `json_schema` without `schema`, an answer that parses to a non-object (array, string, number, boolean, `null`) also fails with `invalid_json`.

#### JSON Modes

A model needs `capabilities: { json: true }` (or an explicit `LlmJsonMode`). Default mode per adapter:

| Mode | Default Adapter | Request |
|---|---|---|
| `native` | Gemini, Anthropic | Sends the cleaned schema in the provider's native schema parameter |
| `strict` | Groq | Sends `toStrictSchema(schema)` (all properties of objects with `properties` required, optional fields nullable, `additionalProperties: false`) with `strict: true` and removes `null` values of optional fields from the answer (`removeOptionalNulls`) |
| `grammar` | Ollama | Sends the cleaned schema in `response_format` and adds the schema with field descriptions to the system prompt |
| `prompt` | DeepSeek | Sends `{ type: 'json_object' }` and adds schema instructions to the system prompt |

`{ type: 'json_object' }` and `json_schema` without `schema` prepare `{ type: 'json_object' }` in every mode (`prompt` and `grammar` also add a generic JSON instruction; Anthropic sends that instruction in `system` instead of `json_object`). The added instruction follows the leading `system` and `developer` messages.

#### Schema Utilities

`@smart-llm/core` exports the schema transformations and validators for custom adapters and standalone use:

- `cleanSchema(schema)`: deep copy of `schema` with `$schema` and `$id` removed from every sub-schema (`properties`, `items`, `anyOf`, `allOf`, `oneOf`, `not`, `prefixItems`, `$defs`, `definitions`, object `additionalProperties`).
- `toStrictSchema(schema)`: cleans `schema`; in nodes whose `type` is or includes `'object'` and that have `properties` (under `properties`, `items`, `anyOf`, `$defs`, `definitions`), marks all properties `required`, sets `additionalProperties: false`, and makes optional fields nullable. Free-form `{ type: 'object' }` nodes and `allOf`, `oneOf`, `not`, `prefixItems`, tuple `items` (array form), and object `additionalProperties` subtrees are only cleaned.
- `removeOptionalNulls(value, schema)`: recursively removes `null` values of declared, non-required object properties (following `items`, matching `anyOf` branches, and local `$ref`s to `#/$defs/...` or `#/definitions/...`).
- `findSchemaMismatch(value, schema, path?)`: validates `value` against `schema` (`type`, `enum`, `const`, `properties`, `required`, `additionalProperties: false`, `items`, `anyOf`, local `$ref`s; `type: 'object'` rejects arrays and class instances); ignores `allOf`, `oneOf`, `not`, `prefixItems`, schema-valued `additionalProperties`, formats, bounds, and non-local refs. Returns a mismatch path string (e.g. `'$.items[0].kind is not one of ["noun","verb"]'`) or `undefined`.
- `parseJsonText(text)`: parses a JSON string (stripping markdown code fences); returns `{ value }` or `undefined`.
- `schemaInstruction(schema)`: builds the system prompt instruction of the `prompt` and `grammar` modes (`schema: LlmJsonSchema | undefined`).

### Images

Images are base64 data URLs (`data:image/...;base64,...`) in `user` messages:

```ts
import { readFile } from 'node:fs/promises';
import { createLlm, imageFromFile, imagePart } from '@smart-llm/core';
import { createGeminiProvider } from '@smart-llm/gemini';

const llm = createLlm({
    providers: {
        gemini: {
            adapter: createGeminiProvider(),
            models: { flash: { model: 'gemini-3.8-flash', capabilities: { images: true } } },
        },
    },
});

// From disk (detects .png, .jpg, .jpeg, .webp, .gif; or pass an explicit mimeType)
const photo = await imageFromFile('photo.jpg');

// From bytes (non-empty Uint8Array or ArrayBuffer) with optional detail ('auto' | 'low' | 'high')
const bytes = await readFile('chart.png');
const chart = imagePart(bytes, 'image/png', 'high');

await llm.generate({
    model: 'flash',
    messages: [{ role: 'user', content: [{ type: 'text', text: 'Compare these images.' }, photo, chart] }],
});
```

`imageFromFile(path, mimeType?)` matches the extension case-insensitively and rejects with `TypeError` for other extensions unless `mimeType` is given. `imagePart` throws `TypeError` when `mimeType` is not `image/<subtype>` without parameters (any case, lowercased in the URL), `data` is not a non-empty `Uint8Array` (including `Buffer`) or `ArrayBuffer`, or `detail` is not `'auto'`, `'low'`, or `'high'`.

### Embeddings

`llm.embed(request, options?)` returns vector embeddings for one or more texts from models with `embeddings: true`:

```ts
import { createLlm } from '@smart-llm/core';
import { createGeminiProvider } from '@smart-llm/gemini';
import { createOllamaProvider } from '@smart-llm/ollama';

const llm = createLlm({
    providers: {
        ollama: {
            adapter: createOllamaProvider(),
            models: { bge: { model: 'bge-m3', capabilities: { embeddings: true } } },
        },
        gemini: {
            adapter: createGeminiProvider(),
            models: {
                vec: {
                    model: 'gemini-embedding-2',
                    capabilities: { embeddings: true },
                    modelLimits: { rpm: 100 },
                },
            },
        },
    },
});

const texts = ['First paragraph', 'Second paragraph'];
const paragraphs = await llm.embed({ model: 'bge', input: texts });
// paragraphs.data: { embeddings: number[][], name: 'bge', provider: 'ollama', model: 'bge-m3', usage: { inputTokens: 42 }, ms: 120, calls: 1, attempts: 1 }

const sections = await llm.embed({
    model: 'vec',
    input: [{ text: 'Cells were incubated at 37 °C.', title: 'Methods' }],
    task: 'document',
    dimensions: 768,
});
```

- **Chunking**: inputs (`string | readonly (string | { text: string; title?: string })[]`, non-empty strings) are split into chunks of `maxInputs` (model setting, else adapter `maxEmbedInputs`, else `64`). Chunks run through the model's embed queue with its rate limits and backoff.
- **Fail-fast**: if a chunk fails, the other chunks of the request are aborted, and the call resolves with the failing error and the total `attempts`. When the shared queue stops for another reason (e.g. another request's error), running provider calls of the request finish on their own and its waiting chunks are released with a `limiter` error.
- **Task and title**: `task` (`'query'`, `'document'`, `'similarity'`, `'classification'`, `'clustering'`) and `title` (only with `task: 'document'`); adapters that do not support them return `unsupported`.
- **Dimensions**: `dimensions` (positive integer) is passed to the adapter.
- **Usage**: `usage.inputTokens` is the sum over chunks, or `null` when the adapter reports none.

### Batch Calls

Batch calls are `generate` calls that resolve when the provider's batch job completes (Gemini, Anthropic):

```ts
import { createLlm } from '@smart-llm/core';
import { createGeminiProvider } from '@smart-llm/gemini';

const llm = createLlm({
    providers: {
        gemini: {
            adapter: createGeminiProvider({ batchWaitSec: 30 }),  // collection window in seconds (default 10, range 1 to 600)
            models: {
                flash: { model: 'gemini-3.8-flash', modelLimits: { rpm: 1_000 } },
                flashBatch: {
                    model: 'gemini-3.8-flash',
                    capabilities: { batch: { supported: true, default: true } },
                    modelLimits: { concurrency: 500 },           // maximum pending batch calls
                },
            },
        },
    },
});

const items = ['First question', 'Second question'];
const answers = await Promise.all(items.map((item) => llm.flashBatch(item)));
```

- **Collection window**: requests that arrive within `batchWaitSec` are collected into one batch job, so dispatch them concurrently. Grouping, size limits, and polling depend on the adapter (see [Providers](#providers)).
- **Dedicated queue**: batch calls use a separate batch queue per model and wait only for `modelLimits.concurrency` (pending batch call slots). They bypass `sharedLimits`, rate limits (`rpm`, `tpm`, `intervalMs`), token booking, and queue locks. Every retryable error retries with backoff and counts against `limiterOptions.maxRetries`.
- **Persistence**: batch jobs are not persisted across process restarts. If the process ends, pending promises are lost while the job continues at the provider. For durable Gemini batch jobs, use `smart-genai` directly.
- **Support**: adapters without batch support return `unsupported` for `batch: true`.

### Responses

Every successful `generate` call returns `{ ok: true, data }` with `data: LlmOutput`:

| Field | Type | Description |
|---|---|---|
| `text` | `string` | Generated text |
| `json` | `unknown` | Parsed and validated JSON value (only for JSON `response_format`; absent when `toolCalls` are returned) |
| `reasoning` | `string` \| `undefined` | Reasoning text returned by the model |
| `sources` | `LlmSource[]` \| `undefined` | Web search citations (`{ url, title? }`) |
| `toolCalls` | `LlmToolCall[]` \| `undefined` | Tool calls requested by the model |
| `name` | `string` | Configured model name |
| `provider` | `string` | Configured provider name |
| `model` | `string` | API-reported model version or configured provider model id |
| `usage` | `LlmUsage` | `{ inputTokens, outputTokens, reasoningTokens }` (`number \| null`); `outputTokens` never includes reasoning tokens |
| `finishReason` | `string` \| `null` | Reason generation stopped (`'stop'`, `'tool_calls'`, `'length'`, etc.) |
| `ms` | `number` | Request time in milliseconds |
| `settings` | `LlmSettings` | Resolved parameters sent to the provider |
| `attempts` | `number` | Provider calls made, including retries |

`data.settings` and `error.settings` hold the resolved options of the call: `name`, `provider`, `model`, `temperature`, `top_p`, `max_tokens`, `seed`, `stop`, `reasoning_effort` (after level mapping), `web_search` (`true` when active), `batch` (`true` when active), `jsonMode`, `tools` (function names), `tool_choice` (absent without `tools`), and the merged `extra` (when non-empty).

`embed` returns `LlmEmbedOutput`: `{ embeddings: number[][], name, provider, model, usage: { inputTokens: number | null }, ms, calls, attempts }`.

### Errors

Failed calls return `{ ok: false, error }` with `error: LlmError`:

- `code`: `LlmErrorCode` (table below).
- `message`: failure description.
- `retryable`: whether a retry could succeed.
- `output`: `LlmAnswer` with the unusable answer of the last attempt when a completion was received (`invalid_json`, `truncated`, `blocked`, or an unexpected finish reason).
- `settings`: `LlmSettings` of the call (on `generate` errors after request preparation; absent on `embed`, `listModels`, and pre-settings errors).
- `attempts`: provider calls attempted; present after request preparation (`0` if the call failed before dispatch, e.g. `limiter` or `aborted`), absent on validation errors (`invalid_request`, `unsupported`), `listModels`, and pre-queue `embed` errors.
- Further fields when available: `status` (HTTP status), `retryAfterMs` (parsed `retry-after` delay), `limiterCode` (smart-limit error code), `text`, `usage`, `model`, `ms`, `name`, `provider`.

| Code | Retryable | Description |
|---|---|---|
| `rate_limited` | yes | HTTP 429; locks the queue (`backoff_global`) until `retry-after` or backoff |
| `timeout` | per adapter | Request timed out (`retry_backoff` when retryable) |
| `network` | yes | Network or connection failure (`backoff_global`) |
| `http` | per adapter | HTTP error; retryable 5xx lock the queue (`backoff_global`), other retryable statuses use `retry_backoff`; each adapter's retryable statuses are listed under [Providers](#providers) |
| `aborted` | no | Call canceled via the caller's `AbortSignal`; canceled before dispatch, it books no tokens |
| `blocked` | no | Output blocked by a provider safety or content filter; carries `output` |
| `truncated` | no | Output token limit reached; carries `output` |
| `invalid_json` | no | Output is not valid JSON or violates the schema; carries `output`; never retried |
| `unsupported` | no | Request needs a capability the model lacks (including `embed` on a model without `embeddings`), the adapter does not support an embedding option (`task` / `title`), or the adapter has no `listModels` |
| `invalid_request` | no | Invalid request or options at call time (unknown fields, unknown model or provider name, invalid `extra`) |
| `limiter` | no | smart-limit rejected the call before dispatch (`limiterCode`: `COST_EXCEEDS_LIMIT` when the estimate alone exceeds a bucket capacity, `MAX_WAIT_EXCEEDED`, `STOPPED` for released embed chunks, or `BUCKET_FAILED`) |
| `unexpected` | per adapter | Unexpected response structure, adapter exception, or adapter result that breaks the contract; retryable only when the adapter marks it |

### Custom Adapter Contract

Custom adapters implement `LlmProvider`:

- `name`: provider name.
- `supports`: `LlmProviderSupport` (`images`, `tools`, `webSearch`, `reasoning`, `batch`, `json`, `embeddings`); boolean flags are `true` or omitted (`false` throws `TypeError`), `json` is `LlmJsonMode` or omitted.
- `generate(request, { signal, extra })`: sends one API request and returns `Promise<Result<LlmProviderOutput, LlmProviderError>>`. Must not throw or reject (throws are caught and mapped to `unexpected`). An output needs string `text` and `model`, a `usage` object, and numeric `ms`; an error needs string `code` and `message` and boolean `retryable`. Each `usage` field (`inputTokens`, `outputTokens`, `reasoningTokens`) on outputs and errors must be a finite number `>= 0` or `null`; otherwise the call fails as `unexpected`.
- `listModels?(options?)`: returns `Promise<Result<LlmModelInfo[], LlmError>>`.
- `validateExtra?(extra, option)`: validates `extra`; throws `TypeError` on invalid options.
- `mergeExtra?(layers)`: custom merge of provider, model, and call `extra`.
- `embed?(request, { signal })`: receives `{ model, input: { text, title? }[], task?, dimensions? }` and sends one embedding request for up to `maxEmbedInputs` texts, returning `Promise<Result<LlmProviderEmbedOutput, LlmProviderError>>`. Required when `supports.embeddings` is `true`. `embeddings.length` must equal the number of inputs; `usage.inputTokens` must be a finite number `>= 0` or `null`, otherwise the call fails as `unexpected`.
- `maxEmbedInputs?`: maximum texts per embedding call (positive integer; default `64`).

`@smart-llm/openai-compatible` exports `fetchJson` and `statusError` for adapters on HTTP JSON APIs (see [OpenAI-Compatible](#openai-compatible-smart-llmopenai-compatible)).

## Providers

The package READMEs also list all exports.

### Anthropic (`@smart-llm/anthropic`)

#### Options

`createAnthropicProvider(clientOrOptions?)` accepts an `Anthropic` SDK client or `AnthropicOptions` and throws `TypeError` on invalid options or unknown keys. With an SDK client, the client's positive `timeout` (else `600_000` ms) applies and the other options keep their defaults. SDK retries are disabled (also per request on a passed client).

| Option | Type | Default | Description |
|---|---|---|---|
| `apiKey` | `string` | env `ANTHROPIC_API_KEY` | API key (`ANTHROPIC_API_KEY_ENV`); throws `TypeError` if missing, `''`, or not a string |
| `env` | `Readonly<Record<string, string \| undefined>>` | `process.env` | Environment map for the API key lookup |
| `baseUrl` | `string` | env `ANTHROPIC_BASE_URL`, else `'https://api.anthropic.com'` | Absolute `http:` or `https:` base URL (the SDK reads `ANTHROPIC_BASE_URL` from `process.env`, not from `env`) |
| `headers` | `Record<string, string>` | `undefined` | Extra HTTP headers (e.g. `anthropic-beta`) |
| `timeoutMs` | `number` | `600_000` | Request timeout in milliseconds (positive integer; 10 minutes) |
| `fetch` | `typeof fetch` | `globalThis.fetch` | Custom fetch implementation |
| `batchWaitSec` | `number` | `10` | Collection window in seconds for batch calls (integer `1` to `600`) |
| `defaultMaxTokens` | `number` | `16_000` | `max_tokens` sent when the request sets none (`DEFAULT_MAX_TOKENS`, positive integer) |

#### Extra Options

`params`, `thinkingBudgets`, and `webSearchTool` deep-merge across layers; later layers replace `thinkingDisplay` and `thinkingOff`.

| Key | Type | Description |
|---|---|---|
| `params` | `Record<string, unknown>` | Raw Messages API fields deep-merged last over the built request (`null` deletes a key; `model`, `messages`, and `stream` are rejected) |
| `thinkingBudgets` | `Partial<Record<Exclude<LlmReasoningEffort, 'none'>, number>>` | Token budget per effort (integer `>= 1024` and `< max_tokens`); sends budget thinking instead of adaptive thinking |
| `thinkingDisplay` | `'summarized'` \| `'omitted'` | Thinking display; `'summarized'` (default) fills `data.reasoning` |
| `thinkingOff` | `'disabled'` \| `'between_tools'` \| `'omit'` | Sent for `reasoning_effort: 'none'`: `thinking: { type: 'disabled' }` (default), `thinking: { type: 'between_tools' }`, or no `thinking` field |
| `webSearchTool` | `Record<string, unknown>` | Merged into `{ type: 'web_search_20260209', name: 'web_search' }` (e.g. `max_uses`, `allowed_domains`) |

#### Capabilities

- **Messages**: `system` and `developer` messages go to the top-level `system` field; `tool` messages become `tool_result` blocks. `temperature`, `top_p`, and `seed` are not sent. `usage.inputTokens` includes cache creation and cache read tokens.
- **Images**: base64 JPEG, PNG, GIF, and WebP; `detail` is not sent; other formats fail with `invalid_request`.
- **Tools**: `tool_choice` `'required'` is sent as `{ type: 'any' }`, `'auto'` as `{ type: 'auto' }`, `'none'` as `{ type: 'none' }`, and a named function as `{ type: 'tool', name }`. Raw content blocks of tool-calling answers are stored on `toolCalls[0].extra_content.anthropic.content` and replayed unchanged by `assistantMessage(output)`; thinking blocks are stripped from turns before the current tool loop.
- **JSON** (`native`): requires an object root schema; open objects, boolean sub-schemas, tuple `items`, and untyped nodes without `type`/`anyOf`/`oneOf`/`allOf`/`$ref`/`enum`/`const` fail with `invalid_request` (same rules for `tool.strict: true`). `json_object` and schemaless `json_schema` append a JSON instruction to `system`; models without native structured output can declare `capabilities: { json: 'prompt' }`.
- **Reasoning**: adaptive thinking with output `effort` (`minimal` is sent as `low`), or budget thinking with `extra.thinkingBudgets` (declare `levels` matching the budget keys); `none` follows `extra.thinkingOff`.
- **Web search**: adds the `web_search` server tool; citations fill `data.sources`.
- **Batch**: requests of all models within `batchWaitSec` go into Message Batches (at most `100_000` requests and `250_000_000` bytes per batch; a single larger request fails with `invalid_request`), polled every 60 s for up to 25 hours; aborting all calls in a batch cancels it.
- **Embeddings**: not supported.

#### Errors

HTTP 429 maps to `rate_limited`; 408, 409, and 5xx to retryable `http`; `retry-after-ms` / `retry-after` are honored; connection failures map to retryable `network`. Request timeouts map to retryable `timeout`. Stop reasons `max_tokens` / `model_context_window_exceeded` map to `truncated`, `refusal` to `blocked`, `pause_turn` and unknown reasons to non-retryable `unexpected`; non-message 2xx bodies, `tool_use` without calls, or non-object `tool_use` inputs map to retryable `unexpected`. A batch that does not finish in 25 hours is canceled and returns non-retryable `timeout`; expired items return retryable `timeout`, canceled items `aborted`, and errored items `rate_limited` (`rate_limit_error`), retryable `http` (`overloaded_error`, `api_error`, `timeout_error`), or non-retryable `http`.

### Gemini (`@smart-llm/gemini`)

#### Options

`createGeminiProvider(clientOrOptions?)` accepts a `GenaiClient` from `smart-genai` or `GeminiOptions` (`Omit<GenaiClientOptions, 'defaults'>`). Passing `defaults` or both `apiKey` and `vertex` throws `TypeError`.

| Option | Type | Default | Description |
|---|---|---|---|
| `apiKey` | `string` | env `GEMINI_API_KEY` | Developer API key (AI Studio; `GEMINI_API_KEY_ENV`) |
| `vertex` | `boolean` \| `GenaiVertexOptions` | `undefined` | `true` or `{ project?, location?, bucket?, credentialsFile?, authClient?, storageUrl? }` for Vertex AI |
| `baseUrl` | `string` | official endpoint | Custom API endpoint URL |
| `env` | `Readonly<Record<string, string \| undefined>>` | `process.env` | Environment map for `GEMINI_API_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` (default `'global'`), `VERTEX_BATCH_BUCKET`, and `GOOGLE_APPLICATION_CREDENTIALS` |
| `timeoutMs` | `number` | `120_000` | HTTP timeout per call in milliseconds (2 minutes) |
| `batchWaitSec` | `number` | `10` | Collection window in seconds for batch calls (`1` to `600`) |

#### Extra Options

| Key | Type | Description |
|---|---|---|
| `config` | `GenerateContentConfig` | Gemini request configuration passed to smart-genai; the only accepted key (`GeminiExtra`). Rejects `abortSignal`, `httpOptions`, `automaticContinuation`, and non-array/non-null `tools` |

`config` merges per top-level field across layers (nested objects other than `thinkingConfig` are replaced whole); `thinkingConfig` merges per member, and `thinkingBudget` and `thinkingLevel` count as one setting, so the last layer that sets either wins.

#### Capabilities

- **Messages**: `stop` takes up to 5 sequences and `seed` must be a 32-bit signed integer (`invalid_request` otherwise).
- **Images**: base64 data URLs; `detail` is not sent.
- **Tools**: missing call IDs become `call_` plus 32 hex characters. Thought signatures are kept in `toolCall.extra_content.google.thought_signature` and must stay in the history (`assistantMessage` keeps them). `tools` with `web_search: true` fail with `invalid_request` on `gemini-2.*` models.
- **JSON** (`native`): sends the cleaned schema as `responseJsonSchema`.
- **Reasoning**: `gemini-2.*` models get `thinkingBudget` (`0` for `none`; at least `128` on `gemini-2.5-pro`). Gemma models get `thinkingLevel` `MINIMAL` for `none` to `low` and `HIGH` above. Other models get `thinkingLevel`: `none` and `minimal` send `MINIMAL` on Gemini 3.0 to 3.6 Flash and Flash-Lite and `LOW` on others, `xhigh` and `max` send `HIGH`. Thought summaries are requested whenever `reasoning_effort` is set, except when `none` sends `thinkingBudget: 0`; turn them off with `extra: { config: { thinkingConfig: { includeThoughts: false } } }`.
- **Web search**: Google Search grounding; citations fill `data.sources`.
- **Batch**: groups calls by `model`, `config`, `response_format`, and `tools` into Gemini Batch API jobs and splits a group above the payload limit; Vertex AI needs `vertex.bucket` or env `VERTEX_BATCH_BUCKET`. Jobs are polled every 30 s for up to 72 hours.
- **Embeddings**: `task` maps to Gemini task types (`RETRIEVAL_QUERY`, `RETRIEVAL_DOCUMENT`, `SEMANTIC_SIMILARITY`, `CLASSIFICATION`, `CLUSTERING`); `dimensions` is supported; `usage.inputTokens` is always `null`. Up to `100` texts per call on the Developer API, and on Vertex AI for `gemini-embedding-001` and non-Gemini (non-MaaS) models. On Vertex AI, other Gemini models (including `gemini-embedding-2`) and MaaS models return `unsupported` for more than one text per call; set `capabilities: { embeddings: { supported: true, maxInputs: 1 } }` there so `llm.embed` sends one text per call. Models other than `gemini-embedding-2` accept only one shared `title` per call (`invalid_request` on mixed titles).

#### Errors

Thrown `TypeError`s of smart-genai map to `invalid_request`, other exceptions to `unexpected`. A batch job that does not finish in 72 hours returns non-retryable `timeout`. Other errors keep the `code`, `retryable`, `status`, and `retryAfterMs` of smart-genai.

### DeepSeek (`@smart-llm/deepseek`)

#### Options

`createDeepseekProvider(options?: DeepseekOptions)`:

| Option | Type | Default | Description |
|---|---|---|---|
| `apiKey` | `string` | env `DEEPSEEK_API_KEY` | API key (`DEEPSEEK_API_KEY_ENV`); throws `TypeError` if missing or `''` |
| `env` | `Readonly<Record<string, string \| undefined>>` | `process.env` | Environment map for the API key lookup |
| `baseUrl` | `string` | `'https://api.deepseek.com'` | API base URL |
| `timeoutMs` | `number` | `600_000` | Request timeout in milliseconds (10 minutes) |
| `fetch` | `typeof fetch` | `globalThis.fetch` | Custom fetch implementation |

#### Extra Options

`extra` fields are spread onto the `/chat/completions` body.

#### Capabilities

- **Messages**: `max_tokens` / `max_completion_tokens` are sent as `max_tokens`.
- **JSON** (`prompt`): `{ type: 'json_object' }` plus schema instructions in the system prompt.
- **Reasoning**: `reasoning_effort` is sent as-is; `reasoning_content` returns in `data.reasoning`. Assistant `reasoning` in the history is sent back as `reasoning_content` (build assistant turns with `assistantMessage` to keep it).
- **Web search, batch, embeddings**: not supported.

#### Errors

Same mapping as [OpenAI-Compatible](#openai-compatible-smart-llmopenai-compatible).

### Groq (`@smart-llm/groq`)

#### Options

`createGroqProvider(options?: GroqOptions)`:

| Option | Type | Default | Description |
|---|---|---|---|
| `apiKey` | `string` | env `GROQ_API_KEY` | API key (`GROQ_API_KEY_ENV`); throws `TypeError` if missing or `''` |
| `env` | `Readonly<Record<string, string \| undefined>>` | `process.env` | Environment map for the API key lookup |
| `baseUrl` | `string` | `'https://api.groq.com/openai/v1'` | API base URL |
| `timeoutMs` | `number` | `120_000` | Request timeout in milliseconds (2 minutes) |
| `fetch` | `typeof fetch` | `globalThis.fetch` | Custom fetch implementation |

#### Extra Options

`extra` fields are spread onto the `/chat/completions` body, e.g. `extra: { reasoning_format: 'parsed' }`.

#### Capabilities

- **Messages**: `max_tokens` / `max_completion_tokens` are sent as `max_completion_tokens`.
- **JSON** (`strict`): `toStrictSchema` with `strict: true`; for models without strict schema support, declare `capabilities: { json: 'prompt' }`.
- **Reasoning**: `reasoning_effort` is sent as-is; `message.reasoning` returns in `data.reasoning`; assistant `reasoning` in the history is not sent. Declare the efforts a model accepts in `capabilities.reasoning.levels`.
- **Web search, batch, embeddings**: not supported.

#### Errors

Same mapping as [OpenAI-Compatible](#openai-compatible-smart-llmopenai-compatible), including `json_validate_failed` (`invalid_json` with `failed_generation` in `error.text` and `error.output.text`) and `tool_use_failed` (retryable `http`).

### Ollama (`@smart-llm/ollama`)

#### Options

`createOllamaProvider(options?: OllamaOptions)` connects to a local or remote Ollama server:

| Option | Type | Default | Description |
|---|---|---|---|
| `baseUrl` | `string` | `'http://localhost:11434/v1'` | OpenAI-compatible Ollama endpoint URL |
| `apiKey` | `string` | `undefined` | Bearer token for authenticated reverse proxies |
| `timeoutMs` | `number` | `600_000` | Request timeout in milliseconds (10 minutes) |
| `fetch` | `typeof fetch` | `globalThis.fetch` | Custom fetch implementation |

`fetchOllamaCapabilities(options?: OllamaCapabilitiesOptions)` reads model capabilities from `GET /api/tags` (`images`, `tools`, `reasoning`, `embeddings`; never `json`). It accepts `{ baseUrl?, apiKey?, timeoutMs?, signal?, fetch? }` (default `timeoutMs: 30_000`), strips a trailing `/v1` and trailing slashes from `baseUrl`, returns a `Result`, and never throws:

```ts
import { fetchOllamaCapabilities } from '@smart-llm/ollama';

const reported = await fetchOllamaCapabilities();
if (!reported.ok) throw new Error(reported.error.message);

const qwen = {
    model: 'qwen2.5:7b',
    capabilities: { ...reported.data['qwen2.5:7b'], json: true },
};
```

#### Extra Options

`extra` fields are spread onto the `/chat/completions` body.

#### Capabilities

- **Messages**: `max_tokens` / `max_completion_tokens` are sent as `max_tokens`.
- **Tools**: `tool_choice` is never sent; `tool_choice: 'none'` removes `tools` from the body.
- **JSON** (`grammar`): sends the cleaned schema in `response_format` and adds it to the system prompt.
- **Reasoning**: `reasoning_effort` is sent as-is (including `'none'`); `message.reasoning` returns in `data.reasoning`; assistant `reasoning` in the history is not sent.
- **Embeddings**: `POST <baseUrl>/embeddings` with `dimensions`; `task` and `title` return `unsupported`.
- **Web search, batch**: not supported.

#### Errors

Same mapping as [OpenAI-Compatible](#openai-compatible-smart-llmopenai-compatible).

### OpenAI-Compatible (`@smart-llm/openai-compatible`)

#### Options

`createOpenAiCompatibleProvider(options: OpenAiCompatibleOptions)` connects to any `/chat/completions`, `/embeddings`, and `/models` API:

```ts
import { createOpenAiCompatibleProvider } from '@smart-llm/openai-compatible';

const apiKey = process.env.TOGETHER_API_KEY;
if (!apiKey) throw new Error('TOGETHER_API_KEY is not set');

const together = createOpenAiCompatibleProvider({
    name: 'together',
    baseUrl: 'https://api.together.xyz/v1',
    apiKey,
    supports: { tools: true, json: 'prompt', embeddings: true }, // omit unsupported capabilities
});
```

| Option | Type | Default | Description |
|---|---|---|---|
| `name` | `string` | *required* | Provider name |
| `baseUrl` | `string` | *required* | API base URL (e.g. `https://api.together.xyz/v1`) |
| `apiKey` | `string` | `undefined` | Bearer token; without it no `authorization` header is sent |
| `headers` | `Record<string, string>` | `undefined` | Custom headers; lowercase `content-type` / `authorization` keys override the defaults |
| `supports` | `Omit<LlmProviderSupport, 'webSearch' \| 'batch'>` | `{}` | Supported capabilities (`images`, `tools`, `reasoning`, `json`, `embeddings`) |
| `maxTokensField` | `'max_tokens'` \| `'max_completion_tokens'` | `'max_completion_tokens'` | Body field for the output token limit |
| `mapBody` | `(body, request) => body` | `undefined` | Transforms the request body after `extra` is applied |
| `reasoningFields` | `readonly string[]` | `['reasoning_content', 'reasoning']` | Response message fields read as reasoning text |
| `reasoningMessageField` | `string` | `undefined` | Assistant message field that replays `reasoning` in the history |
| `timeoutMs` | `number` | `120_000` | Request timeout in milliseconds (2 minutes) |
| `fetch` | `typeof fetch` | `globalThis.fetch` | Custom fetch implementation |

#### Extra Options

`extra` fields are spread onto the `/chat/completions` body before `mapBody`.

#### Capabilities

- **Messages**: `developer` messages are sent as `system`; `extra_content` is stripped from outgoing tool calls; empty tool call `arguments` are sent as `'{}'`.
- **Tools**: tool calls without a function name are dropped; a missing `id` becomes `call_` plus 32 hex characters; missing, `null`, or `''` incoming arguments become `'{}'`; arguments that are not a JSON object fail with retryable `unexpected`.
- **Reasoning**: reasoning text is read from `reasoningFields`; `reasoning_tokens` is deducted from `completion_tokens` (clamped at `0`) into `usage.outputTokens` and reported as `usage.reasoningTokens`.
- **Embeddings**: with `supports.embeddings`, `POST /embeddings` with `{ model, input, dimensions? }`; `usage.prompt_tokens` is summed over chunks; `task` and `title` return `unsupported`.
- **Web search, batch**: never supported.

#### Errors

| Code | Retryable | When |
|---|---|---|
| `rate_limited` | yes | HTTP 429; `retry-after` (seconds or HTTP date) sets `retryAfterMs` |
| `http` | yes | HTTP 5xx, 408, 409, 425, or 498; error body code `tool_use_failed` |
| `http` | no | Other non-2xx statuses |
| `invalid_json` | no | Error body `error.code === 'json_validate_failed'` (`failed_generation` in `error.text` and `error.output.text`) |
| `timeout` / `aborted` / `network` | yes / no / yes | `timeoutMs` elapsed / caller `signal` aborted / `fetch` failed |
| `truncated` / `blocked` | no | Finish reason `length` / `content_filter` |
| `unexpected` | yes | Finish reason `tool_calls` without a usable call or `insufficient_system_resource`; tool call arguments that are not a JSON object |
| `unexpected` | no | Malformed response or other unknown finish reason |

`fetchJson(fetcher, url, init)` and `statusError(status, message, retryAfter)` expose this mapping for custom adapters: `fetchJson` runs `fetch`, parses JSON on 2xx, and returns `Result<unknown, LlmError>`; `statusError` maps an HTTP status and `retry-after` value to `LlmError`.

## License

Apache-2.0, see [LICENSE](LICENSE).
