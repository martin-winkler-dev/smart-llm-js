# smart-llm

[![Test](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml)
[![Package](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml)
[![Bun](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackage.json&query=%24.engines.bun&label=Bun&logo=bun&color=blue)](https://bun.sh)
[![Node.js](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackage.json&query=%24.engines.node&label=Node.js&logo=nodedotjs&color=blue)](https://nodejs.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/martin-winkler-dev/smart-llm-js/blob/main/LICENSE)

Unified multi-provider client for Gemini, DeepSeek, Groq, local Ollama models, and any OpenAI-compatible API. Configures providers, models, capabilities, and [smart-limit](https://www.npmjs.com/package/smart-limit) rate limits in one place; routes API calls by configured model name. No classes; post-initialization methods never throw.

| Package | Description |
|---|---|
| `@smart-llm/core` | Core coordinator (`createLlm`), message & tool helpers (`assistantMessage`, `toolMessage`), image helpers (`imagePart`, `imageFromFile`), JSON schema utilities (`cleanSchema`, `toStrictSchema`, `removeOptionalNulls`, `findSchemaMismatch`, `parseJsonText`, `schemaInstruction`), and TypeScript types |
| `@smart-llm/gemini` | Gemini adapter (`createGeminiProvider`, `GEMINI_SUPPORT`) for Developer API and Vertex AI via [smart-genai](https://www.npmjs.com/package/smart-genai) |
| `@smart-llm/deepseek` | DeepSeek adapter (`createDeepseekProvider`, `DEEPSEEK_SUPPORT`, `DEEPSEEK_API_KEY_ENV`) |
| `@smart-llm/groq` | Groq adapter (`createGroqProvider`, `GROQ_SUPPORT`, `GROQ_API_KEY_ENV`) |
| `@smart-llm/ollama` | Local Ollama adapter (`createOllamaProvider`, `fetchOllamaCapabilities`, `OLLAMA_SUPPORT`) |
| `@smart-llm/openai-compatible` | Generic OpenAI-compatible adapter (`createOpenAiCompatibleProvider`, `fetchJson`, `statusError`) for any `/chat/completions` and `/embeddings` endpoint |

```bash
bun add @smart-llm/core @smart-llm/gemini @smart-llm/deepseek @smart-llm/groq @smart-llm/ollama @smart-llm/openai-compatible
# or
npm install @smart-llm/core @smart-llm/gemini @smart-llm/deepseek @smart-llm/groq @smart-llm/ollama @smart-llm/openai-compatible
```

ESM packages with TypeScript types for Node.js >= 22.12 (`require()` supported) and Bun >= 1.3.

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

`extra` holds provider-specific settings and merges in the same priority order: provider `extra`, model `extra`, then call `extra`. By default, keys are shallow-merged (later keys win), unless the provider defines a custom `mergeExtra`. Adapters with `validateExtra` validate the configuration in `createLlm` (throwing a `TypeError` on invalid options) and call-level `extra` in `generate` (returning an `invalid_request` error). Gemini accepts only `{ config }` (smart-genai configuration) and merges `config` properties deeply across layers. OpenAI-compatible providers pass `extra` fields directly into the request body.

### Limits & Queuing

Limits are managed via [smart-limit](https://github.com/martin-winkler-dev/smart-limit-js). `sharedLimits` on a provider apply across all models of that provider (e.g. an account-wide RPM or single-concurrency local server). `modelLimits` apply only to that specific model. When both are configured, a request must satisfy both sets of limits. Without explicit limits, requests run unthrottled.

```ts
deepseek: {
    adapter: createDeepseekProvider(),
    sharedLimits: { concurrency: 10 },                                               // 10 concurrent requests across all models
    models: {
        'ds-flash': { model: 'deepseek-flash', modelLimits: { intervalMs: 3_000 } }, // at least 3 s between call starts
        'ds-pro': 'deepseek-pro',
    },
},
```

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

#### Rate Limit Locks & Probing

Each model maintains its own FIFO request queue. When an HTTP 429, 5xx, or network failure occurs, that model's queue enters a locked state (`backoff_global`). When the `retry-after` duration (or backoff delay, minimum `1_000` ms) elapses, the queue dispatches a single probe request. Only after this probe succeeds do remaining queued requests resume. Other models (including those under the same provider) continue processing independently unless they encounter errors of their own.

If the estimated wait time for bucket capacity or a lock exceeds `maxWaitMs`, all waiting requests for that model terminate immediately: requests that already attempted a provider call return their last provider error (e.g. `rate_limited` with `retryAfterMs`), while unstarted waiting requests return `error.code: 'limiter'` with `limiterCode: 'MAX_WAIT_EXCEEDED'`. Calls initiated while the lock duration exceeds `maxWaitMs` fail immediately without making a network request.

#### Token Booking & `charsPerToken`

`charsPerToken` calibrates the pre-call token estimate booked against token rate limits (`tpm`, `tpd`). It can be set globally, per provider, or per model (default: `4`, positive finite decimals allowed).

- **Generation calls (`generate`)**:
  `Math.ceil(characters / charsPerToken) + (images * 1_000) + (max_tokens ?? 1_000)`
  Characters are counted across message texts, assistant `reasoning`, `JSON.stringify(tools)`, assistant tool call names and arguments, and tool message contents. When the provider returns actual usage, booked tokens settle to `inputTokens + outputTokens + reasoningTokens`. On HTTP 429 errors without reported usage, booked tokens are refunded (`tokens: 0`).
- **Embedding calls (`embed`)**:
  Each batch chunk books `Math.ceil(characters / charsPerToken)` across input `text` and `title` strings, settling to `usage.inputTokens` when reported.

### Capabilities

Adapters define the features supported by their API (`adapter.supports`). Models then declare which of these features they enable via `capabilities`. Undeclared features default to unsupported (`false`). A model without `capabilities` accepts and returns plain text only.

```ts
models: {
    vision: { model: 'my-finetune', capabilities: { images: true, json: true } },
    oss: {
        model: 'openai/gpt-oss-120b',
        capabilities: { tools: true, json: true, reasoning: { supported: true, levels: ['low', 'medium', 'high'] } },
    },
}
```

| Capability | Values | Description |
|---|---|---|
| `images` | `boolean` | Image parts in `user` messages |
| `tools` | `boolean` | Function tools, tool calls, and tool messages in history |
| `json` | `boolean` \| `LlmJsonMode` | `true` (adapter default), `'native'`, `'strict'`, `'grammar'`, `'prompt'`, or `false` |
| `webSearch` | `{ supported, default? }` | Web search grounding; `default` (default `false`) applies when unset in call |
| `batch` | `{ supported, default? }` | Asynchronous batch execution; `default` (default `false`) applies when unset in call |
| `reasoning` | `{ supported, default?, levels? }` | Reasoning effort; `default` applies when unset; `levels` restricts supported effort levels |
| `embeddings` | `boolean` \| `{ supported, maxInputs? }` | Vector embeddings; `maxInputs` sets maximum texts per provider call |

#### Adapter Support Matrix

| Adapter | `images` | `tools` | `json` | `webSearch` | `batch` | `reasoning` | `embeddings` |
|---|---|---|---|---|---|---|---|
| Gemini (`GEMINI_SUPPORT`) | yes | yes | `native` | yes | yes | yes | yes (`maxEmbedInputs: 100`) |
| DeepSeek (`DEEPSEEK_SUPPORT`) | yes | yes | `prompt` | no | no | yes | no |
| Groq (`GROQ_SUPPORT`) | yes | yes | `strict` | no | no | yes | no |
| Ollama (`OLLAMA_SUPPORT`) | yes | yes | `grammar` | no | no | yes | yes |
| OpenAI-compatible | via `supports` | via `supports` | via `supports` | never | never | via `supports` | via `supports` |

`createLlm` throws a `TypeError` if a model declares a capability its adapter does not support, specifies a default on an unsupported capability, sets `embeddings.maxInputs` above `adapter.maxEmbedInputs`, or defines a reasoning `default` not present in `levels`.

#### Request Validation & Effort Mapping

A request requiring a capability the model lacks fails with `{ ok: false, error: { code: 'unsupported' } }` before a network call is dispatched (e.g. sending an image without `images: true`, or `tools` without `tools: true`). Disabling an unsupported capability is always permitted: `web_search: false`, `batch: false`, and `reasoning_effort: 'none'` on models lacking those capabilities are safely stripped.

When `reasoning.levels` is declared, requested efforts outside the list map to the nearest declared level in the order:

`none` < `minimal` < `low` < `medium` < `high` < `xhigh` < `max`

An effort other than `none` never maps to `none`. Ties resolve to the higher level. If `levels` does not include `none`, reasoning cannot be disabled, and `none` maps to the lowest declared level. Without `levels`, the requested effort is passed as-is.

#### Model Inspection & Discovery

- `llm.models`: `readonly LlmModel[]` listing all configured models (`{ name, provider, model, capabilities }`) with resolved capabilities.
- `llm.listModels(provider, options?)`: Queries available models from the configured provider adapter (`options?: { signal?: AbortSignal }`), returning `Promise<Result<LlmModelInfo[], LlmError>>` (`{ id, displayName?, inputTokenLimit?, outputTokenLimit? }[]`). Returns `invalid_request` if `provider` is unknown, or `unsupported` if the adapter does not implement `listModels`.

## Requests

Requests follow a standardized subset of the OpenAI Chat Completions payload:

| Field | Type | Description |
|---|---|---|
| `model` | `string` | Configured model name |
| `messages` | `LlmMessage[]` | Array of messages (`system`, `developer`, `user`, `assistant`, `tool`; at least one `user` message). Content can be a string or array of parts |
| `response_format` | `LlmResponseFormat` | `{ type: 'text' }`, `{ type: 'json_object' }`, or `{ type: 'json_schema', json_schema: { name, schema?, description?, strict? } }` |
| `temperature` | `number` | Sampling temperature |
| `top_p` | `number` | Nucleus sampling probability |
| `max_tokens` / `max_completion_tokens` | `number` | Maximum tokens to generate (positive integer; mutually exclusive within the same layer) |
| `seed` | `number` | Random seed for deterministic generation |
| `stop` | `string` \| `string[]` | Stop sequence(s) |
| `reasoning_effort` | `LlmReasoningEffort` | `'none'`, `'minimal'`, `'low'`, `'medium'`, `'high'`, `'xhigh'`, `'max'` |
| `web_search` | `boolean` | Enable web search grounding (Gemini) |
| `batch` | `boolean` | Route to asynchronous batch execution (Gemini) |
| `tools` | `LlmTool[]` | Function definitions available to the model |
| `tool_choice` | `LlmToolChoice` | `'auto'`, `'none'`, `'required'`, or `{ type: 'function', function: { name } }` |

### Reasoning per Provider

| Provider / Model | `none` Behavior | Effort Mapping & Constraints | Capability Configuration |
|---|---|---|---|
| **Gemini** | Thinking budget `0` (2.x), minimum budget `128` (2.5 Pro), level `minimal` (3.x, cannot be disabled) | Mapped via smart-genai to thinking budgets or levels; `xhigh`/`max` map to `high` | `reasoning: { supported: true }` |
| **DeepSeek** | Thinking disabled | Without `reasoning_effort`, DeepSeek thinking models default to `high` | `reasoning: { supported: true }` |
| **Groq (gpt-oss)** | Cannot be disabled | Supports `low`, `medium`, `high` | `reasoning: { supported: true, levels: ['low', 'medium', 'high'] }` |
| **Groq (Qwen)** | `none` (disabled) | Up to `high`. Set `extra: { reasoning_format: 'parsed' }` to receive reasoning in `data.reasoning` | `reasoning: { supported: true, levels: ['none', 'low', 'medium', 'high'] }` |
| **Groq (Muse)** | Cannot be disabled | Supports `low` to `xhigh` | `reasoning: { supported: true, levels: ['low', 'medium', 'high', 'xhigh'] }` |
| **Ollama** | Sets `think: false` | Passed directly; `gpt-oss` cannot disable reasoning | `reasoning: { supported: true }` |

Reasoning text returns in `data.reasoning`. Tokens are tracked in `usage.reasoningTokens` and are excluded from `usage.outputTokens`. Gemini returns thoughts whenever `reasoning_effort` is set; set `extra: { config: { thinkingConfig: { includeThoughts: false } } }` to omit thought text.

### Tool Calls

Define tools using standard function schemas. Returned calls appear in `data.toolCalls`. Continue the conversation by appending the assistant message and tool responses:

```ts
import { assistantMessage, toolMessage, type LlmMessage, type LlmTool } from '@smart-llm/core';

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

- `assistantMessage(output)`: Creates an `assistant` message from `data` or `error.output`. Preserves text, `tool_calls`, and `reasoning`. If the model returned only tool calls without text, `content` is set to `null`.
- `toolMessage(call, result)`: Formats a `tool` response message. Strings are passed as-is; other values are serialized to JSON (`undefined` becomes `'null'`).
- `tools`: Non-empty array of unique function declarations. Function names must be 1 to 64 characters matching `^[A-Za-z0-9_-]{1,64}$`. Invalid or duplicate names fail immediately with `invalid_request`.
- `tool_choice`: Dropped automatically when `tools` is omitted. Named function choices must match a function declared in `tools`.
- `response_format`: When tools are called, the intermediate tool call turn is not validated as JSON. Validation applies to the final text response following tool execution.

#### Provider Tool Nuances

- **Gemini**: Function calls and responses are mapped via smart-genai. Missing tool call IDs are automatically generated (`call_<uuid>`). Gemini 3 thought signatures are preserved in `toolCall.extra_content.google.thought_signature` and must remain in the message history. Combining `tools` with `web_search: true` is supported on Gemini 3+ and fails on Gemini 2.x. `extra_content` is stripped before sending to non-Gemini providers.
- **DeepSeek**: In thinking mode, assistant reasoning must be passed back as `reasoning_content` on tool turns (handled automatically by `assistantMessage`). In thinking mode, `tool_choice: 'required'` or named functions fail with HTTP 400.
- **Ollama**: Ollama does not support `tool_choice`. `tool_choice: 'none'` omits `tools` from the request; other values are stripped.
- **Capability Discovery**: `fetchOllamaCapabilities(options?)` queries `GET /api/tags` on the Ollama server, auto-detecting support for `images`, `tools`, `reasoning`, and `embeddings`. It returns a `Result` and never throws:

```ts
import { fetchOllamaCapabilities } from '@smart-llm/ollama';

const reported = await fetchOllamaCapabilities();
if (!reported.ok) throw new Error(reported.error.message);

const qwen = {
    model: 'qwen2.5:7b',
    capabilities: { ...reported.data['qwen2.5:7b'], json: true },
};
```

### Batch Calls

Gemini Batch API offers a 50% discount for non-interactive workloads, with results delivered within minutes to 24 hours. Batch calls use standard `generate` calls that resolve when the batch job completes:

```ts
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
```

- **Collection Window**: smart-genai groups batch requests arriving within `batchWaitSec` into a single batch job per model and Gemini `config`. Dispatch requests concurrently: `await Promise.all(items.map((item) => llm.flashBatch(item)))`.
- **Dedicated Queue**: Batch calls run through a dedicated `batchQueue` per model. They wait exclusively for `modelLimits.concurrency` (pending batch call slots). They bypass `sharedLimits`, rate limits (`rpm`, `tpm`, `intervalMs`), and queue locks from rate-limiting. Transient errors retry with backoff per `limiterOptions`.
- **Persistence**: Batch jobs are not persisted across process restarts. If the process terminates, pending promises are lost (though the job continues on Google Cloud). For durable batch jobs that survive restarts, use `smart-genai` directly.
- **Provider Support**: Supported exclusively by Gemini (requires a Cloud Storage bucket via `vertex.bucket` or env `VERTEX_BATCH_BUCKET` on Vertex AI). OpenAI-compatible adapters reject batch requests with `unsupported`.

### Embeddings

`llm.embed(request, options?)` generates vector embeddings for one or more texts from models declaring `embeddings: true`:

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

const paragraphs = await llm.embed({ model: 'bge', input: texts });
// paragraphs.data: { embeddings: number[][], name: 'bge', provider: 'ollama', model: 'bge-m3', usage: { inputTokens: 42 }, ms: 120, calls: 3, attempts: 3 }

const sections = await llm.embed({
    model: 'vec',
    input: [{ text: 'Cells were incubated at 37 °C.', title: 'Methods' }],
    task: 'document',
    dimensions: 768,
});
```

- **Automatic Chunking**: Inputs (`string | readonly (string | { text: string; title?: string })[]`, non-empty strings required) are split into batches of `maxInputs` (model setting, adapter `maxEmbedInputs`, or default `64`; Gemini supports `100`). Chunks execute through the model's `embedQueue`, sharing rate limits and backoff settings.
- **Vertex AI Per-Call Limit**: On the Gemini Developer API, all embedding models accept up to `100` texts per call. On Vertex AI, `gemini-embedding-001` accepts up to `100` texts per call, whereas `gemini-embedding-2` accepts `1` text per call; configure `capabilities: { embeddings: { supported: true, maxInputs: 1 } }` on Vertex AI so `llm.embed` splits batches into single-text calls automatically.
- **Fail-Fast**: If any chunk fails, remaining pending chunks are aborted immediately, and the call resolves with the failing error and total cumulative `attempts`.
- **Task Types & Titles**: `task` (`'query'`, `'document'`, `'similarity'`, `'classification'`, `'clustering'`) and `title` (valid only when `task: 'document'`) map to Gemini task types (`RETRIEVAL_QUERY`, `RETRIEVAL_DOCUMENT`, `SEMANTIC_SIMILARITY`, `CLASSIFICATION`, `CLUSTERING`). Ollama and OpenAI-compatible endpoints do not support tasks or titles and return `unsupported`.
- **Matryoshka Dimensions**: `dimensions` (positive integer) truncates vectors when supported by the model (Gemini, Ollama, OpenAI-compatible).
- **Token Reporting**: Gemini does not report token usage for embeddings (`inputTokens: null`). OpenAI-compatible endpoints sum `usage.prompt_tokens` across chunks.

### Images

Images must be supplied as base64 data URLs (`data:image/...;base64,...`) in `user` messages:

```ts
import { imageFromFile, imagePart } from '@smart-llm/core';

// From disk (auto-detects .png, .jpg, .jpeg, .webp, .gif; or pass explicit mimeType)
const photo = await imageFromFile('photo.jpg');

// Or from raw bytes (Uint8Array or ArrayBuffer) with optional detail ('auto' | 'low' | 'high')
const bufferPhoto = imagePart(bytes, 'image/png', 'high');

await llm.generate({
    model: 'flash',
    messages: [{ role: 'user', content: [{ type: 'text', text: 'What is this?' }, photo] }],
});
```

### Structured JSON Output

Pass a JSON Schema in `response_format`. The parsed and validated value is returned in `data.json`:

```ts
import type { LlmModelInput, LlmResponseFormat } from '@smart-llm/core';

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
}

// Reusable model call with fixed schema
const classify = (input: LlmModelInput) => llm.qwen(input, { response_format: word });
```

- **Supported Schema Subset**: `type` (`object`, `array`, `string`, `number`, `integer`, `boolean`, `null`, or array of types), `properties`, `required`, `additionalProperties`, `items`, `enum`, `const`, `anyOf`, and `description`. Metadata keys `$schema` and `$id` are automatically stripped.
- **Validation**: Surrounding markdown code fences (```` ```json ... ``` ````) are stripped before parsing. If the response is not valid JSON or violates the schema, the call fails with `code: 'invalid_json'`. The raw output is available in `error.output.text`. If the text was valid JSON but failed schema matching, `error.output.json` contains the parsed value.

#### JSON Modes

A model needs `capabilities: { json: true }` (or an explicit `LlmJsonMode`). Adapters use the following defaults:

| Mode | Default Adapter | Implementation Details |
|---|---|---|
| `native` | Gemini | Transmits cleaned schema directly via the provider's native schema parameter |
| `strict` | Groq | Converts schema to strict format (`toStrictSchema`: all properties required, optional fields made nullable, `additionalProperties: false`, `strict: true`) and strips `null` values on optional fields from the returned output (`removeOptionalNulls`) |
| `grammar` | Ollama | Compiles cleaned schema to engine grammar constraint and injects the schema with field descriptions into the system prompt |
| `prompt` | DeepSeek | Enforces `json_object` mode and injects schema instructions into the system prompt. Automatically retries (`retry_immediate`) if output fails JSON parsing or schema validation |

#### Exported Schema Utilities

`@smart-llm/core` exports the underlying schema transformations and validators for custom adapters and standalone use:

- `cleanSchema(schema)`: Returns a deep copy of `schema` with `$schema` and `$id` removed.
- `toStrictSchema(schema)`: Cleans `schema`, marks all object properties `required`, sets `additionalProperties: false`, and makes optional fields nullable.
- `removeOptionalNulls(value, schema)`: Recursively removes `null` values from optional object properties (resolving matching `anyOf` branches).
- `findSchemaMismatch(value, schema, path?)`: Validates `value` against `schema`; returns a path mismatch string (e.g. `'$.items[0].kind is not one of ["noun","verb"]'`) or `undefined`.
- `parseJsonText(text)`: Parses JSON string (stripping markdown code fences if present); returns `{ value }` or `undefined`.
- `schemaInstruction(schema?)`: Builds the system prompt instruction text for `prompt` and `grammar` modes.

## Responses & Error Handling

### Successful Output (`LlmOutput`)

Every successful `generate` call returns `{ ok: true, data }`:

| Field | Type | Description |
|---|---|---|
| `text` | `string` | Generated text response |
| `json` | `unknown` | Parsed and validated JSON value (present only for JSON `response_format`) |
| `reasoning` | `string` \| `undefined` | Thinking/reasoning text returned by the model |
| `sources` | `LlmSource[]` \| `undefined` | Web search citations grounding the answer (`{ url, title? }`) |
| `toolCalls` | `LlmToolCall[]` \| `undefined` | Tool calls requested by the model |
| `name` | `string` | Configured model name |
| `provider` | `string` | Configured provider name |
| `model` | `string` | API-reported model version or configured provider model id |
| `usage` | `LlmUsage` | `{ inputTokens, outputTokens, reasoningTokens }` (`number \| null`). `outputTokens` never includes reasoning tokens |
| `finishReason` | `string` \| `null` | Reason generation stopped (`'stop'`, `'tool_calls'`, `'length'`, etc.) |
| `ms` | `number` | Elapsed request execution time in milliseconds |
| `settings` | `LlmSettings` | Exact snapshot of resolved parameters sent to the provider |
| `attempts` | `number` | Total provider calls made (including retries for 429s or prompt-mode schema mismatches) |

#### Call Settings (`LlmSettings`)

`data.settings` and `error.settings` capture the resolved options used for the call: `name`, `provider`, `model`, `temperature`, `top_p`, `max_tokens`, `seed`, `stop`, `reasoning_effort` (mapped to model levels), `web_search` (`true` when active), `batch` (`true` when active), `jsonMode`, `tools` (array of function names), `tool_choice`, and merged `extra`.

### Error Output (`LlmError`)

Failed calls return `{ ok: false, error }`:

```ts
const result = await llm.flash('Classify "run"', { response_format: word });

if (!result.ok) {
    console.error(`Error [${result.error.code}]: ${result.error.message}`);
    if (result.error.code === 'invalid_json') {
        console.log('Raw text:', result.error.output?.text);
        console.log('Parsed JSON (if schema mismatch):', result.error.output?.json);
    }
}
```

- `code`: `LlmErrorCode` (see table below).
- `message`: Description of failure.
- `retryable`: Boolean indicating whether the error could succeed on retry.
- `output`: `LlmAnswer` containing the unusable response from the last attempt when a completion was received (`invalid_json`, `truncated`, `blocked`, or unexpected finish reason).
- `settings`: `LlmSettings` of the call (present on all `generate` errors occurring after request preparation).
- `attempts`: Number of provider calls attempted (`0` if failed before dispatch).
- Additional metadata when available: `status` (HTTP status), `retryAfterMs` (parsed `retry-after` delay in ms), `limiterCode` (smart-limit error code), `text`, `usage`, `model`, `ms`, `name`, `provider`.

| Error Code | Retryable | Description |
|---|---|---|
| `rate_limited` | yes | HTTP 429. Queue enters lock (`backoff_global`); retried after `retry-after` or backoff |
| `timeout` | yes | Request timed out (`retry_backoff`) |
| `network` | yes | Network or connection failure (`backoff_global`) |
| `http` | conditional | HTTP error. Retryable for 5xx (`backoff_global`), 408, 409, 425, 498 (Groq flex capacity), and Groq `tool_use_failed` (`retry_backoff`). Non-retryable for 400, 401, 403, 404, 422 |
| `aborted` | no | Call cancelled via caller `AbortSignal`. If cancelled before dispatch, no tokens are booked |
| `blocked` | no | Output blocked by provider safety or content filter. Carries `output` |
| `truncated` | no | Response hit output token limit (`max_tokens`). Carries `output` |
| `invalid_json` | conditional | Output is not valid JSON or violates schema (includes Groq `json_validate_failed`). Carries `output`. Retried automatically (`retry_immediate`) in `prompt` and `grammar` modes |
| `unsupported` | no | Request requires a capability the model lacks, or provider does not support an embedding option (`task` / `title`) |
| `invalid_request` | no | Invalid request payload or options detected at call time (unknown fields, unknown model name, invalid `extra`) |
| `limiter` | no | smart-limit rejected call before provider dispatch (`limiterCode`, e.g. `COST_EXCEEDS_LIMIT`, `MAX_WAIT_EXCEEDED`) |
| `unexpected` | conditional | Unhandled provider exception or unexpected API response structure (retryable on malformed tool call arguments or `insufficient_system_resource`) |

## Provider Adapters

### Gemini (`@smart-llm/gemini`)

`createGeminiProvider(clientOrOptions?)` accepts a `GenaiClient` instance from `smart-genai` or `Omit<GenaiClientOptions, 'defaults'>`:

| Option | Default | Description |
|---|---|---|
| `apiKey` | env `GEMINI_API_KEY` | Gemini Developer API key (AI Studio) |
| `vertex` | `undefined` | `true` or `{ project?, location?, bucket?, credentialsFile?, authClient?, storageUrl? }` for Vertex AI |
| `baseUrl` | official endpoint | Custom API endpoint URL |
| `env` | `process.env` | Environment map used for credential lookup |
| `timeoutMs` | `120_000` | HTTP request timeout in milliseconds (2 minutes) |
| `batchWaitSec` | `10` | Aggregation window in seconds for batch calls (`1` to `600`) |

### DeepSeek (`@smart-llm/deepseek`)

`createDeepseekProvider(options?: DeepseekOptions)` connects to the DeepSeek API:

| Option | Default | Description |
|---|---|---|
| `apiKey` | env `DEEPSEEK_API_KEY` | API key (`DEEPSEEK_API_KEY_ENV`); throws `TypeError` if missing |
| `env` | `process.env` | Environment map used for API key lookup |
| `baseUrl` | `'https://api.deepseek.com'` | API base URL |
| `timeoutMs` | `600_000` | Request timeout in milliseconds (10 minutes) |
| `fetch` | `globalThis.fetch` | Custom fetch implementation |

### Groq (`@smart-llm/groq`)

`createGroqProvider(options?: GroqOptions)` connects to the Groq API:

| Option | Default | Description |
|---|---|---|
| `apiKey` | env `GROQ_API_KEY` | API key (`GROQ_API_KEY_ENV`); throws `TypeError` if missing |
| `env` | `process.env` | Environment map used for API key lookup |
| `baseUrl` | `'https://api.groq.com/openai/v1'` | API base URL |
| `timeoutMs` | `120_000` | Request timeout in milliseconds (2 minutes) |
| `fetch` | `globalThis.fetch` | Custom fetch implementation |

### Ollama (`@smart-llm/ollama`)

`createOllamaProvider(options?: OllamaOptions)` connects to a local or remote Ollama server:

| Option | Default | Description |
|---|---|---|
| `baseUrl` | `'http://localhost:11434/v1'` | OpenAI-compatible Ollama endpoint URL |
| `apiKey` | `undefined` | Optional Bearer token for authenticated reverse proxies |
| `timeoutMs` | `600_000` | Request timeout in milliseconds (10 minutes) |
| `fetch` | `globalThis.fetch` | Custom fetch implementation |

`fetchOllamaCapabilities(options?: OllamaCapabilitiesOptions)` accepts `{ baseUrl?, timeoutMs?, signal?, fetch? }` (default `timeoutMs: 30_000`).

### Custom & OpenAI-Compatible Providers (`@smart-llm/openai-compatible`)

Use `createOpenAiCompatibleProvider` to connect any standard `/chat/completions` and `/embeddings` API:

```ts
import { createOpenAiCompatibleProvider } from '@smart-llm/openai-compatible';

const together = createOpenAiCompatibleProvider({
    name: 'together',
    baseUrl: 'https://api.together.xyz/v1',
    apiKey: process.env.TOGETHER_API_KEY,
    supports: { tools: true, json: 'prompt', embeddings: true }, // omit unsupported capabilities
});
```

#### `OpenAiCompatibleOptions`

| Option | Type | Default | Description |
|---|---|---|---|
| `name` | `string` | *required* | Provider name identifier |
| `baseUrl` | `string` | *required* | Base URL (e.g. `https://api.together.xyz/v1`) |
| `apiKey` | `string` | `undefined` | Optional Bearer token |
| `headers` | `Record<string, string>` | `undefined` | Custom request headers |
| `supports` | `Omit<LlmProviderSupport, 'webSearch' \| 'batch'>` | `{}` | Supported capabilities (`images`, `tools`, `reasoning`, `json`, `embeddings`) |
| `maxTokensField` | `'max_tokens'` \| `'max_completion_tokens'` | `'max_completion_tokens'` | Body field name for output token limits |
| `mapBody` | `(body, request) => body` | `undefined` | Optional request body transformation hook |
| `reasoningFields` | `readonly string[]` | `['reasoning_content', 'reasoning']` | Message fields inspected for reasoning text |
| `reasoningMessageField` | `string` | `undefined` | Assistant message property used to send reasoning history back |
| `timeoutMs` | `number` | `120_000` | Request timeout in milliseconds (2 minutes) |
| `fetch` | `typeof fetch` | `globalThis.fetch` | Custom fetch implementation |

`@smart-llm/openai-compatible` also exports HTTP helpers for custom adapter implementations:
- `fetchJson(fetcher, url, init)`: Executes an HTTP request, parses JSON on 2xx, and maps status codes, `retry-after` headers, `json_validate_failed`, `tool_use_failed`, timeouts, aborts, and network errors to `Result<unknown, LlmError>`.
- `statusError(status, message, retryAfter)`: Maps an HTTP status code and `retry-after` header value to `LlmError`.

### Custom Adapter Contract (`LlmProvider`)

Custom adapters implement the `LlmProvider` interface:

- `name`: Provider name string.
- `supports`: `LlmProviderSupport` declaring supported features (`images`, `tools`, `webSearch`, `reasoning`, `batch`, `json`, `embeddings`).
- `generate(request, { signal, extra })`: Dispatches a single API request and maps responses to `Result<LlmProviderOutput, LlmProviderError>`. Must never throw or reject (unexpected throws are caught and mapped to `unexpected`).
- `listModels?(options?)`: Queries available models (`Result<LlmModelInfo[], LlmError>`).
- `validateExtra?(extra, option)`: Validates provider-specific `extra` options; throws `TypeError` on invalid options.
- `mergeExtra?(layers)`: Custom merge logic for provider, model, and call `extra` objects.
- `embed?(request, { signal })`: Dispatches a single vector embedding request for up to `maxEmbedInputs` texts (`Result<LlmProviderEmbedOutput, LlmProviderError>`). Required when `supports.embeddings` is `true`.
- `maxEmbedInputs?`: Maximum texts permitted in a single embedding call (positive integer; defaults to `64` when omitted).
