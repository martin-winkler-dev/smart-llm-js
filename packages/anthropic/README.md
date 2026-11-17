# @smart-llm/anthropic

[![Test](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/test.yml)
[![Package](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-llm-js/actions/workflows/package.yml)
[![Bun](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fanthropic%2Fpackage.json&query=%24.engines.bun&label=Bun&logo=bun&color=blue)](https://bun.sh)
[![Node.js](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-llm-js%2Fmain%2Fpackages%2Fanthropic%2Fpackage.json&query=%24.engines.node&label=Node.js&logo=nodedotjs&color=blue)](https://nodejs.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Anthropic Claude adapter for [smart-llm](https://github.com/martin-winkler-dev/smart-llm-js#readme): connects the Anthropic Messages API through [`@anthropic-ai/sdk`](https://www.npmjs.com/package/@anthropic-ai/sdk).

## Install

```bash
bun add @smart-llm/anthropic @smart-llm/core
# or
npm install @smart-llm/anthropic @smart-llm/core
```

ESM package with TypeScript types for Node.js >= 22.12 (`require()` supported) and Bun >= 1.3.

## Usage

```ts
import { createLlm } from '@smart-llm/core';
import { createAnthropicProvider } from '@smart-llm/anthropic';

const llm = createLlm({
    providers: {
        anthropic: {
            adapter: createAnthropicProvider(),           // env ANTHROPIC_API_KEY
            defaults: { max_tokens: 16_000 },             // unset: adapter sends defaultMaxTokens (16_000)
            models: {
                opus: {
                    model: 'claude-opus-5-5',
                    capabilities: {
                        images: true,
                        tools: true,
                        json: true,
                        webSearch: { supported: true },
                        batch: { supported: true },
                        reasoning: { supported: true, default: 'medium', levels: ['low', 'medium', 'high', 'xhigh', 'max'] },
                    },
                },
                sonnet: {
                    model: 'claude-sonnet-5-5',
                    capabilities: {
                        images: true,
                        tools: true,
                        json: true,
                        reasoning: { supported: true, levels: ['none', 'low', 'medium', 'high', 'xhigh', 'max'] },
                    },
                    extra: { thinkingOff: 'between_tools' },
                },
                haiku: {
                    model: 'claude-haiku-4-5',
                    capabilities: {
                        images: true,
                        tools: true,
                        json: true,
                        reasoning: { supported: true, levels: ['none', 'low', 'medium', 'high'] },
                    },
                    extra: { thinkingBudgets: { low: 1_024, medium: 4_096, high: 12_000 } },
                },
            },
        },
    },
});

const answer = await llm.opus('Summarize quantum tunneling in two sentences.', { web_search: true });
```

## Options

`createAnthropicProvider(clientOrOptions?: Anthropic | AnthropicOptions)` accepts an existing `Anthropic` SDK client or `AnthropicOptions` and throws `TypeError` on invalid options or unknown keys. With an SDK client, the client's positive `timeout` (else `600_000` ms) applies and `batchWaitSec` and `defaultMaxTokens` keep their defaults.

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

SDK retries are disabled (`maxRetries: 0`, also per request on a passed client); `@smart-llm/core` handles retries and rate limits.

## Extra Options

`extra` (provider, model, or call layer) accepts `AnthropicExtra`. `params`, `thinkingBudgets`, and `webSearchTool` deep-merge across layers; later layers replace `thinkingDisplay` and `thinkingOff`. Invalid keys or types throw `TypeError` at setup or return `invalid_request` per call.

| Key | Type | Description |
|---|---|---|
| `params` | `Record<string, unknown>` | Raw Messages API fields deep-merged last over the built request (`null` deletes a key; `model`, `messages`, and `stream` are rejected), e.g. `{ cache_control: { type: 'ephemeral' } }`, `metadata`, `service_tier`, `inference_geo` |
| `thinkingBudgets` | `Partial<Record<Exclude<LlmReasoningEffort, 'none'>, number>>` | Token budget per effort (`minimal` to `max`; integer `>= 1024` and `< max_tokens`); sends `thinking: { type: 'enabled', budget_tokens, display }` instead of adaptive thinking and `effort`. A requested effort without a budget, or a budget `>= max_tokens`, fails with `invalid_request` |
| `thinkingDisplay` | `'summarized'` \| `'omitted'` | Thinking display for adaptive and budget thinking; `'summarized'` (default) fills `data.reasoning` |
| `thinkingOff` | `'disabled'` \| `'between_tools'` \| `'omit'` | Sent for `reasoning_effort: 'none'`: `thinking: { type: 'disabled' }` (`'disabled'`, default), `thinking: { type: 'between_tools' }` (`'between_tools'`), or no `thinking` field (`'omit'`) |
| `webSearchTool` | `Record<string, unknown>` | Merged into `{ type: 'web_search_20260209', name: 'web_search' }`, e.g. `{ max_uses: 3, allowed_domains: [...] }` or `{ type: 'web_search_20250305' }` |

## Capabilities

`ANTHROPIC_SUPPORT`: `{ images: true, tools: true, webSearch: true, reasoning: true, batch: true, json: 'native' }`.

- **Messages**: `system` and `developer` messages at any position go to the top-level `system` field. `tool` messages become `tool_result` blocks in `user` turns; consecutive turns of the same role merge. Empty text parts are dropped; empty assistant messages without `tool_calls` are omitted (and surrounding turns merge), while a `user` turn left empty fails with `invalid_request`. `temperature`, `top_p`, and `seed` are not sent. Responses are not streamed.
- **Images**: base64 data URLs for JPEG, PNG, GIF, and WebP; other formats or URLs fail with `invalid_request`. `detail` is not sent.
- **Tools**: `parameters` is sent as `input_schema` (`{ type: 'object', properties: {} }` when omitted; must be an object schema). `strict: true` sets the tool's `strict` flag and applies the schema rules of JSON (below). `tool_choice` sends `'required'` as `{ type: 'any' }`, `'auto'` as `{ type: 'auto' }`, `'none'` as `{ type: 'none' }`, and a named function as `{ type: 'tool', name }`.
- **Assistant replay**: raw content blocks of answers with tool calls (thinking with signatures, server tool results, citations, text, `tool_use`) are stored on `toolCalls[0].extra_content.anthropic.content`. `assistantMessage(output)` keeps them, and the adapter sends them unchanged (without empty text blocks) when their `tool_use` ids match the tool calls; otherwise the turn is rebuilt from text and `tool_use` without thinking. Thinking blocks are kept for the current tool loop and stripped from earlier turns.
- **JSON** (`native`): requires an object root schema (`type: 'object'` or omitted). Unsupported keywords (e.g. `minLength`) move into `description`; objects get `additionalProperties: false`; `oneOf` becomes `anyOf`; `$ref` nodes drop sibling keywords other than `$defs` / `definitions`. Open objects (`additionalProperties` other than `false`), boolean sub-schemas, tuple `items`, and nodes without `type`, `anyOf`, `oneOf`, `allOf`, `$ref`, `enum`, or `const` fail with `invalid_request`. `json_object` and `json_schema` without `schema` append a JSON instruction to `system`. For models without native structured output, declare `capabilities: { json: 'prompt' }`.
- **Reasoning**: `reasoning_effort` is sent as adaptive thinking with output `effort` (`minimal` is sent as `low`), or with `thinkingBudgets` as budget thinking; `none` follows `thinkingOff`. Thinking text returns in `data.reasoning`; thinking tokens are deducted from `usage.outputTokens` (clamped at `0`) into `usage.reasoningTokens` (`null` when not reported). With `thinkingBudgets`, declare `capabilities.reasoning.levels` matching the budget keys (plus `'none'` if disabling thinking is allowed).
- **Web search**: adds the `web_search` server tool and fills `data.sources` from web search citations (else from search results), deduplicated by `url`.
- **Batch**: `batch: true` collects requests of all models for `batchWaitSec` into Message Batches (at most `100_000` requests and `250_000_000` bytes per batch; a single larger request fails with `invalid_request`) and polls every 60 s for up to 25 hours. Aborting every call of a batch cancels it.
- **Embeddings**: not supported; `embed` returns `unsupported`.
- **Usage**: `usage.inputTokens` includes cache creation and cache read tokens.
- **Model listing**: `llm.listModels('anthropic')` returns `{ id, displayName?, inputTokenLimit?, outputTokenLimit? }[]`.

## Errors

| Code | Retryable | When |
|---|---|---|
| `rate_limited` | yes | HTTP 429 (`retry-after-ms` / `retry-after` honored); batch item error `rate_limit_error` |
| `http` | yes | HTTP 408, 409, or 5xx; batch item errors `overloaded_error`, `api_error`, `timeout_error` |
| `http` | no | Other HTTP statuses and batch item errors |
| `network` | yes | Connection failure or transient batch results stream error |
| `truncated` | no | Stop reason `max_tokens` or `model_context_window_exceeded`; carries `error.output` |
| `blocked` | no | Stop reason `refusal`; carries `error.output` |
| `timeout` | yes | Request timeout (`timeoutMs`) or expired batch item |
| `timeout` | no | Batch not finished after 25 hours (the batch is canceled) |
| `aborted` | no | Caller `signal` aborted; canceled batch item |
| `unexpected` | yes | 2xx body that is not a message, `tool_use` stop without calls (carries `error.output`), or a `tool_use` input that is not an object (carries `error.output`) |
| `unexpected` | no | Stop reason `pause_turn` or another unknown stop reason (carries `error.output`), ended batch without `results_url`, 3 consecutive batch download failures, missing/unknown batch item result, or non-SDK error |
| `invalid_request` | no | Invalid `extra`, image, schema, `thinkingBudgets` mismatch/overflow, history tool call arguments, batch request `> 250_000_000` bytes, or empty `user` turn |
| `unsupported` | no | `embed` |

## Exports

| Export | Kind | Description |
|---|---|---|
| `createAnthropicProvider` | function | Creates the adapter |
| `ANTHROPIC_SUPPORT` | constant | `LlmProviderSupport` of the adapter |
| `ANTHROPIC_API_KEY_ENV` | constant | `'ANTHROPIC_API_KEY'` |
| `DEFAULT_MAX_TOKENS` | constant | `16_000`, default of `defaultMaxTokens` |
| `AnthropicOptions` | type | Options of `createAnthropicProvider` |
| `AnthropicExtra` | type | `extra` accepted by the adapter |

Configuration, requests, rate limits, and error handling of `@smart-llm/core` are documented in the [smart-llm README](https://github.com/martin-winkler-dev/smart-llm-js#readme).

## License

Apache-2.0, see [LICENSE](LICENSE).
