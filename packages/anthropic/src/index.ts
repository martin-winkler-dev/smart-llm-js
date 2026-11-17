// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { Anthropic } from '@anthropic-ai/sdk';
import type { LlmModelInfo, LlmProvider, LlmProviderSupport, LlmReasoningEffort } from '@smart-llm/core';
import { createBatchCalls } from './batch.js';
import { isPlainObject, mergeDeep, toMessageParams } from './request.js';
import { readMessage, toProviderError } from './response.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/**
 * **Anthropic provider options**.
 *
 * - `apiKey`: API key (defaults to env `ANTHROPIC_API_KEY`)
 * - `env`: environment lookup map (defaults to `process.env`)
 * - `baseUrl`: optional custom API base URL
 * - `headers`: additional request headers (e.g. `anthropic-beta`)
 * - `timeoutMs`: request timeout in milliseconds (defaults to 10 min)
 * - `fetch`: optional custom fetch implementation
 * - `batchWaitSec`: batch collection window in seconds, 1 to 600 (defaults to 10)
 * - `defaultMaxTokens`: output limit for requests without `max_tokens` (defaults to 16,000)
 */
export type AnthropicOptions = {
    apiKey?: string;
    env?: Readonly<Record<string, string | undefined>>;
    baseUrl?: string;
    headers?: Record<string, string>;
    timeoutMs?: number;
    fetch?: typeof fetch;
    batchWaitSec?: number;
    defaultMaxTokens?: number;
};

/**
 * **Anthropic extra options**: provider, model, or call `extra`.
 *
 * - `params`: deep-merged over built Messages params last (`null` deletes key)
 * - `thinkingBudgets`: manual thinking budget per reasoning effort (integers >= 1024) instead of adaptive thinking
 * - `thinkingDisplay`: thinking display (defaults to `summarized`)
 * - `thinkingOff`: thinking sent for reasoning effort `none` (defaults to `disabled`; `omit` sends nothing)
 * - `webSearchTool`: merged into web search tool definition (e.g. `max_uses`, `allowed_domains`)
 */
export type AnthropicExtra = {
    params?: Record<string, unknown>;
    thinkingBudgets?: Partial<Record<Exclude<LlmReasoningEffort, 'none'>, number>>;
    thinkingDisplay?: 'summarized' | 'omitted';
    thinkingOff?: 'disabled' | 'between_tools' | 'omit';
    webSearchTool?: Record<string, unknown>;
};

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Environment variable name for Anthropic API key. */
export const ANTHROPIC_API_KEY_ENV = 'ANTHROPIC_API_KEY';

/** Default output limit for requests without `max_tokens`. */
export const DEFAULT_MAX_TOKENS = 16_000;

/** Capabilities supported by Anthropic: images, tools, web search, reasoning, batch, and native JSON. */
export const ANTHROPIC_SUPPORT: LlmProviderSupport = {
    images: true,
    tools: true,
    webSearch: true,
    reasoning: true,
    batch: true,
    json: 'native',
};

/** Default request timeout in milliseconds. */
const DEFAULT_TIMEOUT_MS = 600_000;

/** Default batch collection window in seconds. */
const DEFAULT_BATCH_WAIT_SEC = 10;

/** Shortest batch collection window in seconds. */
const MIN_BATCH_WAIT_SEC = 1;

/** Longest batch collection window in seconds. */
const MAX_BATCH_WAIT_SEC = 600;

/** Smallest thinking budget accepted by Anthropic. */
const MIN_THINKING_BUDGET = 1024;

/** Model listing page size (API maximum). */
const MODELS_PAGE_SIZE = 1000;

/** Supported provider option keys. */
const OPTION_KEYS: ReadonlySet<string> = new Set([
    'apiKey',
    'env',
    'baseUrl',
    'headers',
    'timeoutMs',
    'fetch',
    'batchWaitSec',
    'defaultMaxTokens',
]);

/** Supported extra keys. */
const EXTRA_KEYS = ['params', 'thinkingBudgets', 'thinkingDisplay', 'thinkingOff', 'webSearchTool'] as const;

/** Extra keys deep-merged across layers. */
const DEEP_EXTRA_KEYS: ReadonlySet<string> = new Set(['params', 'thinkingBudgets', 'webSearchTool']);

/** Request fields that `extra.params` may not set. */
const REQUEST_PARAM_KEYS = ['model', 'messages', 'stream'] as const;

/** Reasoning efforts accepting thinking budget. */
const THINKING_EFFORTS: readonly string[] = ['minimal', 'low', 'medium', 'high', 'xhigh', 'max'];

/** Values of `extra.thinkingDisplay`. */
const THINKING_DISPLAYS: readonly unknown[] = ['summarized', 'omitted'];

/** Values of `extra.thinkingOff`. */
const THINKING_OFF_VALUES: readonly unknown[] = ['disabled', 'between_tools', 'omit'];

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Provider adapter for Anthropic Messages API** via official `@anthropic-ai/sdk`.
 *
 * - supports images, tools, web search, reasoning (adaptive or budget thinking), batch calls, and native JSON
 * - takes SDK client instance (its positive `timeout` per request, else 10 min) or options (SDK retries off: core retries)
 * - replays assistant tool turns verbatim via first tool call `extra_content.anthropic.content`; thinking only in current tool loop
 * - no embeddings (Anthropic has no embeddings endpoint); non-streaming only
 * - never throws from `generate`/`listModels`; failures returned as Result
 * @throws {TypeError} If options are unsupported or invalid, or API key is missing.
 */
export function createAnthropicProvider(client: Anthropic | AnthropicOptions = {}): LlmProvider {
    const options: AnthropicOptions = isClient(client) ? {} : client;
    const anthropic = isClient(client) ? client : toClient(options);
    const timeoutMs = isClient(client) ? toClientTimeout(client) : (options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
    const defaultMaxTokens = options.defaultMaxTokens ?? DEFAULT_MAX_TOKENS;
    const batch = createBatchCalls(anthropic, options.batchWaitSec ?? DEFAULT_BATCH_WAIT_SEC, timeoutMs);
    return {
        name: 'anthropic',
        supports: ANTHROPIC_SUPPORT,
        validateExtra: validateAnthropicExtra,
        mergeExtra: mergeAnthropicExtra,
        async generate(request, { signal, extra }) {
            const startedAt = performance.now();
            try {
                const params = toMessageParams(request, extra as AnthropicExtra, defaultMaxTokens);
                if (!params.ok) {
                    return params;
                }
                if (request.batch === true) {
                    return await batch.add(params.data, signal);
                }
                const message = await anthropic.messages.create(params.data, {
                    signal,
                    timeout: timeoutMs,
                    maxRetries: 0,
                });
                return readMessage(message, request.model, performance.now() - startedAt);
            } catch (error) {
                return { ok: false, error: toProviderError(error, signal) };
            }
        },
        async listModels(listOptions = {}) {
            try {
                const models: LlmModelInfo[] = [];
                const pages = anthropic.models.list(
                    { limit: MODELS_PAGE_SIZE },
                    { signal: listOptions.signal, timeout: timeoutMs, maxRetries: 0 },
                );
                for await (const model of pages) {
                    models.push({
                        id: model.id,
                        ...(typeof model.display_name === 'string' ? { displayName: model.display_name } : {}),
                        ...(typeof model.max_input_tokens === 'number'
                            ? { inputTokenLimit: model.max_input_tokens }
                            : {}),
                        ...(typeof model.max_tokens === 'number' ? { outputTokenLimit: model.max_tokens } : {}),
                    });
                }
                return { ok: true, data: models };
            } catch (error) {
                const { output: _output, ...failed } = toProviderError(error, listOptions.signal);
                return { ok: false, error: failed };
            }
        },
    };
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/** Checks whether value is SDK client (has `messages.create`). */
function isClient(value: unknown): value is Anthropic {
    const messages = (value as { messages?: { create?: unknown } } | null | undefined)?.messages;
    return typeof messages?.create === 'function';
}

/**
 * Validates options and creates SDK client (SDK retries off).
 * @throws {TypeError} If options are unsupported or invalid, or API key is missing.
 */
function toClient(options: AnthropicOptions): Anthropic {
    if (!isPlainObject(options)) {
        throw new TypeError('Invalid value for option "options"');
    }
    for (const key of Object.keys(options)) {
        if (!OPTION_KEYS.has(key)) {
            throw new TypeError(`Option "${key}" is not supported by the Anthropic provider`);
        }
    }
    if (
        options.env !== undefined &&
        !(typeof options.env === 'object' && options.env !== null && !Array.isArray(options.env))
    ) {
        throw new TypeError('Invalid value for option "env": an object');
    }
    if (options.apiKey !== undefined && typeof options.apiKey !== 'string') {
        throw new TypeError('Invalid value for option "apiKey": a string');
    }
    const env = options.env ?? process.env;
    const apiKey = options.apiKey ?? env[ANTHROPIC_API_KEY_ENV];
    if (typeof apiKey !== 'string' || apiKey === '') {
        throw new TypeError(`Invalid value for option "apiKey": set it or env ${ANTHROPIC_API_KEY_ENV}`);
    }
    if (options.baseUrl !== undefined && !isHttpUrl(options.baseUrl)) {
        throw new TypeError('Invalid value for option "baseUrl": an absolute http(s) URL');
    }
    if (
        options.headers !== undefined &&
        !(isPlainObject(options.headers) && Object.values(options.headers).every((value) => typeof value === 'string'))
    ) {
        throw new TypeError('Invalid value for option "headers": a plain object of strings');
    }
    if (options.fetch !== undefined && typeof options.fetch !== 'function') {
        throw new TypeError('Invalid value for option "fetch": a function');
    }
    if (options.timeoutMs !== undefined && !isPositiveInteger(options.timeoutMs)) {
        throw new TypeError('Invalid value for option "timeoutMs": a positive integer');
    }
    const { batchWaitSec } = options;
    if (
        batchWaitSec !== undefined &&
        !(Number.isInteger(batchWaitSec) && batchWaitSec >= MIN_BATCH_WAIT_SEC && batchWaitSec <= MAX_BATCH_WAIT_SEC)
    ) {
        throw new TypeError(
            `Invalid value for option "batchWaitSec": an integer from ${MIN_BATCH_WAIT_SEC} to ${MAX_BATCH_WAIT_SEC}`,
        );
    }
    if (options.defaultMaxTokens !== undefined && !isPositiveInteger(options.defaultMaxTokens)) {
        throw new TypeError('Invalid value for option "defaultMaxTokens": a positive integer');
    }
    return new Anthropic({
        apiKey,
        baseURL: options.baseUrl,
        defaultHeaders: options.headers,
        fetch: options.fetch,
        maxRetries: 0,
        timeout: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
    });
}

/** Returns client `timeout` when positive number (duck-typed wrappers may lack it), else default timeout. */
function toClientTimeout(client: Anthropic): number {
    const { timeout } = client as { timeout?: unknown };
    return typeof timeout === 'number' && Number.isFinite(timeout) && timeout > 0 ? timeout : DEFAULT_TIMEOUT_MS;
}

/** Checks whether value is absolute URL with `http:` or `https:` scheme. */
function isHttpUrl(value: unknown): boolean {
    if (typeof value !== 'string' || !URL.canParse(value)) {
        return false;
    }
    const { protocol } = new URL(value);
    return protocol === 'http:' || protocol === 'https:';
}

/** Checks whether value is positive integer. */
function isPositiveInteger(value: unknown): boolean {
    return Number.isInteger(value) && (value as number) > 0;
}

/**
 * **Validates Anthropic extra options**.
 *
 * - keys: params, thinkingBudgets, thinkingDisplay, thinkingOff, webSearchTool
 * - `params`: plain object without model, messages, stream
 * - `thinkingBudgets`: plain object of effort levels (minimal..max) with integers >= 1024
 * - `webSearchTool`: plain object
 * @throws {TypeError} If keys are unsupported or values invalid.
 */
function validateAnthropicExtra(extra: Record<string, unknown>, option: string): void {
    for (const key of Object.keys(extra)) {
        if (!(EXTRA_KEYS as readonly string[]).includes(key)) {
            throw new TypeError(
                `Option "${option}.${key}" is not supported by the Anthropic provider (only ${EXTRA_KEYS.join(', ')})`,
            );
        }
    }
    const { params, thinkingBudgets, thinkingDisplay, thinkingOff, webSearchTool } = extra;
    if (params !== undefined) {
        if (!isPlainObject(params)) {
            throw new TypeError(`Invalid value for option "${option}.params": a plain object`);
        }
        const fixed = REQUEST_PARAM_KEYS.find((key) => Object.hasOwn(params, key));
        if (fixed !== undefined) {
            throw new TypeError(
                `Option "${option}.params.${fixed}" is not supported by the Anthropic provider (set by the request)`,
            );
        }
    }
    if (thinkingBudgets !== undefined) {
        validateThinkingBudgets(thinkingBudgets, `${option}.thinkingBudgets`);
    }
    if (thinkingDisplay !== undefined && !THINKING_DISPLAYS.includes(thinkingDisplay)) {
        throw new TypeError(`Invalid value for option "${option}.thinkingDisplay": summarized or omitted`);
    }
    if (thinkingOff !== undefined && !THINKING_OFF_VALUES.includes(thinkingOff)) {
        throw new TypeError(`Invalid value for option "${option}.thinkingOff": disabled, between_tools or omit`);
    }
    if (webSearchTool !== undefined && !isPlainObject(webSearchTool)) {
        throw new TypeError(`Invalid value for option "${option}.webSearchTool": a plain object`);
    }
}

/**
 * Validates thinking budgets: effort level keys with integers of at least 1024.
 * @throws {TypeError} If not plain object, key unsupported, or budget invalid.
 */
function validateThinkingBudgets(budgets: unknown, option: string): void {
    if (!isPlainObject(budgets)) {
        throw new TypeError(`Invalid value for option "${option}": a plain object`);
    }
    for (const [key, budget] of Object.entries(budgets)) {
        if (!THINKING_EFFORTS.includes(key)) {
            throw new TypeError(`Option "${option}.${key}" is not supported (only ${THINKING_EFFORTS.join(', ')})`);
        }
        if (budget !== undefined && !(Number.isInteger(budget) && (budget as number) >= MIN_THINKING_BUDGET)) {
            throw new TypeError(
                `Invalid value for option "${option}.${key}": an integer of at least ${MIN_THINKING_BUDGET}`,
            );
        }
    }
}

/** Merges extra layers (later wins); params, thinkingBudgets, webSearchTool deep-merged. */
function mergeAnthropicExtra(layers: Record<string, unknown>[]): Record<string, unknown> {
    const merged: Record<string, unknown> = {};
    for (const layer of layers) {
        for (const [key, value] of Object.entries(layer)) {
            if (value === undefined || key === '__proto__') {
                continue;
            }
            const current = merged[key];
            merged[key] =
                DEEP_EXTRA_KEYS.has(key) && isPlainObject(value) && isPlainObject(current)
                    ? mergeDeep(current, value, false)
                    : value;
        }
    }
    return merged;
}
