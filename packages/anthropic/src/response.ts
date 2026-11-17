// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import {
    type Anthropic,
    AnthropicError,
    APIConnectionError,
    APIConnectionTimeoutError,
    APIError,
    APIUserAbortError,
} from '@anthropic-ai/sdk';
import type { LlmProviderError, LlmProviderOutput, LlmSource, LlmToolCall, LlmUsage, Result } from '@smart-llm/core';
import { isPlainObject } from './request.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Stop reasons of completed usable answer (`null` too). */
const DONE_REASONS: ReadonlySet<string> = new Set(['end_turn', 'stop_sequence']);

/** Retryable HTTP status codes below 500. */
const RETRYABLE_STATUSES: ReadonlySet<number> = new Set([408, 409]);

/** HTTP status code for rate limiting. */
const RATE_LIMITED_STATUS = 429;

/** Lowest HTTP status code for server errors. */
const MIN_SERVER_ERROR_STATUS = 500;

// -------------------------------------------------------------------
// --- 2. Exported Functions
// -------------------------------------------------------------------

/**
 * **Reads Messages API answer into provider output or stop reason error**.
 *
 * - `text`: all text blocks; `reasoning`: non-empty thinking texts joined by blank line
 * - `toolCalls`: `tool_use` blocks; first call carries all raw content blocks in `extra_content.anthropic.content`
 * - `sources`: web search citations, else web search results (deduped by url)
 * - `usage`: input includes cache tokens; output excludes thinking tokens
 * - failures (`truncated`, `blocked`, `unexpected`) carry answer as `output`
 * - no message object with `content` array of objects (e.g. proxy page, empty 2xx) → retryable `unexpected`
 * - `tool_use` input not plain object → retryable `unexpected`
 */
export function readMessage(
    message: Anthropic.Message,
    requestedModel: string,
    ms: number,
): Result<LlmProviderOutput, LlmProviderError> {
    const raw: unknown = message;
    if (!(isPlainObject(raw) && Array.isArray(raw.content) && raw.content.every(isPlainObject))) {
        return {
            ok: false,
            error: { code: 'unexpected', message: 'The response is not an Anthropic message', retryable: true },
        };
    }
    const blocks: Anthropic.ContentBlock[] = message.content;
    const output: LlmProviderOutput = {
        text: blocks.map((block) => (block.type === 'text' ? block.text : '')).join(''),
        model: message.model || requestedModel,
        usage: readUsage(message.usage),
        finishReason: message.stop_reason ?? null,
        ms: Math.round(ms),
    };
    const reasoning = blocks
        .flatMap((block) => (block.type === 'thinking' && block.thinking !== '' ? [block.thinking] : []))
        .join('\n\n');
    if (reasoning !== '') {
        output.reasoning = reasoning;
    }
    const toolCalls = readToolCalls(blocks);
    if (toolCalls.length > 0) {
        output.toolCalls = toolCalls;
    }
    const sources = readSources(blocks);
    if (sources.length > 0) {
        output.sources = sources;
    }
    const reason = output.finishReason;
    if (reason === null || DONE_REASONS.has(reason) || (reason === 'tool_use' && toolCalls.length > 0)) {
        const broken = blocks.find(
            (block): block is Anthropic.ToolUseBlock =>
                block.type === 'tool_use' && block.input != null && !isPlainObject(block.input),
        );
        if (broken !== undefined) {
            return failure(
                output,
                'unexpected',
                `The arguments of tool call ${broken.name} are not a JSON object`,
                true,
            );
        }
        return { ok: true, data: output };
    }
    switch (reason) {
        case 'tool_use':
            return failure(output, 'unexpected', 'Stop reason tool_use without a tool call', true);
        case 'max_tokens':
            return failure(output, 'truncated', 'The answer hit the output token limit');
        case 'model_context_window_exceeded':
            return failure(output, 'truncated', 'The conversation exceeds the context window');
        case 'refusal':
            return failure(output, 'blocked', refusalMessage(message.stop_details));
        case 'pause_turn':
            return failure(
                output,
                'unexpected',
                'The turn paused (pause_turn): continuing a paused server tool turn is not supported',
            );
        default:
            return failure(output, 'unexpected', `Stop reason ${reason}`);
    }
}

/**
 * **Maps thrown SDK or other error to provider error**.
 *
 * - user abort or aborted `signal` → `aborted`; connection timeout → `timeout`; connection error → `network`
 * - API error with status → `HTTP <status>: <api message>`; 429 `rate_limited`; 408, 409, >= 500 retryable `http`
 * - `retryAfterMs` from `retry-after-ms` or `retry-after` header
 * - other SDK errors and TypeError → `invalid_request`; anything else → `unexpected`
 */
export function toProviderError(error: unknown, signal?: AbortSignal): LlmProviderError {
    const message = error instanceof Error ? error.message : String(error);
    if (error instanceof APIUserAbortError || signal?.aborted) {
        return { code: 'aborted', message, retryable: false };
    }
    if (error instanceof APIConnectionTimeoutError) {
        return { code: 'timeout', message, retryable: true };
    }
    if (error instanceof APIConnectionError) {
        return { code: 'network', message, retryable: true };
    }
    if (error instanceof APIError && typeof error.status === 'number') {
        return statusError(error);
    }
    if (error instanceof AnthropicError || error instanceof TypeError) {
        return { code: 'invalid_request', message, retryable: false };
    }
    return { code: 'unexpected', message, retryable: false };
}

// -------------------------------------------------------------------
// --- 3. Private Functions
// -------------------------------------------------------------------

/** Returns failure carrying unusable answer as `output`. */
function failure(
    output: LlmProviderOutput,
    code: LlmProviderError['code'],
    message: string,
    retryable = false,
): { ok: false; error: LlmProviderError } {
    const { text, model, usage, ms } = output;
    return { ok: false, error: { code, message, retryable, text, model, usage, ms, output } };
}

/** Reads token usage; input includes cache tokens, output excludes thinking tokens. */
function readUsage(usage: Anthropic.Usage | undefined): LlmUsage {
    const thinking = usage?.output_tokens_details?.thinking_tokens ?? null;
    const input = usage?.input_tokens;
    const output = usage?.output_tokens;
    return {
        inputTokens:
            typeof input === 'number'
                ? input + (usage?.cache_creation_input_tokens ?? 0) + (usage?.cache_read_input_tokens ?? 0)
                : null,
        outputTokens: typeof output === 'number' ? Math.max(0, output - (thinking ?? 0)) : null,
        reasoningTokens: thinking,
    };
}

/** Maps `tool_use` blocks to tool calls; first call carries all raw content blocks for verbatim replay. */
function readToolCalls(blocks: readonly Anthropic.ContentBlock[]): LlmToolCall[] {
    const calls: LlmToolCall[] = blocks.flatMap((block) =>
        block.type === 'tool_use'
            ? [
                  {
                      id: block.id,
                      type: 'function' as const,
                      function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) },
                  },
              ]
            : [],
    );
    const [first] = calls;
    if (first !== undefined) {
        first.extra_content = { anthropic: { content: [...blocks] } };
    }
    return calls;
}

/** Reads web sources from text citations, else from web search results; deduped by url. */
function readSources(blocks: readonly Anthropic.ContentBlock[]): LlmSource[] {
    const cited: LlmSource[] = [];
    const found: LlmSource[] = [];
    for (const block of blocks) {
        if (block.type === 'text') {
            for (const citation of block.citations ?? []) {
                if (citation.type === 'web_search_result_location') {
                    cited.push(toSource(citation.url, citation.title));
                }
            }
        } else if (block.type === 'web_search_tool_result' && Array.isArray(block.content)) {
            for (const result of block.content) {
                if (result.type === 'web_search_result') {
                    found.push(toSource(result.url, result.title));
                }
            }
        }
    }
    const seen = new Set<string>();
    return (cited.length > 0 ? cited : found).filter((source) => {
        if (seen.has(source.url)) {
            return false;
        }
        seen.add(source.url);
        return true;
    });
}

/** Builds source with title when present. */
function toSource(url: string, title: string | null | undefined): LlmSource {
    return typeof title === 'string' && title !== '' ? { url, title } : { url };
}

/** Formats refusal message with category and explanation when present. */
function refusalMessage(details: Anthropic.RefusalStopDetails | null | undefined): string {
    const category = details?.category ? ` (${details.category})` : '';
    const explanation = details?.explanation ? `: ${details.explanation}` : '';
    return `The model refused the request${category}${explanation}`;
}

/** Maps API error with HTTP status to `rate_limited` or `http` error. */
function statusError(error: APIError): LlmProviderError {
    const status = error.status as number;
    const body = error.error as { error?: { message?: unknown } } | undefined;
    const apiMessage = body?.error?.message;
    const message =
        typeof apiMessage === 'string' && apiMessage !== '' ? `HTTP ${status}: ${apiMessage}` : `HTTP ${status}`;
    const retryAfterMs = readRetryAfter(error.headers);
    return {
        code: status === RATE_LIMITED_STATUS ? 'rate_limited' : 'http',
        message,
        retryable:
            status === RATE_LIMITED_STATUS || status >= MIN_SERVER_ERROR_STATUS || RETRYABLE_STATUSES.has(status),
        status,
        ...(retryAfterMs === undefined ? {} : { retryAfterMs }),
    };
}

/** Reads retry delay in ms from `retry-after-ms` (ms) or `retry-after` (seconds or HTTP date). */
function readRetryAfter(headers: unknown): number | undefined {
    const get = (headers as { get?: unknown } | undefined)?.get;
    if (typeof get !== 'function') {
        return undefined;
    }
    function read(name: string): string | undefined {
        const value: unknown = (get as (key: string) => unknown).call(headers, name);
        return typeof value === 'string' && value.trim() !== '' ? value : undefined;
    }
    const ms = Number(read('retry-after-ms'));
    if (Number.isFinite(ms) && ms >= 0) {
        return Math.ceil(ms);
    }
    const retryAfter = read('retry-after');
    if (retryAfter === undefined) {
        return undefined;
    }
    const seconds = Number(retryAfter);
    if (Number.isFinite(seconds) && seconds >= 0) {
        return Math.ceil(seconds * 1000);
    }
    const date = Date.parse(retryAfter);
    return Number.isNaN(date) ? undefined : Math.max(0, date - Date.now());
}
