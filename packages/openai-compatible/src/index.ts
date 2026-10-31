// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { randomUUID } from 'node:crypto';
import type {
    LlmError,
    LlmMessage,
    LlmModelInfo,
    LlmProvider,
    LlmProviderEmbedOutput,
    LlmProviderEmbedRequest,
    LlmProviderError,
    LlmProviderOutput,
    LlmProviderRequest,
    LlmProviderSupport,
    LlmToolCall,
    Result,
} from '@smart-llm/core';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/**
 * **OpenAI-compatible provider options** (`/chat/completions` endpoint).
 *
 * - `baseUrl`: API endpoint URL (e.g. `https://api.groq.com/openai/v1`)
 * - `apiKey`: optional API bearer token
 * - `headers`: optional additional request headers
 * - `supports`: supported API capabilities (images, tools, reasoning, JSON mode, embeddings via `/embeddings`)
 * - `maxTokensField`: completion limit field (`max_completion_tokens` or `max_tokens`)
 * - `mapBody`: optional callback to transform request body
 * - `reasoningFields`: completion message fields containing reasoning content
 * - `reasoningMessageField`: assistant message field used to return reasoning history
 * - `timeoutMs`: request timeout in milliseconds
 * - `fetch`: optional custom fetch implementation
 */
export type OpenAiCompatibleOptions = {
    name: string;
    baseUrl: string;
    apiKey?: string;
    headers?: Record<string, string>;
    supports?: Omit<LlmProviderSupport, 'webSearch' | 'batch'>;
    maxTokensField?: 'max_tokens' | 'max_completion_tokens';
    mapBody?(body: Record<string, unknown>, request: LlmProviderRequest): Record<string, unknown>;
    reasoningFields?: readonly string[];
    reasoningMessageField?: string;
    timeoutMs?: number;
    fetch?: typeof fetch;
};

/** Chat completion response structure read by adapter. */
type ChatCompletion = {
    model?: string;
    choices?: {
        // biome-ignore lint/style/useNamingConvention: OpenAI field name
        finish_reason?: string | null;
        message?: {
            content?: string | null;
            // biome-ignore lint/style/useNamingConvention: OpenAI field name
            tool_calls?: unknown;
            [field: string]: unknown;
        };
    }[];
    usage?: {
        // biome-ignore lint/style/useNamingConvention: OpenAI field name
        prompt_tokens?: number;
        // biome-ignore lint/style/useNamingConvention: OpenAI field name
        completion_tokens?: number;
        // biome-ignore lint/style/useNamingConvention: OpenAI field name
        completion_tokens_details?: { reasoning_tokens?: number };
    };
};

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Default timeout per call in milliseconds. */
const DEFAULT_TIMEOUT_MS = 120_000;

/** Message fields containing reasoning content (`reasoning_content` or `reasoning`). */
const DEFAULT_REASONING_FIELDS = ['reasoning_content', 'reasoning'];

/** Retryable HTTP status codes outside 5xx (e.g. 498 flex tier capacity). */
const RETRYABLE_STATUSES: ReadonlySet<number> = new Set([408, 409, 425, 498]);

/** Error code for failed JSON schema validation (sent with `failed_generation`). */
const JSON_FAILED_CODE = 'json_validate_failed';

/** Groq error code for malformed tool calls. */
const TOOL_FAILED_CODE = 'tool_use_failed';

/** Finish reasons indicating completed usable response. */
const DONE_REASONS: ReadonlySet<string> = new Set(['stop', 'end_turn', 'tool_calls']);

/** Lowest HTTP status code for server errors. */
const MIN_SERVER_ERROR_STATUS = 500;

/** HTTP status code for rate limiting. */
const RATE_LIMITED_STATUS = 429;

/** Trailing slashes pattern. */
const MATCH_TRAILING_SLASHES = /\/+$/;

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Provider adapter for any OpenAI-compatible chat completion API**.
 *
 * - formats request payload with mapped token limit and reasoning message fields
 * - normalizes response tool calls, generating missing IDs and serializing arguments
 * - handles retryable status codes and extracts `retry-after` header delays
 * - maps finish reasons to error codes (`length` -> `truncated`, `content_filter` -> `blocked`)
 * - deducts reasoning tokens from total completion tokens in reported usage
 * - executes batch `/embeddings` requests and sorts vectors by input index
 * - captures failures into structured `LlmError` results without throwing
 */
export function createOpenAiCompatibleProvider(options: OpenAiCompatibleOptions): LlmProvider {
    const baseUrl = options.baseUrl.replace(MATCH_TRAILING_SLASHES, '');
    const fetcher = options.fetch ?? fetch;
    const headers: Record<string, string> = {
        'content-type': 'application/json',
        ...(options.apiKey === undefined ? {} : { authorization: `Bearer ${options.apiKey}` }),
        ...options.headers,
    };
    const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const reasoningFields = options.reasoningFields ?? DEFAULT_REASONING_FIELDS;
    return {
        name: options.name,
        supports: openAiCompatibleSupport(options.supports),
        async generate(request, { signal, extra }) {
            const missing = request.batch === true ? 'batch calls' : request.web_search === true ? 'web search' : '';
            if (missing !== '') {
                const message = `${options.name} does not support ${missing}`;
                return { ok: false, error: { code: 'unsupported', message, retryable: false } };
            }
            try {
                const body = toBody(
                    request,
                    options.maxTokensField ?? 'max_completion_tokens',
                    options.reasoningMessageField,
                    extra,
                );
                const finalBody = options.mapBody === undefined ? body : options.mapBody(body, request);
                const startedAt = performance.now();
                const sent = await fetchJson(fetcher, `${baseUrl}/chat/completions`, {
                    method: 'POST',
                    headers,
                    body: JSON.stringify(finalBody),
                    signal: AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]),
                });
                if (!sent.ok) {
                    return withFailedGeneration(sent.error, request.model, performance.now() - startedAt);
                }
                return readCompletion(
                    (isRecord(sent.data) ? sent.data : {}) as ChatCompletion,
                    request.model,
                    reasoningFields,
                    performance.now() - startedAt,
                );
            } catch (error) {
                return { ok: false, error: unexpectedError(error) };
            }
        },
        async embed(request, { signal }) {
            if (options.supports?.embeddings !== true) {
                const message = `${options.name} does not support embeddings`;
                return { ok: false, error: { code: 'unsupported', message, retryable: false } };
            }
            if (request.task !== undefined || request.input.some((input) => input.title !== undefined)) {
                const message = `${options.name} embeddings take no task or title`;
                return { ok: false, error: { code: 'unsupported', message, retryable: false } };
            }
            try {
                const body = {
                    model: request.model,
                    input: request.input.map((input) => input.text),
                    ...(request.dimensions === undefined ? {} : { dimensions: request.dimensions }),
                };
                const startedAt = performance.now();
                const sent = await fetchJson(fetcher, `${baseUrl}/embeddings`, {
                    method: 'POST',
                    headers,
                    body: JSON.stringify(body),
                    signal: AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]),
                });
                if (!sent.ok) {
                    return { ok: false, error: { ...sent.error, ms: Math.round(performance.now() - startedAt) } };
                }
                return readEmbeddings(sent.data, request, Math.round(performance.now() - startedAt));
            } catch (error) {
                return { ok: false, error: unexpectedError(error) };
            }
        },
        async listModels(listOptions = {}) {
            try {
                const timeout = AbortSignal.timeout(timeoutMs);
                const sent = await fetchJson(fetcher, `${baseUrl}/models`, {
                    headers,
                    signal: listOptions.signal === undefined ? timeout : AbortSignal.any([listOptions.signal, timeout]),
                });
                if (!sent.ok) {
                    return sent;
                }
                const data = isRecord(sent.data) ? sent.data.data : undefined;
                const models: LlmModelInfo[] = (Array.isArray(data) ? data : []).flatMap((model: unknown) =>
                    isRecord(model) && typeof model.id === 'string' ? [{ id: model.id }] : [],
                );
                return { ok: true, data: models };
            } catch (error) {
                return { ok: false, error: unexpectedError(error) };
            }
        },
    };
}

/** Maps HTTP status code and message to `LlmError`, parsing `retry-after` when present. */
export function statusError(status: number, message: string, retryAfter: string | null): LlmError {
    const retryAfterMs = parseRetryAfter(retryAfter);
    if (status === RATE_LIMITED_STATUS) {
        return {
            code: 'rate_limited',
            message,
            retryable: true,
            status,
            ...(retryAfterMs === undefined ? {} : { retryAfterMs }),
        };
    }
    return {
        code: 'http',
        message,
        retryable: status >= MIN_SERVER_ERROR_STATUS || RETRYABLE_STATUSES.has(status),
        status,
        ...(retryAfterMs === undefined ? {} : { retryAfterMs }),
    };
}

/**
 * **Performs HTTP request and parses JSON response**.
 *
 * - returns parsed JSON data on 2xx responses
 * - maps HTTP status codes, network errors, timeouts, and aborts to `LlmError`
 */
export async function fetchJson(
    fetcher: typeof fetch,
    url: string,
    init: RequestInit,
): Promise<Result<unknown, LlmError>> {
    let response: Response;
    try {
        response = await fetcher(url, init);
    } catch (error) {
        return { ok: false, error: fetchError(error, init.signal) };
    }
    let text: string;
    try {
        text = await response.text();
    } catch (error) {
        return { ok: false, error: fetchError(error, init.signal) };
    }
    if (!response.ok) {
        const body = readErrorBody(text);
        const message =
            body.message === undefined ? `HTTP ${response.status}` : `HTTP ${response.status}: ${body.message}`;
        if (body.code === JSON_FAILED_CODE) {
            const error: LlmError = { code: 'invalid_json', message, retryable: true, status: response.status };
            if (body.failedGeneration !== undefined) {
                error.text = body.failedGeneration;
            }
            return { ok: false, error };
        }
        const error = statusError(response.status, message, response.headers.get('retry-after'));
        if (body.code === TOOL_FAILED_CODE) {
            error.retryable = true;
        }
        return { ok: false, error };
    }
    try {
        return { ok: true, data: JSON.parse(text) };
    } catch {
        return { ok: false, error: { code: 'unexpected', message: 'The response is not JSON', retryable: false } };
    }
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/** Filters capabilities to exclude unsupported `webSearch` and `batch`. */
function openAiCompatibleSupport(given: OpenAiCompatibleOptions['supports']): LlmProviderSupport {
    const { webSearch: _search, batch: _batch, ...supports } = (given ?? {}) as LlmProviderSupport;
    return supports;
}

/** Formats request into chat completion payload. */
function toBody(
    request: LlmProviderRequest,
    maxTokensField: 'max_tokens' | 'max_completion_tokens',
    reasoningMessageField: string | undefined,
    extra: Record<string, unknown>,
): Record<string, unknown> {
    const { max_tokens: maxTokens, web_search: _search, batch: _batch, messages, ...fields } = request;
    return {
        ...fields,
        messages: messages.map((message) => toBodyMessage(message, reasoningMessageField)),
        ...(maxTokens === undefined ? {} : { [maxTokensField]: maxTokens }),
        ...extra,
    };
}

/** Formats message for wire transmission, normalizing developer role and reasoning fields. */
function toBodyMessage(message: LlmMessage, reasoningMessageField: string | undefined): Record<string, unknown> {
    if (message.role === 'developer') {
        return { ...message, role: 'system' };
    }
    if (message.role !== 'assistant') {
        return message;
    }
    const { reasoning, tool_calls: calls, ...rest } = message;
    return {
        ...rest,
        ...(calls === undefined
            ? {}
            : {
                  // biome-ignore lint/style/useNamingConvention: OpenAI field name
                  tool_calls: calls.map(({ extra_content: _extra, ...call }) => call),
              }),
        ...(reasoning === undefined || reasoningMessageField === undefined
            ? {}
            : { [reasoningMessageField]: reasoning }),
    };
}

/** Extracts tool calls from message, generating missing IDs and serializing arguments. */
function readToolCalls(calls: unknown): LlmToolCall[] {
    if (!Array.isArray(calls)) {
        return [];
    }
    return calls.flatMap((call: { id?: unknown; function?: { name?: unknown; arguments?: unknown } } | null) => {
        const name = call?.function?.name;
        if (typeof name !== 'string') {
            return [];
        }
        const args = call?.function?.arguments;
        return [
            {
                id:
                    typeof call?.id === 'string' && call.id !== ''
                        ? call.id
                        : `call_${randomUUID().replaceAll('-', '')}`,
                type: 'function' as const,
                function: { name, arguments: typeof args === 'string' ? args : JSON.stringify(args ?? {}) },
            },
        ];
    });
}

/** Attaches failed generation text as output to invalid JSON errors. */
function withFailedGeneration(
    error: LlmError,
    requestedModel: string,
    ms: number,
): { ok: false; error: LlmProviderError } {
    if (error.code !== 'invalid_json' || error.text === undefined) {
        return { ok: false, error };
    }
    const output: LlmProviderOutput = {
        text: error.text,
        model: requestedModel,
        usage: { inputTokens: null, outputTokens: null, reasoningTokens: null },
        finishReason: null,
        ms: Math.round(ms),
    };
    return { ok: false, error: { ...error, output } };
}

/** Formats thrown exception into unexpected error. */
function unexpectedError(error: unknown): LlmError {
    return { code: 'unexpected', message: error instanceof Error ? error.message : String(error), retryable: false };
}

/** Parses embeddings response and sorts vectors by input index. */
function readEmbeddings(
    data: unknown,
    request: LlmProviderEmbedRequest,
    ms: number,
): Result<LlmProviderEmbedOutput, LlmError> {
    const items = isRecord(data) && Array.isArray(data.data) ? (data.data as unknown[]) : undefined;
    const vectors = (items ?? [])
        .filter(isRecord)
        .map((item, position) => ({
            index: typeof item.index === 'number' ? item.index : position,
            embedding: item.embedding,
        }))
        .sort((a, b) => a.index - b.index)
        .map((item) => item.embedding);
    const valid =
        items !== undefined &&
        vectors.length === request.input.length &&
        vectors.every(
            (vector) =>
                Array.isArray(vector) && vector.length > 0 && vector.every((value) => typeof value === 'number'),
        );
    if (!valid) {
        const message = `The /embeddings response has no ${request.input.length} vectors`;
        return { ok: false, error: { code: 'unexpected', message, retryable: false, ms } };
    }
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    const record = data as { model?: unknown; usage?: { prompt_tokens?: unknown } };
    const tokens = record.usage?.prompt_tokens;
    return {
        ok: true,
        data: {
            embeddings: vectors as number[][],
            model: typeof record.model === 'string' && record.model !== '' ? record.model : request.model,
            usage: { inputTokens: typeof tokens === 'number' ? tokens : null },
            ms,
        },
    };
}

/** Checks whether value is a non-array object. */
function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Maps fetch failure to timeout, aborted, or network error. */
function fetchError(error: unknown, signal: AbortSignal | null | undefined): LlmError {
    const name = (error as { name?: unknown } | null)?.name;
    const message = error instanceof Error ? error.message : String(error);
    if (name === 'TimeoutError' || (signal?.reason as { name?: unknown } | undefined)?.name === 'TimeoutError') {
        return { code: 'timeout', message, retryable: true };
    }
    if (name === 'AbortError' || signal?.aborted) {
        return { code: 'aborted', message, retryable: false };
    }
    return { code: 'network', message, retryable: true };
}

/** Parses error details from response JSON body. */
function readErrorBody(text: string): { message?: string; code?: string; failedGeneration?: string } {
    let body: { error?: unknown; message?: unknown };
    try {
        body = JSON.parse(text) as { error?: unknown; message?: unknown };
    } catch {
        return {};
    }
    const error = (typeof body?.error === 'object' && body.error !== null ? body.error : {}) as Record<string, unknown>;
    const message = typeof body?.error === 'string' ? body.error : (error.message ?? body?.message);
    return {
        ...(typeof message === 'string' && message !== '' ? { message } : {}),
        ...(typeof error.code === 'string' ? { code: error.code } : {}),
        ...(typeof error.failed_generation === 'string' ? { failedGeneration: error.failed_generation } : {}),
    };
}

/** Parses chat completion response into output or finish reason error. */
function readCompletion(
    completion: ChatCompletion,
    requestedModel: string,
    reasoningFields: readonly string[],
    ms: number,
): Result<LlmProviderOutput, LlmProviderError> {
    const choice = completion.choices?.[0];
    const message = choice?.message;
    const usage = completion.usage;
    const reasoningTokens = usage?.completion_tokens_details?.reasoning_tokens ?? null;
    const completionTokens = usage?.completion_tokens ?? null;
    const output: LlmProviderOutput = {
        text: typeof message?.content === 'string' ? message.content : '',
        model: completion.model ?? requestedModel,
        usage: {
            inputTokens: usage?.prompt_tokens ?? null,
            outputTokens: completionTokens === null ? null : completionTokens - (reasoningTokens ?? 0),
            reasoningTokens,
        },
        finishReason: choice?.finish_reason ?? null,
        ms: Math.round(ms),
    };
    const reasoning = reasoningFields
        .map((field) => message?.[field])
        .find((value) => typeof value === 'string' && value !== '');
    if (typeof reasoning === 'string') {
        output.reasoning = reasoning;
    }
    const toolCalls = readToolCalls(message?.tool_calls);
    if (toolCalls.length > 0) {
        output.toolCalls = toolCalls;
    }
    if (choice === undefined || message === undefined) {
        return { ok: false, error: { code: 'unexpected', message: 'The response has no choice', retryable: false } };
    }
    const reason = output.finishReason;
    const answered = { text: output.text, model: output.model, usage: output.usage, ms: output.ms, output };
    if (reason === 'tool_calls' && toolCalls.length === 0) {
        // Missing tool name: unusable response
        return {
            ok: false,
            error: {
                code: 'unexpected',
                message: 'Finish reason tool_calls without a usable tool call',
                retryable: true,
                ...answered,
            },
        };
    }
    if (reason === null || DONE_REASONS.has(reason)) {
        const broken = toolCalls.find((toolCall) => !isJsonObjectText(toolCall.function.arguments));
        if (broken !== undefined) {
            // Malformed arguments cannot be fed back: retryable
            return {
                ok: false,
                error: {
                    code: 'unexpected',
                    message: `The arguments of tool call ${broken.function.name} are not a JSON object`,
                    retryable: true,
                    ...answered,
                },
            };
        }
        return { ok: true, data: output };
    }
    if (reason === 'length') {
        return {
            ok: false,
            error: {
                code: 'truncated',
                message: 'The answer hit the output token limit',
                retryable: false,
                ...answered,
            },
        };
    }
    if (reason === 'content_filter') {
        return {
            ok: false,
            error: {
                code: 'blocked',
                message: 'The answer was blocked by a content filter',
                retryable: false,
                ...answered,
            },
        };
    }
    return {
        ok: false,
        error: {
            code: 'unexpected',
            message: `Finish reason ${reason}`,
            retryable: reason === 'insufficient_system_resource',
            ...answered,
        },
    };
}

/** Checks whether text is valid JSON object. */
function isJsonObjectText(text: string): boolean {
    try {
        const value: unknown = JSON.parse(text);
        return typeof value === 'object' && value !== null && !Array.isArray(value);
    } catch {
        return false;
    }
}

/** Parses `retry-after` header as milliseconds (seconds or HTTP date). */
function parseRetryAfter(value: string | null): number | undefined {
    if (value === null || value.trim() === '') {
        return undefined;
    }
    const seconds = Number(value);
    if (Number.isFinite(seconds) && seconds >= 0) {
        return Math.ceil(seconds * 1000);
    }
    const date = Date.parse(value);
    return Number.isNaN(date) ? undefined : Math.max(0, date - Date.now());
}
