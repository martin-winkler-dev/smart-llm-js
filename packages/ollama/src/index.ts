// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type {
    LlmError,
    LlmModelCapabilities,
    LlmProvider,
    LlmProviderRequest,
    LlmProviderSupport,
    Result,
} from '@smart-llm/core';
import { callerAbortError, createOpenAiCompatibleProvider, fetchJson } from '@smart-llm/openai-compatible';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/**
 * **Ollama provider options**.
 *
 * - `baseUrl`: OpenAI-compatible endpoint URL (defaults to `http://localhost:11434/v1`)
 * - `apiKey`: optional bearer token (authenticated reverse proxies)
 * - `timeoutMs`: request timeout in milliseconds (defaults to 10 min)
 * - `fetch`: optional custom fetch implementation
 */
export type OllamaOptions = { baseUrl?: string; apiKey?: string; timeoutMs?: number; fetch?: typeof fetch };

/**
 * **Options for `fetchOllamaCapabilities`**.
 *
 * - `baseUrl`: server endpoint URL
 * - `apiKey`: optional bearer token (authenticated reverse proxies)
 * - `signal`: abort signal
 * - `timeoutMs`: request timeout (defaults to 30 s)
 * - `fetch`: optional custom fetch implementation
 */
export type OllamaCapabilitiesOptions = Pick<OllamaOptions, 'baseUrl' | 'apiKey' | 'fetch' | 'timeoutMs'> & {
    signal?: AbortSignal;
};

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Local Ollama endpoint URL. */
const OLLAMA_BASE_URL = 'http://localhost:11434/v1';

/** Default request timeout in milliseconds. */
const DEFAULT_TIMEOUT_MS = 600_000;

/** Default timeout for `fetchOllamaCapabilities` in milliseconds. */
const DEFAULT_TAGS_TIMEOUT_MS = 30_000;

/** Error message when `/api/tags` response lacks valid model list. */
const NO_MODEL_LIST = 'The /api/tags response is not a model list';

/** Trailing `/v1` endpoint path and trailing slashes (both optional). */
const MATCH_V1_SUFFIX = /(?:\/+v1)?\/*$/;

/** Capabilities supported by Ollama: images, tools, reasoning, grammar-based JSON format, and embeddings. */
export const OLLAMA_SUPPORT: LlmProviderSupport = {
    images: true,
    tools: true,
    reasoning: true,
    json: 'grammar',
    embeddings: true,
};

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Provider adapter for local Ollama server** via OpenAI-compatible endpoint.
 *
 * - supports images, tools, reasoning, grammar-based JSON format, and batch embeddings
 * - strips unsupported `tool_choice` parameter (removes `tools` when choice is `none`)
 * - extracts reasoning content from provider responses
 */
export function createOllamaProvider(options: OllamaOptions = {}): LlmProvider {
    return createOpenAiCompatibleProvider({
        name: 'ollama',
        baseUrl: options.baseUrl ?? OLLAMA_BASE_URL,
        ...(options.apiKey === undefined ? {} : { apiKey: options.apiKey }),
        supports: OLLAMA_SUPPORT,
        maxTokensField: 'max_tokens',
        reasoningFields: ['reasoning'],
        mapBody: withoutToolChoice,
        timeoutMs: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
        ...(options.fetch === undefined ? {} : { fetch: options.fetch }),
    });
}

/**
 * **Fetches installed model capabilities from Ollama server** (`GET /api/tags`).
 *
 * - queries model tags and maps reported capabilities (`vision`, `tools`, `thinking`, `embedding`)
 * - strips `/v1` suffix and trailing slashes from base URL
 * - sends `apiKey` as bearer token when set
 * - reports caller `signal` abort as `aborted`, internal `timeoutMs` as `timeout`
 * - returns result error on failure instead of throwing
 */
export async function fetchOllamaCapabilities(
    options: OllamaCapabilitiesOptions = {},
): Promise<Result<Record<string, LlmModelCapabilities>, LlmError>> {
    try {
        const root = (options.baseUrl ?? OLLAMA_BASE_URL).replace(MATCH_V1_SUFFIX, '');
        const timeout = AbortSignal.timeout(options.timeoutMs ?? DEFAULT_TAGS_TIMEOUT_MS);
        const sent = await fetchJson(options.fetch ?? fetch, `${root}/api/tags`, {
            ...(options.apiKey === undefined ? {} : { headers: { authorization: `Bearer ${options.apiKey}` } }),
            signal: options.signal === undefined ? timeout : AbortSignal.any([options.signal, timeout]),
        });
        if (!sent.ok) {
            return { ok: false, error: callerAbortError(sent.error, options.signal) };
        }
        const models = (sent.data as { models?: unknown } | null)?.models ?? [];
        if (!Array.isArray(models)) {
            return { ok: false, error: { code: 'unexpected', message: NO_MODEL_LIST, retryable: false } };
        }
        const entries: [string, LlmModelCapabilities][] = [];
        for (const model of models as { name?: unknown; capabilities?: unknown }[]) {
            const name = model?.name;
            if (typeof name !== 'string') {
                return { ok: false, error: { code: 'unexpected', message: NO_MODEL_LIST, retryable: false } };
            }
            const reported: unknown[] = Array.isArray(model.capabilities) ? model.capabilities : [];
            entries.push([
                name,
                {
                    ...(reported.includes('vision') ? { images: true } : {}),
                    ...(reported.includes('tools') ? { tools: true } : {}),
                    ...(reported.includes('thinking') ? { reasoning: { supported: true } } : {}),
                    ...(reported.includes('embedding') ? { embeddings: true } : {}),
                },
            ]);
        }
        return { ok: true, data: Object.fromEntries(entries) };
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return { ok: false, error: { code: 'unexpected', message, retryable: false } };
    }
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/** Removes unsupported `tool_choice` from request body, deleting `tools` when choice is `none`. */
function withoutToolChoice(body: Record<string, unknown>, _request: LlmProviderRequest): Record<string, unknown> {
    const { tool_choice: choice, ...rest } = body;
    if (choice === 'none') {
        delete rest.tools;
    }
    return rest;
}
