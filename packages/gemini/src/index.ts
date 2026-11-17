// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type {
    LlmEmbedTask,
    LlmError,
    LlmProvider,
    LlmProviderError,
    LlmProviderOutput,
    LlmProviderRequest,
    LlmProviderSupport,
} from '@smart-llm/core';
import {
    createGenaiClient,
    type GenaiCallOptions,
    type GenaiClient,
    type GenaiClientOptions,
    type GenaiEmbedTask,
    type GenaiError,
    type GenaiOutput,
    type GenaiRequest,
    MAX_EMBED_INPUTS,
    mergeGenaiConfig,
    validateGenaiConfig,
} from 'smart-genai';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Gemini configuration type for smart-genai. */
type GenaiConfig = NonNullable<GenaiCallOptions['config']>;

/** Options of `createGeminiProvider`: smart-genai client options without `defaults` (set in `createLlm`). */
export type GeminiOptions = Omit<GenaiClientOptions, 'defaults'>;

/** Gemini extra options (provider, model, or call `extra`): smart-genai `config`, merged per top-level field. */
export type GeminiExtra = { config?: GenaiConfig };

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Supported keys for Gemini extra options. */
const EXTRA_KEYS: ReadonlySet<string> = new Set(['config']);

/** Capabilities supported by Gemini: images, tools, web search, reasoning, batch, native JSON, and embeddings. */
export const GEMINI_SUPPORT: LlmProviderSupport = {
    images: true,
    tools: true,
    webSearch: true,
    reasoning: true,
    batch: true,
    json: 'native',
    embeddings: true,
};

/** Maps embedding tasks to Gemini task types. */
const EMBED_TASK_TYPES: Readonly<Record<LlmEmbedTask, GenaiEmbedTask>> = {
    query: 'RETRIEVAL_QUERY',
    document: 'RETRIEVAL_DOCUMENT',
    similarity: 'SEMANTIC_SIMILARITY',
    classification: 'CLASSIFICATION',
    clustering: 'CLUSTERING',
};

/** Environment variable name for Gemini Developer API key. */
export { GEMINI_API_KEY_ENV } from 'smart-genai';

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Provider adapter for Gemini API** via smart-genai (Developer API or Vertex AI).
 *
 * - supports images, tools, web search, reasoning, batch calls, native JSON, and embeddings
 * - forwards `extra.config` to smart-genai configuration
 * - strips assistant reasoning history before sending
 * - embeds up to 100 texts per call mapping tasks to Gemini task types
 * - smart-genai `invalid_json` with output: returns output as answer, so core parses and validates JSON
 * @throws {TypeError} If client options contain `defaults`.
 */
export function createGeminiProvider(client: GenaiClient | GeminiOptions = {}): LlmProvider {
    const genai = toGenaiClient(client);
    return {
        name: 'gemini',
        supports: GEMINI_SUPPORT,
        validateExtra: validateGeminiExtra,
        mergeExtra: mergeGeminiExtra,
        async generate(request, { signal, extra }) {
            try {
                const options: GenaiCallOptions = { signal };
                if (extra.config !== undefined) {
                    options.config = extra.config as GenaiConfig;
                }
                const result = await genai.generate(toGenaiRequest(request), options);
                if (!result.ok) {
                    // leave JSON parsing to core: one message and fence rule for every provider
                    if (result.error.code === 'invalid_json' && result.error.output !== undefined) {
                        return { ok: true, data: toProviderOutput(result.error.output) };
                    }
                    return { ok: false, error: toProviderError(result.error) };
                }
                return { ok: true, data: toProviderOutput(result.data) };
            } catch (error) {
                return { ok: false, error: mapThrownError(error) };
            }
        },
        maxEmbedInputs: MAX_EMBED_INPUTS,
        async embed(request, { signal }) {
            try {
                const result = await genai.embed(
                    {
                        model: request.model,
                        input: request.input,
                        ...(request.task === undefined ? {} : { taskType: EMBED_TASK_TYPES[request.task] }),
                        ...(request.dimensions === undefined ? {} : { dimensions: request.dimensions }),
                    },
                    { signal },
                );
                if (!result.ok) {
                    return { ok: false, error: toProviderError(result.error) };
                }
                const { embeddings, model, ms } = result.data;
                return { ok: true, data: { embeddings, model, usage: { inputTokens: null }, ms } };
            } catch (error) {
                return { ok: false, error: mapThrownError(error) };
            }
        },
        async listModels(options = {}) {
            try {
                const result = await genai.listModels(options);
                if (!result.ok) {
                    const { output: _output, settings: _settings, attempts: _attempts, ...error } = result.error;
                    return { ok: false, error };
                }
                return {
                    ok: true,
                    data: result.data.map((model) => ({
                        id: model.id,
                        ...(model.displayName === undefined ? {} : { displayName: model.displayName }),
                        ...(model.inputTokenLimit === undefined ? {} : { inputTokenLimit: model.inputTokenLimit }),
                        ...(model.outputTokenLimit === undefined ? {} : { outputTokenLimit: model.outputTokenLimit }),
                    })),
                };
            } catch (error) {
                return { ok: false, error: mapThrownError(error) };
            }
        },
    };
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/**
 * Resolves smart-genai client from instance or options.
 * @throws {TypeError} If options contain `defaults`.
 */
function toGenaiClient(client: GenaiClient | GeminiOptions): GenaiClient {
    if (typeof (client as Partial<GenaiClient>).generate === 'function') {
        return client as GenaiClient;
    }
    if (typeof client === 'object' && client !== null && 'defaults' in client) {
        throw new TypeError('Option "defaults" is not supported: request defaults go into createLlm defaults');
    }
    return createGenaiClient(client as GenaiClientOptions);
}

/**
 * Validates Gemini extra options, accepting only `config`.
 * @throws {TypeError} If unsupported options or invalid config are provided.
 */
function validateGeminiExtra(extra: Record<string, unknown>, option: string): void {
    for (const key of Object.keys(extra)) {
        if (!EXTRA_KEYS.has(key)) {
            throw new TypeError(`Option "${option}.${key}" is not supported by the Gemini provider (only config)`);
        }
    }
    if (extra.config !== undefined) {
        validateGenaiConfig(extra.config, `${option}.config`);
    }
}

/**
 * **Merges extra options across layers**: `config` per top-level field (later layers win).
 *
 * - nested objects other than `thinkingConfig` are replaced whole; `thinkingConfig` merges per member
 * - `thinkingBudget` and `thinkingLevel` count as one setting: last layer setting either wins
 * - earlier layers lose both keys; other `thinkingConfig` members (`includeThoughts`) still merge
 */
function mergeGeminiExtra(layers: Record<string, unknown>[]): Record<string, unknown> {
    const merged: Record<string, unknown> = Object.assign({}, ...layers);
    const configs = layers.map((layer) => layer.config as GenaiConfig | undefined);
    const last = configs.findLastIndex(setsThinking);
    const config = mergeGenaiConfig(
        ...configs.map((layerConfig, index) => (index < last ? withoutThinkingSetting(layerConfig) : layerConfig)),
    );
    if (config === undefined) {
        delete merged.config;
    } else {
        merged.config = config;
    }
    return merged;
}

/** Checks whether config sets `thinkingBudget` or `thinkingLevel`; `thinkingConfig: null` sets neither. */
function setsThinking(config: GenaiConfig | undefined): boolean {
    const thinking = config?.thinkingConfig;
    return thinking?.thinkingBudget !== undefined || thinking?.thinkingLevel !== undefined;
}

/** Copies config without `thinkingBudget` and `thinkingLevel`; drops `thinkingConfig` if nothing remains. */
function withoutThinkingSetting(config: GenaiConfig | undefined): GenaiConfig | undefined {
    // smart-genai accepts `thinkingConfig: null`: nothing to strip
    if (typeof config?.thinkingConfig !== 'object' || config.thinkingConfig === null) {
        return config;
    }
    const { thinkingConfig, ...rest } = config;
    const { thinkingBudget: _budget, thinkingLevel: _level, ...thinking } = thinkingConfig;
    return Object.keys(thinking).length === 0 ? rest : { ...rest, thinkingConfig: thinking };
}

/** Strips wrapper metadata from smart-genai output. */
function toProviderOutput(output: GenaiOutput): LlmProviderOutput {
    const { json: _json, settings: _settings, attempts: _attempts, ...rest } = output;
    return rest;
}

/** Normalizes smart-genai error into provider error. */
function toProviderError(error: GenaiError): LlmProviderError {
    const { settings: _settings, attempts: _attempts, output, ...rest } = error;
    return output === undefined ? rest : { ...rest, output: toProviderOutput(output) };
}

/** Maps uncaught smart-genai exception to `invalid_request` or `unexpected` error. */
function mapThrownError(error: unknown): LlmError {
    const message = error instanceof Error ? error.message : String(error);
    return { code: error instanceof TypeError ? 'invalid_request' : 'unexpected', message, retryable: false };
}

/** Formats provider request for smart-genai, stripping assistant reasoning history. */
function toGenaiRequest(request: LlmProviderRequest): GenaiRequest {
    const messages = request.messages.map((message) => {
        if (message.role !== 'assistant' || message.reasoning === undefined) {
            return message;
        }
        const { reasoning: _reasoning, ...rest } = message;
        return rest;
    });
    return { ...request, messages } as GenaiRequest;
}
