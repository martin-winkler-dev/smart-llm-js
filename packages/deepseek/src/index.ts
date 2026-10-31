// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { LlmProvider, LlmProviderSupport } from '@smart-llm/core';
import { createOpenAiCompatibleProvider } from '@smart-llm/openai-compatible';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/**
 * **DeepSeek provider options**.
 *
 * - `apiKey`: API key (defaults to env `DEEPSEEK_API_KEY`)
 * - `env`: environment lookup map (defaults to `process.env`)
 * - `timeoutMs`: request timeout in milliseconds (defaults to 10 min)
 * - `baseUrl`: optional custom API base URL
 * - `fetch`: optional custom fetch implementation
 */
export type DeepseekOptions = {
    apiKey?: string;
    env?: Readonly<Record<string, string | undefined>>;
    baseUrl?: string;
    timeoutMs?: number;
    fetch?: typeof fetch;
};

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Environment variable name for DeepSeek API key. */
export const DEEPSEEK_API_KEY_ENV = 'DEEPSEEK_API_KEY';

/** Official DeepSeek API endpoint URL. */
const DEEPSEEK_BASE_URL = 'https://api.deepseek.com';

/** Default request timeout in milliseconds. */
const DEFAULT_TIMEOUT_MS = 600_000;

/** Capabilities supported by DeepSeek: images, tools, reasoning, and prompt-based JSON schemas. */
export const DEEPSEEK_SUPPORT: LlmProviderSupport = { images: true, tools: true, reasoning: true, json: 'prompt' };

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Provider adapter for DeepSeek API**.
 *
 * - supports images, tools, reasoning, and prompt-based JSON schemas
 * - preserves assistant `reasoning_content` in conversation history
 * - maps `max_tokens` field and forwards `reasoning_effort`
 * @throws {TypeError} If API key is missing.
 */
export function createDeepseekProvider(options: DeepseekOptions = {}): LlmProvider {
    const env = options.env ?? process.env;
    const apiKey = options.apiKey ?? env[DEEPSEEK_API_KEY_ENV];
    if (!apiKey) {
        throw new TypeError(`Invalid value for option "apiKey": set it or env ${DEEPSEEK_API_KEY_ENV}`);
    }
    return createOpenAiCompatibleProvider({
        name: 'deepseek',
        baseUrl: options.baseUrl ?? DEEPSEEK_BASE_URL,
        apiKey,
        supports: DEEPSEEK_SUPPORT,
        maxTokensField: 'max_tokens',
        reasoningFields: ['reasoning_content'],
        reasoningMessageField: 'reasoning_content',
        timeoutMs: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
        ...(options.fetch === undefined ? {} : { fetch: options.fetch }),
    });
}
