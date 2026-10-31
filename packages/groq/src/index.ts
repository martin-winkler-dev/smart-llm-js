// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { LlmProvider, LlmProviderSupport } from '@smart-llm/core';
import { createOpenAiCompatibleProvider } from '@smart-llm/openai-compatible';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/**
 * **Groq provider options**.
 *
 * - `apiKey`: API key (defaults to env `GROQ_API_KEY`)
 * - `env`: environment lookup map (defaults to `process.env`)
 * - `timeoutMs`: request timeout in milliseconds (defaults to 120 s)
 * - `baseUrl`: optional custom API base URL
 * - `fetch`: optional custom fetch implementation
 */
export type GroqOptions = {
    apiKey?: string;
    env?: Readonly<Record<string, string | undefined>>;
    baseUrl?: string;
    timeoutMs?: number;
    fetch?: typeof fetch;
};

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Environment variable name for Groq API key. */
export const GROQ_API_KEY_ENV = 'GROQ_API_KEY';

/** Official Groq API endpoint URL. */
const GROQ_BASE_URL = 'https://api.groq.com/openai/v1';

/** Capabilities supported by Groq: images, tools, reasoning, and strict structured output. */
export const GROQ_SUPPORT: LlmProviderSupport = { images: true, tools: true, reasoning: true, json: 'strict' };

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Provider adapter for Groq API**.
 *
 * - supports images, tools, reasoning, and strict structured output
 * - maps `max_tokens` to `max_completion_tokens`
 * - extracts reasoning content from provider responses
 * @throws {TypeError} If API key is missing.
 */
export function createGroqProvider(options: GroqOptions = {}): LlmProvider {
    const env = options.env ?? process.env;
    const apiKey = options.apiKey ?? env[GROQ_API_KEY_ENV];
    if (!apiKey) {
        throw new TypeError(`Invalid value for option "apiKey": set it or env ${GROQ_API_KEY_ENV}`);
    }
    return createOpenAiCompatibleProvider({
        name: 'groq',
        baseUrl: options.baseUrl ?? GROQ_BASE_URL,
        apiKey,
        supports: GROQ_SUPPORT,
        maxTokensField: 'max_completion_tokens',
        reasoningFields: ['reasoning'],
        ...(options.timeoutMs === undefined ? {} : { timeoutMs: options.timeoutMs }),
        ...(options.fetch === undefined ? {} : { fetch: options.fetch }),
    });
}
