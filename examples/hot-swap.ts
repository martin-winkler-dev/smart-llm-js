// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

//
// One configuration with Gemini, DeepSeek, Groq and a local Ollama model, and the same request sent to each.
// Run: bun examples/hot-swap.ts (keys from env GEMINI_API_KEY, DEEPSEEK_API_KEY, GROQ_API_KEY; Ollama on localhost)

import { createLlm, imageFromFile, type LlmResponseFormat } from '@smart-llm/core';
import { createDeepseekProvider } from '@smart-llm/deepseek';
import { createGeminiProvider } from '@smart-llm/gemini';
import { createGroqProvider } from '@smart-llm/groq';
import { createOllamaProvider } from '@smart-llm/ollama';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Multi-provider LLM instance configured with Gemini, DeepSeek, Groq, and Ollama. */
const llm = createLlm({
    // Base defaults across all providers and models
    defaults: { temperature: 0.2, max_tokens: 2000 },
    // Global retry and wait limits
    limiterOptions: { maxRetries: 2, maxWaitMs: 120_000 },
    providers: {
        gemini: {
            adapter: createGeminiProvider(),
            models: {
                // Model-level capability declarations (unlisted capabilities remain unsupported)
                // Gemini per-model rate limits
                flash: {
                    model: 'gemini-3.8-flash',
                    capabilities: { images: true, json: true, reasoning: { supported: true } },
                    modelLimits: { rpm: 1000, tpm: 1_000_000 },
                },
                // Alias with default reasoning effort set to high
                'flash-think': {
                    model: 'gemini-3.8-flash',
                    capabilities: { images: true, json: true, reasoning: { supported: true, default: 'high' } },
                    modelLimits: { rpm: 1000 },
                },
            },
        },
        deepseek: {
            adapter: createDeepseekProvider(),
            // Account-wide shared concurrency limit across DeepSeek models
            sharedLimits: { concurrency: 10 },
            models: {
                'ds-flash': {
                    model: 'deepseek-flash',
                    capabilities: {
                        images: true,
                        tools: true,
                        json: true,
                        reasoning: { supported: true, default: 'none' },
                    },
                },
            },
        },
        groq: {
            adapter: createGroqProvider(),
            models: {
                'gpt-oss': {
                    model: 'openai/gpt-oss-120b',
                    // Models with fixed reasoning levels map requested effort to nearest supported level
                    capabilities: {
                        tools: true,
                        json: true,
                        reasoning: { supported: true, levels: ['low', 'medium', 'high'] },
                    },
                    // Free tier rate limits with minimum request interval
                    modelLimits: { rpm: 30, tpm: 8000, rpd: 1000, intervalMs: 3000 },
                },
            },
        },
        ollama: {
            adapter: createOllamaProvider(),
            // Local Ollama server single-concurrency limit
            sharedLimits: { concurrency: 1 },
            charsPerToken: 3.5,
            // Alternative: dynamically inspect server models via fetchOllamaCapabilities()
            models: { qwen: { model: 'qwen2.5:7b', capabilities: { tools: true, json: true } } },
        },
    },
});

/** Structured output JSON schema for word classification. */
const word: LlmResponseFormat = {
    type: 'json_schema',
    json_schema: {
        name: 'word',
        schema: {
            type: 'object',
            properties: {
                kind: { type: 'string', enum: ['noun', 'verb', 'adjective'], description: 'Word class' },
                plural: { type: 'string', description: 'Plural form, only for nouns' },
            },
            required: ['kind'],
        },
    },
};

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Demonstrates uniform requests across multiple LLM providers. */
async function main(): Promise<void> {
    for (const model of ['flash', 'ds-flash', 'gpt-oss', 'qwen'] as const) {
        const result = await llm.generate({
            model,
            messages: [{ role: 'user', content: 'Classify the word "house".' }],
            response_format: word,
        });
        console.log(model, result.ok ? result.data.json : result.error);
    }

    // Unsupported capabilities are rejected before request transmission
    const photo = await imageFromFile('examples/photo.jpg');
    for (const model of ['flash', 'qwen'] as const) {
        const result = await llm.generate({
            model,
            messages: [{ role: 'user', content: [{ type: 'text', text: 'What is in the picture?' }, photo] }],
        });
        console.log(model, result.ok ? result.data.text : result.error.code);
    }

    // Convenience method invocation: llm.<modelName>(prompt, options)
    const short = await llm.flash('Say hi in one word.', { temperature: 0 });
    console.log('flash', short.ok ? short.data.text : short.error);
}

if (import.meta.main) {
    await main();
}
