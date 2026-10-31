// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'bun:test';
import {
    createLlm,
    type LlmProviderRequest,
    type LlmReasoningEffort,
    type LlmResponseFormat,
    type LlmSettings,
} from '@smart-llm/core';
import { createGroqProvider, GROQ_SUPPORT } from '../src/index.js';

/** Creates fake fetch recording requests and answering with "ok". */
function fakeFetch(): { fetch: typeof fetch; bodies: Record<string, unknown>[]; urls: string[] } {
    const bodies: Record<string, unknown>[] = [];
    const urls: string[] = [];
    async function fetcher(url: string, init: RequestInit): Promise<Response> {
        urls.push(url);
        bodies.push(JSON.parse(String(init.body)));
        return new Response(JSON.stringify({ choices: [{ finish_reason: 'stop', message: { content: 'ok' } }] }));
    }
    return { fetch: fetcher as typeof fetch, bodies, urls };
}

const call = { signal: new AbortController().signal, extra: {} };

/** Formats test provider request with optional reasoning effort. */
function request(model: string, effort?: LlmReasoningEffort): LlmProviderRequest {
    return {
        model,
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 10,
        ...(effort === undefined ? {} : { reasoning_effort: effort }),
    };
}

describe('createGroqProvider', () => {
    it('throws without an API key and reads it from env', () => {
        expect(() => createGroqProvider({ env: {} })).toThrow(/GROQ_API_KEY/);
        expect(createGroqProvider({ env: { GROQ_API_KEY: 'k' } }).name).toBe('groq');
    });

    it('sends reasoning_effort as it is, max_completion_tokens, and extra such as reasoning_format', async () => {
        const fake = fakeFetch();
        const groq = createGroqProvider({ apiKey: 'k', fetch: fake.fetch });
        await groq.generate(request('openai/gpt-oss-20b', 'high'), call);
        await groq.generate(request('qwen/qwen3.8-27b', 'none'), { ...call, extra: { reasoning_format: 'parsed' } });
        await groq.generate(request('llama-3.3-70b-versatile'), call);
        expect(fake.urls[0]).toBe('https://api.groq.com/openai/v1/chat/completions');
        expect(fake.bodies[0]).toMatchObject({ reasoning_effort: 'high', max_completion_tokens: 10 });
        expect(fake.bodies[0]?.max_tokens).toBeUndefined();
        expect(fake.bodies[0]?.reasoning_format).toBeUndefined();
        expect(fake.bodies[1]).toMatchObject({ reasoning_effort: 'none', reasoning_format: 'parsed' });
        expect(fake.bodies[2]?.reasoning_effort).toBeUndefined();
    });

    it('supports images, tools, reasoning and strict JSON schemas', () => {
        expect(GROQ_SUPPORT).toEqual({ images: true, tools: true, reasoning: true, json: 'strict' });
        expect(createGroqProvider({ apiKey: 'k' }).supports).toEqual(GROQ_SUPPORT);
    });
});

/** Creates fake queued fetch implementation recording request bodies. */
function queueFetch(...answers: (() => Response)[]): { fetch: typeof fetch; bodies: Record<string, unknown>[] } {
    const bodies: Record<string, unknown>[] = [];
    async function fetcher(_url: string, init: RequestInit): Promise<Response> {
        bodies.push(JSON.parse(String(init.body)));
        const next = answers.length > 1 ? answers.shift() : answers[0];
        return (next as () => Response)();
    }
    return { fetch: fetcher as typeof fetch, bodies };
}

/** Returns mock response factory with specified content and finish reason. */
function reply(content: string, finishReason = 'stop'): () => Response {
    return () =>
        new Response(
            JSON.stringify({
                choices: [{ finish_reason: finishReason, message: { content, reasoning: 'why' } }],
                usage: { prompt_tokens: 3, completion_tokens: 9, completion_tokens_details: { reasoning_tokens: 5 } },
            }),
        );
}

/** Creates test LLM instance wrapping Groq provider with queued responses. */
function groqLlm(...answers: (() => Response)[]) {
    const fake = queueFetch(...answers);
    const llm = createLlm({
        limiterOptions: { maxRetries: 1, retryDelayMs: 1 },
        providers: {
            groq: {
                adapter: createGroqProvider({ apiKey: 'k', fetch: fake.fetch }),
                defaults: { temperature: 0.5, seed: 3 },
                models: {
                    oss: {
                        model: 'openai/gpt-oss-20b',
                        defaults: { max_completion_tokens: 200 },
                        extra: { reasoning_format: 'parsed' },
                        capabilities: { json: true, reasoning: { supported: true, levels: ['low', 'medium', 'high'] } },
                    },
                },
            },
        },
    });
    return { llm, bodies: fake.bodies };
}

const format: LlmResponseFormat = {
    type: 'json_schema',
    json_schema: {
        name: 'word',
        schema: { type: 'object', properties: { kind: { type: 'string' } }, required: ['kind'] },
    },
};

const names = { name: 'oss', provider: 'groq' };

const settings: LlmSettings = {
    ...names,
    model: 'openai/gpt-oss-20b',
    temperature: 0.5,
    seed: 3,
    max_tokens: 200,
    reasoning_effort: 'low',
    jsonMode: 'strict',
    extra: { reasoning_format: 'parsed' },
};

const options = { response_format: format, reasoning_effort: 'minimal' } as const;

/** Returns mock response simulating `json_validate_failed` schema error. */
function failedGeneration(): Response {
    return new Response(
        JSON.stringify({
            error: {
                message: 'Generated JSON does not match the expected schema',
                code: 'json_validate_failed',
                failed_generation: '{"kind":',
            },
        }),
        { status: 400 },
    );
}

describe('Groq through createLlm: settings, attempts and the answer of failed calls', () => {
    it('reports the settings with the effort mapped to the model levels and one attempt', async () => {
        const { llm, bodies } = groqLlm(reply('{"kind":"noun"}'));
        const result = await llm.oss('word?', options);
        expect(result).toEqual({
            ok: true,
            data: {
                text: '{"kind":"noun"}',
                json: { kind: 'noun' },
                reasoning: 'why',
                model: 'openai/gpt-oss-20b',
                usage: { inputTokens: 3, outputTokens: 4, reasoningTokens: 5 },
                finishReason: 'stop',
                ms: expect.any(Number),
                ...names,
                settings,
                attempts: 1,
            },
        });
        expect(bodies[0]).toMatchObject({ reasoning_effort: 'low', max_completion_tokens: 200 });
    });

    it('does not retry a schema mismatch in strict mode and puts the answer with reasoning and json on it', async () => {
        const { llm, bodies } = groqLlm(reply('{"kind":1}'));
        const result = await llm.oss('word?', options);
        expect(result).toMatchObject({
            ok: false,
            error: { code: 'invalid_json', retryable: false, text: '{"kind":1}', ...names, settings, attempts: 1 },
        });
        expect(result.ok ? undefined : result.error.output).toEqual({
            text: '{"kind":1}',
            json: { kind: 1 },
            reasoning: 'why',
            model: 'openai/gpt-oss-20b',
            usage: { inputTokens: 3, outputTokens: 4, reasoningTokens: 5 },
            finishReason: 'stop',
            ms: expect.any(Number),
            ...names,
        });
        expect(bodies).toHaveLength(1);
    });

    it('retries json_validate_failed and puts the failed generation on the error as output', async () => {
        const { llm, bodies } = groqLlm(failedGeneration);
        const result = await llm.oss('word?', options);
        expect(result).toMatchObject({
            ok: false,
            error: { code: 'invalid_json', status: 400, text: '{"kind":', ...names, settings, attempts: 2 },
        });
        expect(result.ok ? undefined : result.error.output).toEqual({
            text: '{"kind":',
            model: 'openai/gpt-oss-20b',
            usage: { inputTokens: null, outputTokens: null, reasoningTokens: null },
            finishReason: null,
            ms: expect.any(Number),
            ...names,
        });
        expect(bodies).toHaveLength(2);
        const recovered = await groqLlm(failedGeneration, reply('{"kind":"verb"}')).llm.oss('word?', options);
        expect(recovered).toMatchObject({ ok: true, data: { json: { kind: 'verb' }, attempts: 2 } });
    });

    it('puts the answer with reasoning on truncated and blocked errors after one attempt', async () => {
        for (const [reason, code] of [
            ['length', 'truncated'],
            ['content_filter', 'blocked'],
        ] as const) {
            const { llm, bodies } = groqLlm(reply('{"kind', reason));
            expect(await llm.oss('word?', options)).toMatchObject({
                ok: false,
                error: {
                    code,
                    ...names,
                    output: { text: '{"kind', reasoning: 'why', finishReason: reason, ...names },
                    settings,
                    attempts: 1,
                },
            });
            expect(bodies).toHaveLength(1);
        }
    });

    it('puts no output but the settings on an HTTP error', async () => {
        const { llm } = groqLlm(() => new Response('{"error":{"message":"invalid key"}}', { status: 401 }));
        const result = await llm.oss('word?', options);
        expect(result).toMatchObject({
            ok: false,
            error: { code: 'http', status: 401, retryable: false, ...names, settings, attempts: 1 },
        });
        expect(result.ok ? undefined : result.error).not.toHaveProperty('output');
    });
});
