// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'bun:test';
import { createLlm, type LlmResponseFormat, type LlmSettings } from '@smart-llm/core';
import { createDeepseekProvider, DEEPSEEK_SUPPORT } from '../src/index.js';

describe('createDeepseekProvider', () => {
    it('throws without an API key', () => {
        expect(() => createDeepseekProvider({ env: {} })).toThrow(/DEEPSEEK_API_KEY/);
    });

    it('sends max_tokens and reasoning_effort as they are and reads reasoning_content', async () => {
        const sent: { url: string; init: RequestInit }[] = [];
        async function fetcher(url: string, init: RequestInit): Promise<Response> {
            sent.push({ url, init });
            return new Response(
                JSON.stringify({
                    choices: [{ finish_reason: 'stop', message: { content: '{}', reasoning_content: 'why' } }],
                }),
            );
        }
        const deepseek = createDeepseekProvider({ env: { DEEPSEEK_API_KEY: 'k' }, fetch: fetcher as typeof fetch });
        const result = await deepseek.generate(
            {
                model: 'deepseek-flash',
                messages: [{ role: 'user', content: 'hi' }],
                max_tokens: 5,
                reasoning_effort: 'none',
            },
            { signal: new AbortController().signal, extra: {} },
        );
        expect(result).toMatchObject({ ok: true, data: { text: '{}', reasoning: 'why' } });
        expect(sent[0]?.url).toBe('https://api.deepseek.com/chat/completions');
        expect((sent[0]?.init.headers as Record<string, string> | undefined)?.authorization).toBe('Bearer k');
        expect(JSON.parse(String(sent[0]?.init.body))).toMatchObject({ max_tokens: 5, reasoning_effort: 'none' });
    });
});

describe('DeepSeek tool calls', () => {
    it('sends the reasoning of assistant messages back as reasoning_content and reads tool calls', async () => {
        const bodies: Record<string, unknown>[] = [];
        async function fetcher(_url: string, init: RequestInit): Promise<Response> {
            bodies.push(JSON.parse(String(init.body)));
            return new Response(
                JSON.stringify({
                    choices: [
                        {
                            finish_reason: 'tool_calls',
                            message: {
                                content: '',
                                reasoning_content: 'again',
                                tool_calls: [{ id: 'c2', type: 'function', function: { name: 'f', arguments: '{}' } }],
                            },
                        },
                    ],
                }),
            );
        }
        const deepseek = createDeepseekProvider({ env: { DEEPSEEK_API_KEY: 'k' }, fetch: fetcher as typeof fetch });
        const toolCall = { id: 'c1', type: 'function' as const, function: { name: 'f', arguments: '{}' } };
        const result = await deepseek.generate(
            {
                model: 'deepseek-pro',
                messages: [
                    { role: 'user', content: 'go' },
                    { role: 'assistant', content: null, reasoning: 'first', tool_calls: [toolCall] },
                    { role: 'tool', tool_call_id: 'c1', content: 'done' },
                ],
                tools: [{ type: 'function', function: { name: 'f' } }],
            },
            { signal: new AbortController().signal, extra: {} },
        );
        expect(result).toMatchObject({
            ok: true,
            data: { reasoning: 'again', finishReason: 'tool_calls', toolCalls: [{ id: 'c2' }] },
        });
        expect(bodies[0]?.messages).toEqual([
            { role: 'user', content: 'go' },
            { role: 'assistant', content: null, reasoning_content: 'first', tool_calls: [toolCall] },
            { role: 'tool', tool_call_id: 'c1', content: 'done' },
        ]);
    });
});

describe('DEEPSEEK_SUPPORT', () => {
    it('supports images, tools, reasoning and JSON through the prompt', () => {
        expect(DEEPSEEK_SUPPORT).toEqual({ images: true, tools: true, reasoning: true, json: 'prompt' });
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
                model: 'deepseek-flash-0901',
                choices: [{ finish_reason: finishReason, message: { content, reasoning_content: 'why' } }],
                usage: { prompt_tokens: 3, completion_tokens: 4 },
            }),
        );
}

/** Creates test LLM instance wrapping DeepSeek provider with queued responses. */
function deepseekLlm(...answers: (() => Response)[]) {
    const fake = queueFetch(...answers);
    const llm = createLlm({
        defaults: { temperature: 0.2 },
        limiterOptions: { maxRetries: 1, retryDelayMs: 1 },
        providers: {
            deepseek: {
                adapter: createDeepseekProvider({ apiKey: 'k', fetch: fake.fetch }),
                defaults: { max_tokens: 100 },
                extra: { user: 'u' },
                models: {
                    flash: {
                        model: 'deepseek-flash',
                        defaults: { top_p: 0.9 },
                        capabilities: { json: true, reasoning: { supported: true, default: 'high' } },
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

const settings: LlmSettings = {
    name: 'flash',
    provider: 'deepseek',
    model: 'deepseek-flash',
    temperature: 0.2,
    top_p: 0.9,
    max_tokens: 100,
    reasoning_effort: 'high',
    jsonMode: 'prompt',
    extra: { user: 'u' },
};

const names = { name: 'flash', provider: 'deepseek' };

describe('DeepSeek through createLlm: settings, attempts and the answer of failed calls', () => {
    it('reports the settings the call was sent with and one attempt', async () => {
        const { llm, bodies } = deepseekLlm(reply('{"kind":"noun"}'));
        const result = await llm.flash('word?', { response_format: format });
        expect(result).toEqual({
            ok: true,
            data: {
                text: '{"kind":"noun"}',
                json: { kind: 'noun' },
                reasoning: 'why',
                model: 'deepseek-flash-0901',
                usage: { inputTokens: 3, outputTokens: 4, reasoningTokens: null },
                finishReason: 'stop',
                ms: expect.any(Number),
                ...names,
                settings,
                attempts: 1,
            },
        });
        expect(bodies[0]).toMatchObject({ response_format: { type: 'json_object' }, user: 'u' });
    });

    it('retries an answer that does not match the schema and counts both attempts', async () => {
        const { llm, bodies } = deepseekLlm(reply('{"kind":1}'), reply('{"kind":"verb"}'));
        const result = await llm.flash('word?', { response_format: format });
        expect(result).toMatchObject({ ok: true, data: { json: { kind: 'verb' }, settings, attempts: 2 } });
        expect(bodies).toHaveLength(2);
    });

    it('puts the last unusable answer with reasoning and parsed json on invalid_json after the retries', async () => {
        const { llm, bodies } = deepseekLlm(reply('{"kind":1}'));
        const result = await llm.flash('word?', { response_format: format });
        const usage = { inputTokens: 3, outputTokens: 4, reasoningTokens: null };
        expect(result).toMatchObject({
            ok: false,
            error: { code: 'invalid_json', retryable: true, text: '{"kind":1}', ...names, settings, attempts: 2 },
        });
        expect(result.ok ? undefined : result.error.output).toEqual({
            text: '{"kind":1}',
            json: { kind: 1 },
            reasoning: 'why',
            model: 'deepseek-flash-0901',
            usage,
            finishReason: 'stop',
            ms: expect.any(Number),
            ...names,
        });
        expect(bodies).toHaveLength(2);
        const notJson = await deepseekLlm(reply('no json')).llm.flash('word?', { response_format: format });
        expect(notJson.ok ? undefined : notJson.error.output).toMatchObject({ text: 'no json', reasoning: 'why' });
        expect(notJson.ok ? undefined : notJson.error.output).not.toHaveProperty('json');
    });

    it('puts the answer with reasoning on truncated and blocked errors after one attempt', async () => {
        for (const [reason, code] of [
            ['length', 'truncated'],
            ['content_filter', 'blocked'],
        ] as const) {
            const { llm, bodies } = deepseekLlm(reply('{"kind', reason));
            const result = await llm.flash('word?', { response_format: format });
            expect(result).toMatchObject({
                ok: false,
                error: {
                    code,
                    text: '{"kind',
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
        const { llm } = deepseekLlm(() => new Response('{"error":{"message":"bad"}}', { status: 400 }));
        const result = await llm.flash('hi');
        const { jsonMode: _jsonMode, ...withoutJson } = settings;
        expect(result).toEqual({
            ok: false,
            error: {
                code: 'http',
                status: 400,
                message: 'HTTP 400: bad',
                retryable: false,
                ...names,
                settings: withoutJson,
                attempts: 1,
            },
        });
    });
});
