// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'bun:test';
import { createLlm, type LlmResponseFormat, type LlmSettings } from '@smart-llm/core';
import { createOllamaProvider, fetchOllamaCapabilities, OLLAMA_SUPPORT } from '../src/index.js';

describe('createOllamaProvider', () => {
    it('posts to the local server without a key and reads reasoning', async () => {
        const sent: { url: string; init: RequestInit }[] = [];
        async function fetcher(url: string, init: RequestInit): Promise<Response> {
            sent.push({ url, init });
            return new Response(
                JSON.stringify({ choices: [{ finish_reason: 'stop', message: { content: 'ok', reasoning: 'hmm' } }] }),
            );
        }
        const ollama = createOllamaProvider({ fetch: fetcher as typeof fetch });
        const result = await ollama.generate(
            { model: 'qwen2.5:7b', messages: [{ role: 'user', content: 'hi' }], max_tokens: 5 },
            { signal: new AbortController().signal, extra: {} },
        );
        expect(result).toMatchObject({ ok: true, data: { text: 'ok', reasoning: 'hmm' } });
        expect(sent[0]?.url).toBe('http://localhost:11434/v1/chat/completions');
        expect((sent[0]?.init.headers as Record<string, string> | undefined)?.authorization).toBeUndefined();
        expect(JSON.parse(String(sent[0]?.init.body))).toMatchObject({ max_tokens: 5 });
    });
});

describe('Ollama tool calls', () => {
    it('sends tools without tool_choice, and no tools with tool_choice none', async () => {
        const bodies: Record<string, unknown>[] = [];
        async function fetcher(_url: string, init: RequestInit): Promise<Response> {
            bodies.push(JSON.parse(String(init.body)));
            return new Response(JSON.stringify({ choices: [{ finish_reason: 'stop', message: { content: 'ok' } }] }));
        }
        const ollama = createOllamaProvider({ fetch: fetcher as typeof fetch });
        const tools = [{ type: 'function' as const, function: { name: 'f' } }];
        const call = { signal: new AbortController().signal, extra: {} };
        const messages = [{ role: 'user' as const, content: 'hi' }];
        await ollama.generate({ model: 'qwen2.5:7b', messages, tools, tool_choice: 'required' }, call);
        await ollama.generate({ model: 'qwen2.5:7b', messages, tools, tool_choice: 'none' }, call);
        expect(bodies[0]).toEqual({ model: 'qwen2.5:7b', messages, tools });
        expect(bodies[1]).toEqual({ model: 'qwen2.5:7b', messages });
    });
});

describe('fetchOllamaCapabilities', () => {
    it('reads vision, tools, thinking and embedding from /api/tags; entries without capabilities get none', async () => {
        const urls: string[] = [];
        async function fetcher(url: string): Promise<Response> {
            urls.push(url);
            return new Response(
                JSON.stringify({
                    models: [
                        { name: 'qwen2.5:7b', capabilities: ['completion', 'tools'] },
                        { name: 'gemma3:4b', capabilities: ['completion', 'vision'] },
                        { name: 'gpt-oss:20b', capabilities: ['completion', 'tools', 'thinking'] },
                        { name: 'bge-m3:latest', capabilities: ['embedding'] },
                        { name: 'qwen3:8b' },
                    ],
                }),
            );
        }
        const capabilities = await fetchOllamaCapabilities({
            baseUrl: 'http://box:11434/v1',
            fetch: fetcher as typeof fetch,
        });
        expect(urls).toEqual(['http://box:11434/api/tags']);
        expect(capabilities).toEqual({
            ok: true,
            data: {
                'qwen2.5:7b': { tools: true },
                'gemma3:4b': { images: true },
                'gpt-oss:20b': { tools: true, reasoning: { supported: true } },
                'bge-m3:latest': { embeddings: true },
                'qwen3:8b': {},
            },
        });
    });

    it('returns http, network, timeout and unexpected errors instead of throwing', async () => {
        const answers: (() => Promise<Response>)[] = [
            async () => new Response('down', { status: 503 }),
            async () => {
                throw new TypeError('connection refused');
            },
            async () => {
                throw new DOMException('The operation timed out.', 'TimeoutError');
            },
            async () => new Response('not json'),
            async () => new Response(JSON.stringify({ models: 'x' })),
            async () => new Response('null'),
        ];
        const results = [];
        for (const fetcher of answers) {
            results.push(await fetchOllamaCapabilities({ fetch: fetcher as unknown as typeof fetch }));
        }
        expect(results).toMatchObject([
            { ok: false, error: { code: 'http', status: 503, retryable: true } },
            { ok: false, error: { code: 'network', message: 'connection refused', retryable: true } },
            { ok: false, error: { code: 'timeout', retryable: true } },
            { ok: false, error: { code: 'unexpected', retryable: false } },
            { ok: false, error: { code: 'unexpected', message: expect.stringContaining('model list') } },
            { ok: true, data: {} },
        ]);
    });
});

describe('OLLAMA_SUPPORT', () => {
    it('supports images, tools, reasoning, JSON as a grammar and embeddings', () => {
        expect(OLLAMA_SUPPORT).toEqual({
            images: true,
            tools: true,
            reasoning: true,
            json: 'grammar',
            embeddings: true,
        });
        expect(createOllamaProvider().supports).toEqual(OLLAMA_SUPPORT);
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
                model: 'qwen2.5:7b',
                choices: [{ finish_reason: finishReason, message: { content, reasoning: 'hmm' } }],
                usage: { prompt_tokens: 2, completion_tokens: 6 },
            }),
        );
}

/** Creates test LLM instance wrapping Ollama provider with queued responses. */
function ollamaLlm(...answers: (() => Response)[]) {
    const fake = queueFetch(...answers);
    const llm = createLlm({
        defaults: { seed: 1 },
        limiterOptions: { maxRetries: 2, retryDelayMs: 1, maxWaitMs: 50 },
        providers: {
            ollama: {
                adapter: createOllamaProvider({ fetch: fake.fetch }),
                defaults: { stop: ['END'] },
                models: { qwen: { model: 'qwen2.5:7b', defaults: { max_tokens: 64 }, capabilities: { json: true } } },
            },
        },
    });
    return { llm, bodies: fake.bodies };
}

const format: LlmResponseFormat = {
    type: 'json_schema',
    json_schema: {
        name: 'count',
        schema: { type: 'object', properties: { n: { type: 'number' } }, required: ['n'] },
    },
};

const names = { name: 'qwen', provider: 'ollama' };

const settings: LlmSettings = {
    ...names,
    model: 'qwen2.5:7b',
    seed: 1,
    stop: ['END'],
    max_tokens: 64,
    jsonMode: 'grammar',
};

describe('Ollama through createLlm: settings, attempts and the answer of failed calls', () => {
    it('reports the settings with JSON as a grammar and one attempt', async () => {
        const { llm, bodies } = ollamaLlm(reply('{"n":2}'));
        const result = await llm.qwen('count?', { response_format: format });
        expect(result).toEqual({
            ok: true,
            data: {
                text: '{"n":2}',
                json: { n: 2 },
                reasoning: 'hmm',
                model: 'qwen2.5:7b',
                usage: { inputTokens: 2, outputTokens: 6, reasoningTokens: null },
                finishReason: 'stop',
                ms: expect.any(Number),
                ...names,
                settings,
                attempts: 1,
            },
        });
        expect(bodies[0]).toMatchObject({ response_format: { type: 'json_schema' }, seed: 1, stop: ['END'] });
    });

    it('retries a schema mismatch and puts the last answer with reasoning and json on invalid_json', async () => {
        const { llm, bodies } = ollamaLlm(reply('{"n":"x"}'));
        const result = await llm.qwen('count?', { response_format: format });
        expect(result).toMatchObject({
            ok: false,
            error: { code: 'invalid_json', retryable: true, text: '{"n":"x"}', ...names, settings, attempts: 3 },
        });
        expect(result.ok ? undefined : result.error.output).toEqual({
            text: '{"n":"x"}',
            json: { n: 'x' },
            reasoning: 'hmm',
            model: 'qwen2.5:7b',
            usage: { inputTokens: 2, outputTokens: 6, reasoningTokens: null },
            finishReason: 'stop',
            ms: expect.any(Number),
            ...names,
        });
        expect(bodies).toHaveLength(3);
        const recovered = await ollamaLlm(reply('{"n":"x"}'), reply('{"n":3}')).llm.qwen('count?', {
            response_format: format,
        });
        expect(recovered).toMatchObject({ ok: true, data: { json: { n: 3 }, attempts: 2 } });
    });

    it('puts the answer with reasoning on truncated and blocked errors after one attempt', async () => {
        for (const [reason, code] of [
            ['length', 'truncated'],
            ['content_filter', 'blocked'],
        ] as const) {
            const { llm, bodies } = ollamaLlm(reply('{"n', reason));
            expect(await llm.qwen('count?', { response_format: format })).toMatchObject({
                ok: false,
                error: {
                    code,
                    ...names,
                    output: { text: '{"n', reasoning: 'hmm', finishReason: reason, ...names },
                    settings,
                    attempts: 1,
                },
            });
            expect(bodies).toHaveLength(1);
        }
    });

    it('puts no output but the settings on an HTTP error and a network failure', async () => {
        /** Returns mock 404 response for unknown model. */
        function notFound(): Response {
            return new Response('{"error":"model \\"qwen2.5:7b\\" not found"}', { status: 404 });
        }
        /** Simulates connection refused TypeError. */
        function refused(): Response {
            throw new TypeError('connection refused');
        }
        const { jsonMode: _jsonMode, ...withoutJson } = settings;
        // Network failure lock duration (>=1s) exceeds maxWaitMs, surfacing network error directly
        for (const answer of [notFound, refused]) {
            const result = await ollamaLlm(answer).llm.qwen('count?');
            expect(result).toMatchObject({ ok: false, error: { ...names, settings: withoutJson, attempts: 1 } });
            expect(result.ok ? undefined : result.error).not.toHaveProperty('output');
            expect(result.ok ? undefined : result.error.settings).toEqual(withoutJson);
        }
    });
});
