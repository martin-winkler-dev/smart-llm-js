// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'bun:test';
import type { LlmProviderRequest, LlmProviderSupport } from '@smart-llm/core';
import { createOpenAiCompatibleProvider, type OpenAiCompatibleOptions, statusError } from '../src/index.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Recorded HTTP request sent to the mock fetch implementation. */
type Sent = { url: string; init: RequestInit; body: Record<string, unknown> | undefined };

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

const supports: LlmProviderSupport = { images: true, tools: true, reasoning: true, json: 'native' };

const request: LlmProviderRequest = {
    model: 'm',
    messages: [{ role: 'user', content: 'hi' }],
    max_tokens: 64,
    reasoning_effort: 'low',
};

const call = { signal: new AbortController().signal, extra: {} };

// -------------------------------------------------------------------
// --- 3. Private Functions
// -------------------------------------------------------------------

/** Creates fake queued fetch implementation recording requests. */
function fakeFetch(...answers: (() => Response)[]): { fetch: typeof fetch; sent: Sent[] } {
    const sent: Sent[] = [];
    async function fetcher(url: string | URL | Request, init: RequestInit = {}): Promise<Response> {
        sent.push({
            url: String(url),
            init,
            body: init.body === undefined ? undefined : JSON.parse(String(init.body)),
        });
        const next = answers.length > 1 ? answers.shift() : answers[0];
        return (next as () => Response)();
    }
    return { fetch: fetcher as typeof fetch, sent };
}

/** Returns mock response factory with JSON payload. */
function json(body: unknown, status = 200, headers: Record<string, string> = {}): () => Response {
    return () => new Response(JSON.stringify(body), { status, headers });
}

/** Returns mock completion response factory. */
function completion(finishReason = 'stop', message: Record<string, unknown> = { content: 'hello' }): () => Response {
    return json({
        model: 'served-model',
        choices: [{ finish_reason: finishReason, message }],
        usage: { prompt_tokens: 12, completion_tokens: 30, completion_tokens_details: { reasoning_tokens: 20 } },
    });
}

describe('createOpenAiCompatibleProvider', () => {
    it('posts request with mapped maxTokensField, extra, mapBody, and bearer token', async () => {
        const fake = fakeFetch(completion());
        const provider = createOpenAiCompatibleProvider({
            name: 'p',
            baseUrl: 'https://api.example.com/v1/',
            apiKey: 'key',
            supports,
            maxTokensField: 'max_tokens',
            mapBody: (body) => ({ ...body, mapped: true }),
            fetch: fake.fetch,
        });
        await provider.generate(request, { ...call, extra: { user: 'u' } });
        const sent = fake.sent[0] as Sent;
        expect(sent.url).toBe('https://api.example.com/v1/chat/completions');
        expect((sent.init.headers as Record<string, string>).authorization).toBe('Bearer key');
        expect(sent.body).toEqual({
            model: 'm',
            messages: [{ role: 'user', content: 'hi' }],
            max_tokens: 64,
            reasoning_effort: 'low',
            user: 'u',
            mapped: true,
        });
    });

    it('sends a developer message as system', async () => {
        const fake = fakeFetch(completion());
        const provider = createOpenAiCompatibleProvider({
            name: 'p',
            baseUrl: 'http://x',
            supports,
            fetch: fake.fetch,
        });
        await provider.generate(
            {
                model: 'm',
                messages: [
                    { role: 'developer', content: 'Be brief.' },
                    { role: 'user', content: 'hi' },
                ],
            },
            call,
        );
        expect(fake.sent[0]?.body?.messages).toEqual([
            { role: 'system', content: 'Be brief.' },
            { role: 'user', content: 'hi' },
        ]);
    });

    it('sends max_completion_tokens by default and no authorization without a key', async () => {
        const fake = fakeFetch(completion());
        const provider = createOpenAiCompatibleProvider({
            name: 'p',
            baseUrl: 'http://x',
            supports,
            fetch: fake.fetch,
        });
        await provider.generate(request, call);
        expect(fake.sent[0]?.body).toMatchObject({ max_completion_tokens: 64 });
        expect(fake.sent[0]?.body?.max_tokens).toBeUndefined();
        expect((fake.sent[0]?.init.headers as Record<string, string> | undefined)?.authorization).toBeUndefined();
    });

    it('reads text, reasoning and usage with output tokens without reasoning tokens', async () => {
        const fake = fakeFetch(completion('stop', { content: 'hello', reasoning_content: 'thought' }));
        const provider = createOpenAiCompatibleProvider({
            name: 'p',
            baseUrl: 'http://x',
            supports,
            fetch: fake.fetch,
        });
        expect(await provider.generate(request, call)).toMatchObject({
            ok: true,
            data: {
                text: 'hello',
                reasoning: 'thought',
                model: 'served-model',
                usage: { inputTokens: 12, outputTokens: 10, reasoningTokens: 20 },
                finishReason: 'stop',
            },
        });
    });

    it('sends tools and tool turns without extra_content, with the reasoning under reasoningMessageField', async () => {
        const fake = fakeFetch(completion());
        const provider = createOpenAiCompatibleProvider({
            name: 'p',
            baseUrl: 'http://x',
            supports,
            reasoningMessageField: 'reasoning_content',
            fetch: fake.fetch,
        });
        const tools = [
            { type: 'function' as const, function: { name: 'get_weather', parameters: { type: 'object' } } },
        ];
        const toolCall = { id: 'c1', type: 'function' as const, function: { name: 'get_weather', arguments: '{}' } };
        await provider.generate(
            {
                model: 'm',
                messages: [
                    { role: 'user', content: 'weather?' },
                    {
                        role: 'assistant',
                        content: null,
                        reasoning: 'look it up',
                        tool_calls: [{ ...toolCall, extra_content: { google: { thought_signature: 's' } } }],
                    },
                    { role: 'tool', tool_call_id: 'c1', content: 'sunny' },
                ],
                tools,
                tool_choice: 'auto',
            },
            call,
        );
        expect(fake.sent[0]?.body).toEqual({
            model: 'm',
            messages: [
                { role: 'user', content: 'weather?' },
                { role: 'assistant', content: null, reasoning_content: 'look it up', tool_calls: [toolCall] },
                { role: 'tool', tool_call_id: 'c1', content: 'sunny' },
            ],
            tools,
            tool_choice: 'auto',
        });
    });

    it('leaves out the reasoning of assistant messages without reasoningMessageField', async () => {
        const fake = fakeFetch(completion());
        const provider = createOpenAiCompatibleProvider({
            name: 'p',
            baseUrl: 'http://x',
            supports,
            fetch: fake.fetch,
        });
        await provider.generate(
            {
                model: 'm',
                messages: [
                    { role: 'user', content: 'hi' },
                    { role: 'assistant', content: 'hello', reasoning: 'greet' },
                    { role: 'user', content: 'bye' },
                ],
            },
            call,
        );
        expect(fake.sent[0]?.body?.messages).toMatchObject([{}, { role: 'assistant', content: 'hello' }, {}]);
        expect((fake.sent[0]?.body?.messages as Record<string, unknown>[] | undefined)?.[1]?.reasoning).toBeUndefined();
    });

    it('reads tool calls with finish reason tool_calls as success, generating missing ids', async () => {
        const fake = fakeFetch(
            completion('tool_calls', {
                content: null,
                tool_calls: [
                    { id: 'c1', type: 'function', function: { name: 'a', arguments: '{"x":1}' } },
                    { type: 'function', function: { name: 'b', arguments: { y: 2 } } },
                    { type: 'function', function: {} },
                ],
            }),
        );
        const provider = createOpenAiCompatibleProvider({
            name: 'p',
            baseUrl: 'http://x',
            supports,
            fetch: fake.fetch,
        });
        const result = await provider.generate(request, call);
        expect(result).toMatchObject({
            ok: true,
            data: {
                text: '',
                finishReason: 'tool_calls',
                toolCalls: [
                    { id: 'c1', type: 'function', function: { name: 'a', arguments: '{"x":1}' } },
                    { type: 'function', function: { name: 'b', arguments: '{"y":2}' } },
                ],
            },
        });
        const calls = result.ok ? (result.data.toolCalls ?? []) : [];
        expect(calls).toHaveLength(2);
        expect(calls[1]?.id).toMatch(/^call_[0-9a-f]{32}$/);
    });

    it('fails with a retryable unexpected for finish reason tool_calls without a usable tool call', async () => {
        for (const message of [
            { content: null },
            { content: null, tool_calls: [{ type: 'function', function: {} }] },
        ]) {
            const fake = fakeFetch(completion('tool_calls', message));
            const provider = createOpenAiCompatibleProvider({
                name: 'p',
                baseUrl: 'http://x',
                supports,
                fetch: fake.fetch,
            });
            expect(await provider.generate(request, call)).toMatchObject({
                ok: false,
                error: { code: 'unexpected', retryable: true, output: { text: '', finishReason: 'tool_calls' } },
            });
        }
    });

    it('fails with retryable unexpected and output for non-object tool call arguments', async () => {
        for (const args of ['{"city": "Graz"', '[1]']) {
            const fake = fakeFetch(
                completion('tool_calls', {
                    content: null,
                    tool_calls: [{ id: 'c1', type: 'function', function: { name: 'get_weather', arguments: args } }],
                }),
            );
            const provider = createOpenAiCompatibleProvider({
                name: 'p',
                baseUrl: 'http://x',
                supports,
                fetch: fake.fetch,
            });
            expect(await provider.generate(request, call)).toMatchObject({
                ok: false,
                error: {
                    code: 'unexpected',
                    retryable: true,
                    output: { toolCalls: [{ function: { name: 'get_weather', arguments: args } }] },
                },
            });
        }
    });

    it('maps statuses, finish reasons and fetch failures to errors', async () => {
        const cases: [() => Response, Record<string, unknown>][] = [
            [
                json({ error: { message: 'slow down' } }, 429, { 'retry-after': '2' }),
                { code: 'rate_limited', retryable: true, retryAfterMs: 2000, message: 'HTTP 429: slow down' },
            ],
            [json({ error: 'busy' }, 503), { code: 'http', status: 503, retryable: true, message: 'HTTP 503: busy' }],
            [() => new Response('nope', { status: 400 }), { code: 'http', status: 400, retryable: false }],
            [
                json({ error: { message: 'flex tier at capacity' } }, 498),
                { code: 'http', status: 498, retryable: true },
            ],
            [
                json(
                    {
                        error: {
                            message: 'Generated JSON does not match the expected schema',
                            code: 'json_validate_failed',
                            failed_generation: '{"kind":',
                        },
                    },
                    400,
                ),
                { code: 'invalid_json', status: 400, retryable: false, text: '{"kind":' },
            ],
            [
                json({ error: { message: 'Failed to call a function', code: 'tool_use_failed' } }, 400),
                { code: 'http', status: 400, retryable: true, message: 'HTTP 400: Failed to call a function' },
            ],
            [
                json({ error: 'model "x" not found, try pulling it first' }, 404),
                {
                    code: 'http',
                    status: 404,
                    retryable: false,
                    message: 'HTTP 404: model "x" not found, try pulling it first',
                },
            ],
            [completion('length'), { code: 'truncated', text: 'hello', retryable: false }],
            [completion('content_filter'), { code: 'blocked', retryable: false }],
            [completion('function_call'), { code: 'unexpected', message: 'Finish reason function_call' }],
            [() => new Response('<html>'), { code: 'unexpected', message: 'The response is not JSON' }],
            [json({ choices: [] }), { code: 'unexpected', message: 'The response has no choice' }],
            [
                () => {
                    throw new TypeError('connection refused');
                },
                { code: 'network', retryable: true },
            ],
        ];
        for (const [answer, error] of cases) {
            const provider = createOpenAiCompatibleProvider({
                name: 'p',
                baseUrl: 'http://x',
                supports,
                fetch: fakeFetch(answer).fetch,
            });
            expect(await provider.generate(request, call)).toMatchObject({ ok: false, error });
        }
    });

    it('reports the caller abort as aborted', async () => {
        const controller = new AbortController();
        controller.abort();
        const provider = createOpenAiCompatibleProvider({
            name: 'p',
            baseUrl: 'http://x',
            supports,
            fetch: (async (_url: string, init: RequestInit) => {
                init.signal?.throwIfAborted();
                return new Response('{}');
            }) as typeof fetch,
        });
        expect(await provider.generate(request, { signal: controller.signal, extra: {} })).toMatchObject({
            ok: false,
            error: { code: 'aborted', retryable: false },
        });
    });

    it('lists models from /models', async () => {
        const fake = fakeFetch(json({ data: [{ id: 'a' }, { id: 'b' }, { name: 'no id' }] }));
        const provider = createOpenAiCompatibleProvider({
            name: 'p',
            baseUrl: 'http://x/v1',
            supports,
            fetch: fake.fetch,
        });
        expect(await provider.listModels?.()).toEqual({ ok: true, data: [{ id: 'a' }, { id: 'b' }] });
        expect(fake.sent[0]?.url).toBe('http://x/v1/models');
    });

    it('rejects batch and web_search as unsupported without sending request', async () => {
        const fake = fakeFetch(completion());
        const provider = createOpenAiCompatibleProvider({
            name: 'p',
            baseUrl: 'http://x',
            supports: { ...supports, webSearch: true, batch: true } as never,
            fetch: fake.fetch,
        });
        expect(provider.supports).toEqual(supports);
        expect(createOpenAiCompatibleProvider({ name: 'q', baseUrl: 'http://x' }).supports).toEqual({});
        for (const field of [{ batch: true }, { web_search: true }]) {
            expect(await provider.generate({ ...request, ...field }, call)).toMatchObject({
                ok: false,
                error: { code: 'unsupported', retryable: false },
            });
        }
        expect(fake.sent).toHaveLength(0);
    });

    it('returns unexpected instead of throwing on mapBody errors, null body, or bad model list', async () => {
        const options = { name: 'p', baseUrl: 'http://x', supports };
        const throwing = createOpenAiCompatibleProvider({
            ...options,
            fetch: fakeFetch(completion()).fetch,
            mapBody() {
                throw new Error('bad body');
            },
        });
        expect(await throwing.generate(request, call)).toEqual({
            ok: false,
            error: { code: 'unexpected', message: 'bad body', retryable: false },
        });
        const cyclic: Record<string, unknown> = {};
        cyclic.self = cyclic;
        const plain = createOpenAiCompatibleProvider({ ...options, fetch: fakeFetch(json(null)).fetch });
        expect(await plain.generate(request, { ...call, extra: { cyclic } })).toMatchObject({
            ok: false,
            error: { code: 'unexpected', retryable: false },
        });
        expect(await plain.generate(request, call)).toMatchObject({
            ok: false,
            error: { code: 'unexpected', message: 'The response has no choice' },
        });
        expect(await plain.listModels?.()).toEqual({ ok: true, data: [] });
        const odd = createOpenAiCompatibleProvider({ ...options, fetch: fakeFetch(json({ data: 'x' })).fetch });
        expect(await odd.listModels?.()).toEqual({ ok: true, data: [] });
    });
});

describe('answers of failed calls', () => {
    const reasoned = { content: 'hel', reasoning_content: 'thought' };

    /** Creates test provider instance with queued responses. */
    function provider(...answers: (() => Response)[]) {
        return createOpenAiCompatibleProvider({
            name: 'p',
            baseUrl: 'http://x',
            supports,
            fetch: fakeFetch(...answers).fetch,
        });
    }

    it('puts the read answer with reasoning as output on truncated, blocked and unexpected finish reasons', async () => {
        const cases = [
            ['length', 'truncated', 'The answer hit the output token limit'],
            ['content_filter', 'blocked', 'The answer was blocked by a content filter'],
            ['function_call', 'unexpected', 'Finish reason function_call'],
        ] as const;
        for (const [reason, code, message] of cases) {
            const result = await provider(completion(reason, reasoned)).generate(request, call);
            const usage = { inputTokens: 12, outputTokens: 10, reasoningTokens: 20 };
            const ms = expect.any(Number);
            expect(result).toEqual({
                ok: false,
                error: {
                    code,
                    message,
                    retryable: false,
                    text: 'hel',
                    model: 'served-model',
                    usage,
                    ms,
                    output: {
                        text: 'hel',
                        reasoning: 'thought',
                        model: 'served-model',
                        usage,
                        finishReason: reason,
                        ms,
                    },
                },
            });
        }
    });

    it('puts the failed generation of json_validate_failed as output, without usage or finish reason', async () => {
        const failed = json(
            {
                error: {
                    message: 'Generated JSON does not match the expected schema',
                    code: 'json_validate_failed',
                    failed_generation: '{"kind":',
                },
            },
            400,
        );
        const result = await provider(failed).generate(request, call);
        expect(result).toMatchObject({ ok: false, error: { code: 'invalid_json', status: 400, text: '{"kind":' } });
        expect(result.ok ? undefined : result.error.output).toEqual({
            text: '{"kind":',
            model: 'm',
            usage: { inputTokens: null, outputTokens: null, reasoningTokens: null },
            finishReason: null,
            ms: expect.any(Number),
        });
    });

    it('puts no output on transport errors and other HTTP errors', async () => {
        const answers = [
            () => {
                throw new TypeError('connection refused');
            },
            json({ error: 'busy' }, 500),
            json({ error: { message: 'bad', code: 'invalid_request_error' } }, 400),
        ];
        for (const answer of answers) {
            const result = await provider(answer).generate(request, call);
            expect(result.ok).toBe(false);
            expect(result.ok ? undefined : result.error).not.toHaveProperty('output');
        }
    });
});

describe('embeddings', () => {
    it('embeds all texts with one /embeddings call, sorted by index, with dimensions and token usage', async () => {
        const fake = fakeFetch(
            json({
                model: 'bge-m3',
                data: [
                    { object: 'embedding', index: 1, embedding: [0, 1] },
                    { object: 'embedding', index: 0, embedding: [1, 0] },
                ],
                usage: { prompt_tokens: 7, total_tokens: 7 },
            }),
        );
        const provider = createOpenAiCompatibleProvider({
            name: 'p',
            baseUrl: 'http://box/v1',
            supports: { embeddings: true },
            fetch: fake.fetch,
        });
        const result = await provider.embed?.(
            { model: 'bge', input: [{ text: 'a' }, { text: 'b' }], dimensions: 2 },
            { signal: call.signal },
        );
        expect(result).toMatchObject({
            ok: true,
            data: {
                embeddings: [
                    [1, 0],
                    [0, 1],
                ],
                model: 'bge-m3',
                usage: { inputTokens: 7 },
            },
        });
        expect(fake.sent[0]?.url).toBe('http://box/v1/embeddings');
        expect(fake.sent[0]?.body).toEqual({ model: 'bge', input: ['a', 'b'], dimensions: 2 });
    });

    it('reports null input tokens for a negative or non-finite embeddings prompt_tokens', async () => {
        for (const tokens of ['-1', '1e400']) {
            const body = `{"data":[{"index":0,"embedding":[1]}],"usage":{"prompt_tokens":${tokens}}}`;
            const fake = fakeFetch(() => new Response(body));
            const provider = createOpenAiCompatibleProvider({
                name: 'p',
                baseUrl: 'http://box',
                supports: { embeddings: true },
                fetch: fake.fetch,
            });
            const result = await provider.embed?.({ model: 'm', input: [{ text: 'a' }] }, { signal: call.signal });
            expect(result).toMatchObject({ ok: true, data: { usage: { inputTokens: null } } });
        }
    });

    it('rejects unsupported embeddings, task, or title, and maps invalid responses/errors', async () => {
        const fake = fakeFetch(json({ data: [{ index: 0, embedding: [1] }] }));
        const signal = { signal: call.signal };
        const plain = createOpenAiCompatibleProvider({ name: 'p', baseUrl: 'http://box', fetch: fake.fetch });
        expect(await plain.embed?.({ model: 'm', input: [{ text: 'a' }] }, signal)).toEqual({
            ok: false,
            error: { code: 'unsupported', message: 'p does not support embeddings', retryable: false },
        });
        const provider = createOpenAiCompatibleProvider({
            name: 'p',
            baseUrl: 'http://box',
            supports: { embeddings: true },
            fetch: fake.fetch,
        });
        const unsupported = {
            code: 'unsupported' as const,
            message: 'p embeddings take no task or title',
            retryable: false,
        };
        expect(await provider.embed?.({ model: 'm', input: [{ text: 'a' }], task: 'query' }, signal)).toEqual({
            ok: false,
            error: unsupported,
        });
        expect(await provider.embed?.({ model: 'm', input: [{ text: 'a', title: 'T' }] }, signal)).toEqual({
            ok: false,
            error: unsupported,
        });
        expect(fake.sent).toHaveLength(0);
        expect(await provider.embed?.({ model: 'm', input: [{ text: 'a' }] }, signal)).toMatchObject({
            ok: true,
            data: { embeddings: [[1]], model: 'm', usage: { inputTokens: null } },
        });
        const bad = [
            json({ data: [{ index: 0, embedding: [1] }] }),
            json({
                data: [
                    { index: 0, embedding: [] },
                    { index: 1, embedding: [1] },
                ],
            }),
            json({
                data: [
                    { index: 0, embedding: ['x'] },
                    { index: 1, embedding: [1] },
                ],
            }),
            json({ embeddings: [[1], [2]] }),
            json({ error: { message: "model 'm' not found" } }, 404),
            json({ error: 'busy' }, 503),
        ];
        const results = [];
        for (const answer of bad) {
            const one = createOpenAiCompatibleProvider({
                name: 'p',
                baseUrl: 'http://box',
                supports: { embeddings: true },
                fetch: fakeFetch(answer).fetch,
            });
            results.push(await one.embed?.({ model: 'm', input: [{ text: 'a' }, { text: 'b' }] }, signal));
        }
        const noVectors = {
            code: 'unexpected',
            message: 'The /embeddings response has no 2 vectors',
            retryable: false,
        };
        expect(results).toMatchObject([
            { ok: false, error: noVectors },
            { ok: false, error: noVectors },
            { ok: false, error: noVectors },
            { ok: false, error: noVectors },
            {
                ok: false,
                error: { code: 'http', status: 404, retryable: false, message: "HTTP 404: model 'm' not found" },
            },
            { ok: false, error: { code: 'http', status: 503, retryable: true } },
        ]);
    });
});

describe('statusError', () => {
    it('reads retry-after as seconds or an HTTP date', () => {
        expect(statusError(429, 'x', '1.5')).toMatchObject({ retryAfterMs: 1500 });
        const later = new Date(Date.now() + 60_000).toUTCString();
        expect((statusError(503, 'x', later).retryAfterMs ?? 0) > 50_000).toBe(true);
        expect(statusError(408, 'x', null)).toEqual({ code: 'http', message: 'x', retryable: true, status: 408 });
    });
});

describe('caller abort and internal timeout', () => {
    /** Fetch that rejects with the abort reason once the signal fires. */
    function hangingFetch(_url: string, init: RequestInit): Promise<Response> {
        return new Promise((_resolve, reject) => {
            init.signal?.addEventListener('abort', () => reject(init.signal?.reason));
        });
    }
    const hanging = hangingFetch as typeof fetch;

    it('reports a caller AbortSignal.timeout as aborted on generate, embed and listModels', async () => {
        const provider = createOpenAiCompatibleProvider({
            name: 'p',
            baseUrl: 'http://x',
            supports: { embeddings: true },
            fetch: hanging,
        });
        const aborted = { ok: false, error: { code: 'aborted', retryable: false } };
        expect(await provider.generate(request, { signal: AbortSignal.timeout(5), extra: {} })).toMatchObject(aborted);
        expect(
            await provider.embed?.({ model: 'm', input: [{ text: 'a' }] }, { signal: AbortSignal.timeout(5) }),
        ).toMatchObject(aborted);
        expect(await provider.listModels?.({ signal: AbortSignal.timeout(5) })).toMatchObject(aborted);
    });

    it('reports a caller abort with a TimeoutError reason as aborted', async () => {
        const controller = new AbortController();
        controller.abort(new DOMException('caller deadline', 'TimeoutError'));
        const provider = createOpenAiCompatibleProvider({
            name: 'p',
            baseUrl: 'http://x',
            fetch: ((_url: string, init: RequestInit) =>
                init.signal?.aborted ? Promise.reject(init.signal.reason) : hanging(_url, init)) as typeof fetch,
        });
        expect(await provider.generate(request, { signal: controller.signal, extra: {} })).toMatchObject({
            ok: false,
            error: { code: 'aborted', retryable: false },
        });
    });

    it('reports the internal timeoutMs as a retryable timeout on generate, embed and listModels', async () => {
        const provider = createOpenAiCompatibleProvider({
            name: 'p',
            baseUrl: 'http://x',
            supports: { embeddings: true },
            timeoutMs: 5,
            fetch: hanging,
        });
        const timeout = { ok: false, error: { code: 'timeout', retryable: true } };
        expect(await provider.generate(request, call)).toMatchObject(timeout);
        expect(await provider.embed?.({ model: 'm', input: [{ text: 'a' }] }, { signal: call.signal })).toMatchObject(
            timeout,
        );
        expect(await provider.listModels?.({ signal: call.signal })).toMatchObject(timeout);
        expect(await provider.listModels?.()).toMatchObject(timeout);
    });
});

describe('tool call normalization', () => {
    it('reads empty-string arguments as an empty object and skips tool calls without a name', async () => {
        const fake = fakeFetch(
            completion('tool_calls', {
                content: null,
                tool_calls: [
                    { id: 'c1', type: 'function', function: { name: 'now', arguments: '' } },
                    { id: 'c2', type: 'function', function: { name: '', arguments: '{}' } },
                ],
            }),
        );
        const provider = createOpenAiCompatibleProvider({ name: 'p', baseUrl: 'http://x', fetch: fake.fetch });
        const result = await provider.generate(request, call);
        expect(result).toMatchObject({ ok: true, data: { finishReason: 'tool_calls' } });
        expect(result.ok ? result.data.toolCalls : undefined).toEqual([
            { id: 'c1', type: 'function', function: { name: 'now', arguments: '{}' } },
        ]);
    });

    it('sends empty-string history arguments as an empty object and leaves out empty tool_calls', async () => {
        const fake = fakeFetch(completion());
        const provider = createOpenAiCompatibleProvider({ name: 'p', baseUrl: 'http://x', fetch: fake.fetch });
        await provider.generate(
            {
                model: 'm',
                messages: [
                    { role: 'user', content: 'time?' },
                    {
                        role: 'assistant',
                        content: null,
                        tool_calls: [{ id: 'c1', type: 'function', function: { name: 'now', arguments: '' } }],
                    },
                    { role: 'tool', tool_call_id: 'c1', content: 'noon' },
                    { role: 'assistant', content: 'It is noon.', tool_calls: [] },
                    { role: 'user', content: 'thanks' },
                ],
            },
            call,
        );
        expect(fake.sent[0]?.body?.messages).toEqual([
            { role: 'user', content: 'time?' },
            {
                role: 'assistant',
                content: null,
                tool_calls: [{ id: 'c1', type: 'function', function: { name: 'now', arguments: '{}' } }],
            },
            { role: 'tool', tool_call_id: 'c1', content: 'noon' },
            { role: 'assistant', content: 'It is noon.' },
            { role: 'user', content: 'thanks' },
        ]);
    });
});

describe('malformed completions', () => {
    /** Creates provider answering every call with given body. */
    function answering(body: unknown) {
        return createOpenAiCompatibleProvider({ name: 'p', baseUrl: 'http://x', fetch: fakeFetch(json(body)).fetch });
    }

    it('fails with unexpected for a choice or message that is no object', async () => {
        for (const choices of [[{ finish_reason: 'stop', message: null }], [{ message: 'str' }], [null], 'abc']) {
            expect(await answering({ choices }).generate(request, call)).toEqual({
                ok: false,
                error: { code: 'unexpected', message: 'The response has no choice', retryable: false },
            });
        }
    });

    it('falls back to the requested model for an empty or non-string model', async () => {
        for (const model of ['', 42, null]) {
            const body = { model, choices: [{ finish_reason: 'stop', message: { content: 'ok' } }] };
            expect(await answering(body).generate(request, call)).toMatchObject({ ok: true, data: { model: 'm' } });
        }
    });

    it('reads non-numeric usage fields as null and clamps output tokens at 0', async () => {
        const message = { content: 'ok' };
        const odd = {
            choices: [{ finish_reason: 'stop', message }],
            usage: {
                prompt_tokens: '5',
                completion_tokens: '10',
                completion_tokens_details: { reasoning_tokens: '3' },
            },
        };
        expect(await answering(odd).generate(request, call)).toMatchObject({
            ok: true,
            data: { usage: { inputTokens: null, outputTokens: null, reasoningTokens: null } },
        });
        const exclusive = {
            choices: [{ finish_reason: 'stop', message }],
            usage: { prompt_tokens: 5, completion_tokens: 10, completion_tokens_details: { reasoning_tokens: 300 } },
        };
        expect(await answering(exclusive).generate(request, call)).toMatchObject({
            ok: true,
            data: { usage: { inputTokens: 5, outputTokens: 0, reasoningTokens: 300 } },
        });
        const negative = {
            choices: [{ finish_reason: 'stop', message }],
            usage: { prompt_tokens: -1, completion_tokens: -10, completion_tokens_details: { reasoning_tokens: -3 } },
        };
        expect(await answering(negative).generate(request, call)).toMatchObject({
            ok: true,
            data: { usage: { inputTokens: null, outputTokens: null, reasoningTokens: null } },
        });
    });
});

describe('options snapshot', () => {
    it('ignores mutations of options after creation', async () => {
        const fake = fakeFetch(completion('stop', { content: 'ok', thought: 'first' }));
        const reasoningFields = ['thought'];
        const supports: { embeddings?: true } = { embeddings: true };
        const options: OpenAiCompatibleOptions = {
            name: 'a',
            baseUrl: 'http://x',
            supports,
            maxTokensField: 'max_completion_tokens',
            reasoningFields,
            fetch: fake.fetch,
        };
        const provider = createOpenAiCompatibleProvider(options);
        options.name = 'b';
        options.maxTokensField = 'max_tokens';
        options.reasoningMessageField = 'injected';
        options.mapBody = (body) => ({ ...body, mapped: true });
        delete supports.embeddings;
        reasoningFields[0] = 'other';
        expect(await provider.generate({ ...request, web_search: true }, call)).toMatchObject({
            error: { code: 'unsupported', message: 'a does not support web search' },
        });
        expect(await provider.generate(request, call)).toMatchObject({ ok: true, data: { reasoning: 'first' } });
        expect(fake.sent[0]?.body).toMatchObject({ max_completion_tokens: 64 });
        expect(fake.sent[0]?.body).not.toHaveProperty('max_tokens');
        expect(fake.sent[0]?.body).not.toHaveProperty('mapped');
        expect(provider.supports.embeddings).toBe(true);
        await provider.embed?.({ model: 'm', input: [{ text: 'a' }] }, { signal: call.signal });
        expect(fake.sent[1]?.url).toBe('http://x/embeddings');
        const history: LlmProviderRequest = {
            model: 'm',
            messages: [
                { role: 'user', content: 'hi' },
                { role: 'assistant', content: 'hello', reasoning: 'thinking' },
                { role: 'user', content: 'again' },
            ],
        };
        await provider.generate(history, call);
        expect(fake.sent[2]?.body?.messages).toEqual([
            { role: 'user', content: 'hi' },
            { role: 'assistant', content: 'hello' },
            { role: 'user', content: 'again' },
        ]);
    });

    it('falls back to max_completion_tokens for a null maxTokensField', async () => {
        const fake = fakeFetch(completion());
        const provider = createOpenAiCompatibleProvider({
            name: 'p',
            baseUrl: 'http://x',
            maxTokensField: null as never,
            fetch: fake.fetch,
        });
        await provider.generate(request, call);
        expect(fake.sent[0]?.body).toMatchObject({ max_completion_tokens: 64 });
        expect(fake.sent[0]?.body).not.toHaveProperty('null');
    });

    it('calls a method-style mapBody with options as this', async () => {
        const fake = fakeFetch(completion());
        const options = {
            name: 'p',
            baseUrl: 'http://x',
            tag: 'from-this',
            fetch: fake.fetch,
            mapBody(this: { tag: string }, body: Record<string, unknown>) {
                return { ...body, tag: this.tag };
            },
        };
        const provider = createOpenAiCompatibleProvider(options as OpenAiCompatibleOptions);
        expect(await provider.generate(request, call)).toMatchObject({ ok: true });
        expect(fake.sent[0]?.body).toMatchObject({ tag: 'from-this' });
    });
});
