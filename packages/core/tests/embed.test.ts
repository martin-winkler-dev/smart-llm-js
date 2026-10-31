// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'bun:test';
import { createLlm } from '../src/llm.js';
import type {
    LlmEmbedRequest,
    LlmProvider,
    LlmProviderEmbedOutput,
    LlmProviderEmbedRequest,
    LlmProviderError,
    Result,
} from '../src/types.js';
import { createFakeProvider } from './fake-provider.js';

/** Reply handler for fake embedding provider. */
type EmbedReply = (request: LlmProviderEmbedRequest) => Result<LlmProviderEmbedOutput, LlmProviderError>;

/** Creates fake embedding provider returning length-based vectors and token usage. */
function createEmbedProvider(
    maxEmbedInputs?: number,
): LlmProvider & { requests: LlmProviderEmbedRequest[]; reply(...replies: EmbedReply[]): void } {
    const requests: LlmProviderEmbedRequest[] = [];
    let replies: EmbedReply[] = [vectors];
    return {
        ...createFakeProvider('vec', { embeddings: true }),
        requests,
        ...(maxEmbedInputs === undefined ? {} : { maxEmbedInputs }),
        reply(...next) {
            replies = next;
        },
        async embed(request) {
            requests.push(request);
            const reply = replies.length > 1 ? (replies.shift() as EmbedReply) : (replies[0] as EmbedReply);
            return reply(request);
        },
    };

    function vectors(request: LlmProviderEmbedRequest): Result<LlmProviderEmbedOutput, LlmProviderError> {
        return {
            ok: true,
            data: {
                embeddings: request.input.map((input) => [input.text.length, requests.length]),
                model: `${request.model}-served`,
                usage: { inputTokens: request.input.length },
                ms: 1,
            },
        };
    }
}

/** Creates LLM instance with `vectors` (maxInputs 2) and `chat` (no embeddings) models. */
function setup(adapter: LlmProvider, maxRetries = 0) {
    return createLlm({
        limiterOptions: { maxRetries, retryDelayMs: 1, maxRetryDelayMs: 5 },
        providers: {
            p: {
                adapter,
                models: {
                    vectors: { model: 'vec-id', capabilities: { embeddings: { supported: true, maxInputs: 2 } } },
                    chat: 'chat-id',
                },
            },
        },
    });
}

describe('embed', () => {
    it('splits the input into calls of maxInputs texts and returns the vectors in input order', async () => {
        const adapter = createEmbedProvider();
        const llm = setup(adapter);
        const result = await llm.embed({ model: 'vectors', input: ['a', 'bb', 'ccc', 'dddd', 'eeeee'] });
        expect(result).toMatchObject({
            ok: true,
            data: {
                name: 'vectors',
                provider: 'p',
                model: 'vec-id-served',
                usage: { inputTokens: 5 },
                calls: 3,
                attempts: 3,
            },
        });
        expect(result.ok && result.data.embeddings.map((vector) => vector[0])).toEqual([1, 2, 3, 4, 5]);
        expect(adapter.requests.map((request) => request.input.map((input) => input.text))).toEqual([
            ['a', 'bb'],
            ['ccc', 'dddd'],
            ['eeeee'],
        ]);
    });

    it('passes a single string, task, titles and dimensions to the provider', async () => {
        const adapter = createEmbedProvider();
        const llm = setup(adapter);
        await llm.embed({ model: 'vectors', input: 'one' });
        await llm.embed({
            model: 'vectors',
            input: [{ text: 'x', title: 'Methods' }, 'y'],
            task: 'document',
            dimensions: 256,
        });
        expect(adapter.requests).toEqual([
            { model: 'vec-id', input: [{ text: 'one' }] },
            {
                model: 'vec-id',
                input: [{ text: 'x', title: 'Methods' }, { text: 'y' }],
                task: 'document',
                dimensions: 256,
            },
        ]);
    });

    it('takes maxInputs from the adapter, else 64, and rejects more than the adapter allows', () => {
        const capped = createLlm({
            providers: {
                p: {
                    adapter: createEmbedProvider(10),
                    models: { v: { model: 'id', capabilities: { embeddings: true } } },
                },
            },
        });
        expect(capped.models[0]?.capabilities.embeddings).toEqual({ supported: true, maxInputs: 10 });
        const open = createLlm({
            providers: {
                p: {
                    adapter: createEmbedProvider(),
                    models: {
                        v: { model: 'id', capabilities: { embeddings: true } },
                        w: { model: 'id', capabilities: { embeddings: { supported: true, maxInputs: 512 } } },
                    },
                },
            },
        });
        expect(open.models.map((model) => model.capabilities.embeddings.maxInputs)).toEqual([64, 512]);
        const invalid: [() => unknown, string][] = [
            [
                () =>
                    createLlm({
                        providers: {
                            p: {
                                adapter: createEmbedProvider(10),
                                models: {
                                    v: {
                                        model: 'id',
                                        capabilities: { embeddings: { supported: true, maxInputs: 11 } },
                                    },
                                },
                            },
                        },
                    }),
                'Invalid value for option "providers.p.models.v.capabilities.embeddings.maxInputs": an integer from 1 to 10',
            ],
            [
                () =>
                    createLlm({
                        providers: {
                            p: {
                                adapter: createEmbedProvider(),
                                models: {
                                    v: { model: 'id', capabilities: { embeddings: { supported: true, maxInputs: 0 } } },
                                },
                            },
                        },
                    }),
                'Invalid value for option "providers.p.models.v.capabilities.embeddings.maxInputs": a positive integer',
            ],
            [
                () =>
                    createLlm({
                        providers: {
                            p: {
                                adapter: createEmbedProvider(),
                                models: { v: { model: 'id', capabilities: { embeddings: { maxInputs: 2 } as never } } },
                            },
                        },
                    }),
                'Invalid value for option "providers.p.models.v.capabilities.embeddings.supported": true or false',
            ],
            [
                () =>
                    createLlm({
                        providers: {
                            p: {
                                adapter: createEmbedProvider(),
                                models: {
                                    v: { model: 'id', capabilities: { embeddings: { supported: true, size: 2 } } },
                                },
                            },
                        },
                    } as never),
                'Option "providers.p.models.v.capabilities.embeddings.size" is not supported',
            ],
            [
                () =>
                    createLlm({
                        providers: {
                            p: {
                                adapter: createFakeProvider(),
                                models: { v: { model: 'id', capabilities: { embeddings: true } } },
                            },
                        },
                    }),
                'Invalid value for option "providers.p.models.v.capabilities.embeddings": the fake adapter does not support it',
            ],
            [
                () =>
                    createLlm({
                        providers: {
                            p: { adapter: createFakeProvider('f', { embeddings: true }), models: { v: 'id' } },
                        },
                    }),
                'Invalid value for option "providers.p.adapter": supports embeddings without embed',
            ],
            [
                () => createLlm({ providers: { p: { adapter: createEmbedProvider(0), models: { v: 'id' } } } }),
                'Invalid value for option "providers.p.adapter"',
            ],
        ];
        for (const [create, message] of invalid) {
            expect(create).toThrow(message);
        }
    });

    it('answers unsupported for a model without embeddings and invalid_request for invalid requests', async () => {
        const adapter = createEmbedProvider();
        const llm = setup(adapter);
        expect(await llm.embed({ model: 'chat', input: 'x' })).toEqual({
            ok: false,
            error: {
                code: 'unsupported',
                message: 'chat does not support embeddings',
                retryable: false,
                name: 'chat',
                provider: 'p',
            },
        });
        const invalid: [unknown, unknown, string][] = [
            [{ model: 'nope', input: 'x' }, {}, 'Invalid value for option "model": nope is not configured'],
            [{ model: 'vectors', input: [] }, {}, 'Invalid value for option "input": one or more texts'],
            [{ model: 'vectors', input: 3 }, {}, 'Invalid value for option "input": one or more texts'],
            [{ model: 'vectors', input: ['a', ''] }, {}, 'Invalid value for option "input.1": a non-empty text'],
            [
                { model: 'vectors', input: [{ text: '' }] },
                {},
                'Invalid value for option "input.0.text": a non-empty text',
            ],
            [
                { model: 'vectors', input: [{ text: 'a', title: '' }], task: 'document' },
                {},
                'Invalid value for option "input.0.title": a non-empty text',
            ],
            [{ model: 'vectors', input: [{ text: 'a', name: 'b' }] }, {}, 'Option "input.0.name" is not supported'],
            [
                { model: 'vectors', input: [{ text: 'a', title: 'T' }], task: 'query' },
                {},
                'Option "input.title" is not supported: only with task document',
            ],
            [
                { model: 'vectors', input: 'a', task: 'search' },
                {},
                'Invalid value for option "task": query, document, similarity, classification or clustering',
            ],
            [
                { model: 'vectors', input: 'a', dimensions: 0 },
                {},
                'Invalid value for option "dimensions": a positive integer',
            ],
            [{ model: 'vectors', input: 'a', temperature: 0 }, {}, 'Option "temperature" is not supported'],
            [{ model: 'vectors', input: 'a' }, { extra: {} }, 'Option "extra" is not supported'],
            [{ model: 'vectors', input: 'a' }, { signal: 'x' }, 'Invalid value for option "signal"'],
        ];
        for (const [request, options, message] of invalid) {
            const result = await llm.embed(request as LlmEmbedRequest<'vectors'>, options as never);
            expect(result).toMatchObject({ ok: false, error: { code: 'invalid_request', message, retryable: false } });
        }
        expect(adapter.requests).toHaveLength(0);
    });

    it('retries a retryable failure and counts the attempts', async () => {
        const adapter = createEmbedProvider();
        const llm = setup(adapter, 2);
        adapter.reply(
            () => ({ ok: false, error: { code: 'http', message: 'HTTP 503', retryable: true, status: 503 } }),
            (request) => ({
                ok: true,
                data: { embeddings: [[1]], model: request.model, usage: { inputTokens: null }, ms: 1 },
            }),
        );
        expect(await llm.embed({ model: 'vectors', input: 'x' })).toMatchObject({
            ok: true,
            data: { embeddings: [[1]], calls: 1, attempts: 2, usage: { inputTokens: null } },
        });
    });

    it('sends the next call while a retry sleeps in its backoff', async () => {
        const adapter = createEmbedProvider();
        const llm = createLlm({
            limiterOptions: { retryDelayMs: 200, maxRetryDelayMs: 200 },
            providers: {
                p: {
                    adapter,
                    models: {
                        vectors: {
                            model: 'vec-id',
                            modelLimits: { concurrency: 1 },
                            capabilities: { embeddings: { supported: true, maxInputs: 1 } },
                        },
                    },
                },
            },
        });
        adapter.reply(
            () => ({ ok: false, error: { code: 'timeout', message: 'slow', retryable: true } }),
            (request) => ({
                ok: true,
                data: { embeddings: [[request.input.length]], model: request.model, usage: { inputTokens: 1 }, ms: 1 },
            }),
        );
        expect(await llm.embed({ model: 'vectors', input: ['a', 'b'] })).toMatchObject({
            ok: true,
            data: { calls: 2, attempts: 3 },
        });
        expect(adapter.requests.map((request) => request.input[0]?.text)).toEqual(['a', 'b', 'a']);
    });

    it('fails with the first failing call, aborts the others, and maps a wrong vector count to unexpected', async () => {
        const adapter = createEmbedProvider();
        const serial = createLlm({
            limiterOptions: { maxRetries: 0 },
            providers: {
                p: {
                    adapter,
                    models: {
                        v: {
                            model: 'id',
                            modelLimits: { concurrency: 1 },
                            capabilities: { embeddings: { supported: true, maxInputs: 1 } },
                        },
                    },
                },
            },
        });
        adapter.reply(() => ({
            ok: false,
            error: { code: 'http', message: 'HTTP 400', retryable: false, status: 400 },
        }));
        expect(await serial.embed({ model: 'v', input: ['a', 'b', 'c'] })).toMatchObject({
            ok: false,
            error: { code: 'http', status: 400 },
        });
        // Queue may dispatch second call before abort takes effect; third call never starts
        expect(adapter.requests.length).toBeLessThan(3);
        adapter.requests.length = 0;
        const llm = setup(adapter);
        adapter.reply(() => ({
            ok: false,
            error: { code: 'http', message: 'HTTP 400', retryable: false, status: 400 },
        }));
        expect(await llm.embed({ model: 'vectors', input: ['a', 'b', 'c'] })).toMatchObject({
            ok: false,
            error: { code: 'http', message: 'HTTP 400', status: 400, name: 'vectors', provider: 'p' },
        });
        adapter.reply((request) => ({
            ok: true,
            data: { embeddings: [[1]], model: request.model, usage: { inputTokens: null }, ms: 1 },
        }));
        expect(await llm.embed({ model: 'vectors', input: ['a', 'b'] })).toMatchObject({
            ok: false,
            error: { code: 'unexpected', message: 'vec returned 1 vectors for 2 texts', attempts: 1 },
        });
    });

    it('ends with aborted when the caller aborts, and with unexpected when the adapter throws', async () => {
        const adapter = createEmbedProvider();
        const llm = setup(adapter);
        const controller = new AbortController();
        controller.abort();
        expect(await llm.embed({ model: 'vectors', input: 'x' }, { signal: controller.signal })).toMatchObject({
            ok: false,
            error: { code: 'aborted', name: 'vectors' },
        });
        adapter.reply(() => {
            throw new Error('boom');
        });
        expect(await llm.embed({ model: 'vectors', input: 'x' })).toMatchObject({
            ok: false,
            error: { code: 'unexpected', message: 'boom', retryable: false },
        });
    });
});
