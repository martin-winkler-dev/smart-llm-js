// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'bun:test';
import { createLlm, type LlmSettings } from '@smart-llm/core';
import type { GenaiCallOptions, GenaiClient, GenaiEmbedRequest, GenaiRequest } from 'smart-genai';
import { createGeminiProvider, GEMINI_SUPPORT } from '../src/index.js';

/** Creates fake smart-genai client recording calls and returning mock data. */
function fakeClient(): GenaiClient & {
    calls: { request: GenaiRequest; options?: GenaiCallOptions }[];
    embedCalls: GenaiEmbedRequest[];
} {
    const calls: { request: GenaiRequest; options?: GenaiCallOptions }[] = [];
    const embedCalls: GenaiEmbedRequest[] = [];
    return {
        mode: 'developer',
        calls,
        embedCalls,
        async embed(request) {
            embedCalls.push(request);
            const count = typeof request.input === 'string' ? 1 : request.input.length;
            const embeddings = Array.from({ length: count }, (_, index) => [index, 1]);
            return { ok: true, data: { embeddings, model: request.model, ms: 2 } };
        },
        async generate(request, options) {
            calls.push({ request, ...(options === undefined ? {} : { options }) });
            return {
                ok: true,
                data: {
                    text: '{"a":1}',
                    json: { a: 1 },
                    sources: [{ url: 'https://example.com' }],
                    model: request.model ?? 'gemini',
                    usage: { inputTokens: 1, outputTokens: 2, reasoningTokens: 3 },
                    finishReason: 'STOP',
                    ms: 4,
                },
            };
        },
        async listModels() {
            return {
                ok: true,
                data: [{ id: 'gemini-3.8-flash', displayName: 'Flash', actions: ['generateContent'], thinking: true }],
            };
        },
    };
}

const call = { signal: new AbortController().signal, extra: {} };

describe('createGeminiProvider', () => {
    it('passes the request and extra.config to smart-genai and leaves json to createLlm', async () => {
        const client = fakeClient();
        const gemini = createGeminiProvider(client);
        const config = { thinkingConfig: { includeThoughts: true } };
        const request = { model: 'gemini-3.8-flash', messages: [{ role: 'user' as const, content: 'hi' }] };
        const result = await gemini.generate(request, { ...call, extra: { config } });
        expect(result).toEqual({
            ok: true,
            data: {
                text: '{"a":1}',
                sources: [{ url: 'https://example.com' }],
                model: 'gemini-3.8-flash',
                usage: { inputTokens: 1, outputTokens: 2, reasoningTokens: 3 },
                finishReason: 'STOP',
                ms: 4,
            },
        });
        expect(client.calls[0]?.request).toEqual(request);
        expect(client.calls[0]?.options).toEqual({ signal: call.signal, config });
    });

    it('passes tools and tool calls with extra_content, leaves out assistant reasoning and returns toolCalls', async () => {
        const client = fakeClient();
        const toolCall = {
            id: 'c1',
            type: 'function' as const,
            function: { name: 'get_weather', arguments: '{}' },
            extra_content: { google: { thought_signature: 'sig' } },
        };
        client.generate = async (request) => {
            client.calls.push({ request });
            return {
                ok: true,
                data: {
                    text: '',
                    toolCalls: [toolCall],
                    model: 'gemini-3.8-flash',
                    usage: { inputTokens: 1, outputTokens: 2, reasoningTokens: 0 },
                    finishReason: 'STOP',
                    ms: 1,
                },
            };
        };
        const gemini = createGeminiProvider(client);
        const tools = [{ type: 'function' as const, function: { name: 'get_weather' } }];
        const result = await gemini.generate(
            {
                model: 'gemini-3.8-flash',
                messages: [
                    { role: 'user', content: 'weather?' },
                    { role: 'assistant', content: null, reasoning: 'think', tool_calls: [toolCall] },
                    { role: 'tool', tool_call_id: 'c1', content: 'sunny' },
                ],
                tools,
                tool_choice: 'required',
            },
            call,
        );
        expect(result).toMatchObject({ ok: true, data: { toolCalls: [toolCall] } });
        expect(client.calls[0]?.request).toEqual({
            model: 'gemini-3.8-flash',
            messages: [
                { role: 'user', content: 'weather?' },
                { role: 'assistant', content: null, tool_calls: [toolCall] },
                { role: 'tool', tool_call_id: 'c1', content: 'sunny' },
            ],
            tools,
            tool_choice: 'required',
        });
    });

    it('validateExtra accepts only config, checked by smart-genai', () => {
        const gemini = createGeminiProvider(fakeClient());
        expect(() => gemini.validateExtra?.({ config: { temperature: 0 } }, 'extra')).not.toThrow();
        expect(() => gemini.validateExtra?.({ x: 1 }, 'providers.g.extra')).toThrow(
            'Option "providers.g.extra.x" is not supported by the Gemini provider (only config)',
        );
        expect(() => gemini.validateExtra?.({ config: [] }, 'extra')).toThrow(/extra\.config/);
        expect(() => gemini.validateExtra?.({ config: null }, 'extra')).toThrow(TypeError);
        expect(() => gemini.validateExtra?.({ config: { httpOptions: {} } }, 'extra')).toThrow(/extra\.config/);
    });

    it('merges config per field across provider, model and call, thinkingConfig per member', async () => {
        const client = fakeClient();
        const safetySettings = [{ category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_NONE' }];
        const llm = createLlm({
            providers: {
                gemini: {
                    adapter: createGeminiProvider(client),
                    extra: { config: { safetySettings } },
                    models: {
                        flash: {
                            model: 'gemini-3.8-flash',
                            extra: { config: { temperature: 0.1, thinkingConfig: { thinkingBudget: 512 } } },
                        },
                    },
                },
            },
        } as never);
        const result = await llm.generate(
            { model: 'flash', messages: [{ role: 'user', content: 'hi' }] },
            { extra: { config: { thinkingConfig: { includeThoughts: false } } } },
        );
        expect(result.ok).toBe(true);
        expect(client.calls[0]?.options?.config).toEqual({
            safetySettings,
            temperature: 0.1,
            thinkingConfig: { thinkingBudget: 512, includeThoughts: false },
        } as never);
        const gemini = createGeminiProvider(fakeClient());
        expect(gemini.mergeExtra?.([{}, {}, {}])).toEqual({});
    });

    it('passes batch to smart-genai and its unsupported error through', async () => {
        const client = fakeClient();
        const gemini = createGeminiProvider(client);
        const request = { model: 'm', messages: [{ role: 'user' as const, content: 'hi' }], batch: true };
        expect(await gemini.generate(request, call)).toMatchObject({ ok: true });
        expect(client.calls[0]?.request.batch).toBe(true);
        const unsupported = { code: 'unsupported' as const, message: 'no batch API', retryable: false };
        client.generate = async () => ({ ok: false, error: unsupported });
        expect(await gemini.generate(request, call)).toEqual({ ok: false, error: unsupported });
    });

    it('rejects client options with defaults, which belong into createLlm', () => {
        expect(() => createGeminiProvider({ apiKey: 'k', defaults: { temperature: 0 } } as never)).toThrow(
            'Option "defaults" is not supported: request defaults go into createLlm defaults',
        );
    });

    it('returns an error instead of throwing when smart-genai throws', async () => {
        const client = fakeClient();
        client.generate = () => {
            throw new TypeError('bad config');
        };
        client.listModels = () => Promise.reject(new Error('boom'));
        const gemini = createGeminiProvider(client);
        expect(await gemini.generate({ model: 'm', messages: [{ role: 'user', content: 'hi' }] }, call)).toEqual({
            ok: false,
            error: { code: 'invalid_request', message: 'bad config', retryable: false },
        });
        expect(await gemini.listModels?.()).toEqual({
            ok: false,
            error: { code: 'unexpected', message: 'boom', retryable: false },
        });
    });

    it('lists models without the smart-genai only fields', async () => {
        const gemini = createGeminiProvider(fakeClient());
        expect(await gemini.listModels?.()).toEqual({
            ok: true,
            data: [{ id: 'gemini-3.8-flash', displayName: 'Flash' }],
        });
    });

    it('supports images, tools, web search, reasoning, batch calls, native JSON schemas and embeddings', () => {
        expect(GEMINI_SUPPORT).toEqual({
            images: true,
            tools: true,
            webSearch: true,
            reasoning: true,
            batch: true,
            json: 'native',
            embeddings: true,
        });
        expect(createGeminiProvider(fakeClient()).supports).toBe(GEMINI_SUPPORT);
    });
});

describe('Gemini embeddings', () => {
    it('maps tasks to task types, passes titles and dimensions, and reports no usage', async () => {
        const client = fakeClient();
        const gemini = createGeminiProvider(client);
        expect(gemini.maxEmbedInputs).toBe(100);
        const tasks = [
            ['query', 'RETRIEVAL_QUERY'],
            ['document', 'RETRIEVAL_DOCUMENT'],
            ['similarity', 'SEMANTIC_SIMILARITY'],
            ['classification', 'CLASSIFICATION'],
            ['clustering', 'CLUSTERING'],
        ] as const;
        for (const [task, taskType] of tasks) {
            const input = [{ text: 'a', title: 'T' }, { text: 'b' }];
            const result = await gemini.embed?.(
                { model: 'gemini-embedding-2', input, task, dimensions: 768 },
                { signal: call.signal },
            );
            expect(result).toEqual({
                ok: true,
                data: {
                    embeddings: [
                        [0, 1],
                        [1, 1],
                    ],
                    model: 'gemini-embedding-2',
                    usage: { inputTokens: null },
                    ms: 2,
                },
            });
            expect(client.embedCalls.at(-1)).toEqual({ model: 'gemini-embedding-2', input, taskType, dimensions: 768 });
        }
        await gemini.embed?.({ model: 'm', input: [{ text: 'x' }] }, { signal: call.signal });
        expect(client.embedCalls.at(-1)).toEqual({ model: 'm', input: [{ text: 'x' }] });
    });

    it('passes smart-genai errors without settings and turns a throw into an error', async () => {
        const client = fakeClient();
        client.embed = async () => ({
            ok: false,
            error: { code: 'rate_limited', message: 'slow down', retryable: true, retryAfterMs: 5, attempts: 1 },
        });
        const gemini = createGeminiProvider(client);
        const request = { model: 'm', input: [{ text: 'x' }] };
        expect(await gemini.embed?.(request, { signal: call.signal })).toEqual({
            ok: false,
            error: { code: 'rate_limited', message: 'slow down', retryable: true, retryAfterMs: 5 },
        });
        client.embed = () => {
            throw new Error('boom');
        };
        expect(await gemini.embed?.(request, { signal: call.signal })).toEqual({
            ok: false,
            error: { code: 'unexpected', message: 'boom', retryable: false },
        });
    });

    it('embeds through createLlm in calls of at most 100 texts', async () => {
        const client = fakeClient();
        const llm = createLlm({
            providers: {
                gemini: {
                    adapter: createGeminiProvider(client),
                    models: { vectors: { model: 'gemini-embedding-001', capabilities: { embeddings: true } } },
                },
            },
        });
        const input = Array.from({ length: 150 }, (_, index) => `text ${index}`);
        const result = await llm.embed({ model: 'vectors', input, task: 'document' });
        expect(result).toMatchObject({ ok: true, data: { calls: 2, attempts: 2, usage: { inputTokens: null } } });
        expect(result.ok && result.data.embeddings).toHaveLength(150);
        expect(client.embedCalls.map((request) => request.input.length)).toEqual([100, 50]);
        expect(() =>
            createLlm({
                providers: {
                    gemini: {
                        adapter: createGeminiProvider(client),
                        models: {
                            v: { model: 'm', capabilities: { embeddings: { supported: true, maxInputs: 101 } } },
                        },
                    },
                },
            }),
        ).toThrow(
            'Invalid value for option "providers.gemini.models.v.capabilities.embeddings.maxInputs": an integer from 1 to 100',
        );
    });
});

describe('Gemini answers of failed calls, settings and attempts', () => {
    const usage = { inputTokens: 5, outputTokens: 1, reasoningTokens: 2 };
    const request = { model: 'gemini-3.8-flash', messages: [{ role: 'user' as const, content: 'hi' }] };

    it('keeps the output of a blocked or truncated smart-genai error and leaves out its json, settings and attempts', async () => {
        const client = fakeClient();
        const gemini = createGeminiProvider(client);
        const genaiSettings = { model: 'gemini-3.8-flash', json: true as const };
        for (const [code, finishReason] of [
            ['blocked', 'SAFETY'],
            ['truncated', 'MAX_TOKENS'],
        ] as const) {
            const answer = {
                text: 'par',
                reasoning: 'thought',
                model: 'gemini-3.8-flash-001',
                usage,
                finishReason,
                ms: 7,
            };
            const error = {
                code,
                message: 'no answer',
                retryable: false,
                text: 'par',
                model: answer.model,
                usage,
                ms: 7,
            };
            client.generate = async () => ({
                ok: false,
                error: {
                    ...error,
                    output: { ...answer, json: null, settings: genaiSettings, attempts: 1 },
                    settings: genaiSettings,
                    attempts: 1,
                },
            });
            expect(await gemini.generate(request, call)).toEqual({ ok: false, error: { ...error, output: answer } });
        }
    });

    it('puts no output on a smart-genai error without an answer', async () => {
        const client = fakeClient();
        const error = { code: 'network' as const, message: 'fetch failed', retryable: true };
        client.generate = async () => ({ ok: false, error: { ...error, settings: { model: 'm' }, attempts: 1 } });
        expect(await createGeminiProvider(client).generate(request, call)).toEqual({ ok: false, error });
    });

    /** Creates test LLM instance wrapping Gemini adapter. */
    function geminiLlm(client: GenaiClient) {
        return createLlm({
            defaults: { temperature: 0 },
            providers: {
                gemini: {
                    adapter: createGeminiProvider(client),
                    extra: { config: { safetySettings: [] } },
                    models: {
                        flash: {
                            model: 'gemini-3.8-flash',
                            extra: { config: { thinkingConfig: { includeThoughts: true } } },
                            capabilities: { json: true },
                        },
                    },
                },
            },
        });
    }

    const format = {
        type: 'json_schema' as const,
        json_schema: { name: 'a', schema: { type: 'object', properties: { a: { type: 'number' } }, required: ['a'] } },
    };

    const settings: LlmSettings = {
        name: 'flash',
        provider: 'gemini',
        model: 'gemini-3.8-flash',
        temperature: 0,
        jsonMode: 'native',
        extra: { config: { safetySettings: [], thinkingConfig: { includeThoughts: true, thinkingBudget: 128 } } },
    };

    it('reports the settings with native JSON and the merged extra, and one attempt', async () => {
        const result = await geminiLlm(fakeClient()).flash('hi', {
            response_format: format,
            extra: { config: { thinkingConfig: { thinkingBudget: 128 } } },
        });
        expect(result).toMatchObject({ ok: true, data: { json: { a: 1 }, name: 'flash', provider: 'gemini' } });
        expect(result.ok ? result.data.settings : undefined).toEqual(settings);
        expect(result.ok ? result.data.attempts : undefined).toBe(1);
    });

    it('puts the answer with the client reasoning and parsed json on a schema mismatch, without a retry', async () => {
        const client = fakeClient();
        client.generate = async (request) => {
            client.calls.push({ request });
            return {
                ok: true,
                data: {
                    text: '{"a":"x"}',
                    reasoning: 'thought',
                    model: 'gemini-3.8-flash',
                    usage,
                    finishReason: 'STOP',
                    ms: 3,
                },
            };
        };
        const result = await geminiLlm(client).flash('hi', {
            response_format: format,
            extra: { config: { thinkingConfig: { thinkingBudget: 128 } } },
        });
        expect(result).toEqual({
            ok: false,
            error: {
                code: 'invalid_json',
                message: expect.stringContaining('does not match the schema'),
                retryable: false,
                text: '{"a":"x"}',
                model: 'gemini-3.8-flash',
                usage,
                ms: 3,
                name: 'flash',
                provider: 'gemini',
                output: {
                    text: '{"a":"x"}',
                    json: { a: 'x' },
                    reasoning: 'thought',
                    model: 'gemini-3.8-flash',
                    usage,
                    finishReason: 'STOP',
                    ms: 3,
                    name: 'flash',
                    provider: 'gemini',
                },
                settings,
                attempts: 1,
            },
        });
        expect(client.calls).toHaveLength(1);
    });

    it('carries the output of a blocked smart-genai error through createLlm', async () => {
        const client = fakeClient();
        client.generate = async () => ({
            ok: false,
            error: {
                code: 'blocked',
                message: 'SAFETY',
                retryable: false,
                text: '',
                model: 'gemini-3.8-flash',
                usage,
                ms: 2,
                output: {
                    text: '',
                    reasoning: 'thought',
                    model: 'gemini-3.8-flash',
                    usage,
                    finishReason: 'SAFETY',
                    ms: 2,
                },
            },
        });
        const result = await geminiLlm(client).flash('hi');
        const { jsonMode: _jsonMode, ...withoutJson } = settings;
        expect(result).toMatchObject({
            ok: false,
            error: {
                code: 'blocked',
                output: {
                    text: '',
                    reasoning: 'thought',
                    model: 'gemini-3.8-flash',
                    usage,
                    finishReason: 'SAFETY',
                    ms: 2,
                    name: 'flash',
                    provider: 'gemini',
                },
                settings: {
                    ...withoutJson,
                    extra: { config: { safetySettings: [], thinkingConfig: { includeThoughts: true } } },
                },
                attempts: 1,
            },
        });
    });
});
