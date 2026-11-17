// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'bun:test';
import { createRPM } from 'smart-limit';
import { imagePart } from '../src/images.js';
import { estimateTokens } from '../src/limits.js';
import { createLlm } from '../src/llm.js';
import { assistantMessage, toolMessage } from '../src/messages.js';
import type { LlmMessage, LlmModelCall, LlmModelInput, LlmRequest, LlmTool, LlmToolCall } from '../src/types.js';
import { answer, createFakeProvider, type FakeReply, failure } from './fake-provider.js';

const messages: LlmMessage[] = [{ role: 'user', content: 'hi' }];

const schema = {
    type: 'object',
    properties: { kind: { type: 'string', enum: ['noun', 'verb'] }, note: { type: 'string' } },
    required: ['kind'],
};

describe('createLlm', () => {
    it('routes requests by model name across providers', async () => {
        const gemini = createFakeProvider('gemini');
        const local = createFakeProvider('local');
        const llm = createLlm({
            providers: {
                gemini: { adapter: gemini, models: { flash: 'gemini-3.8-flash' } },
                local: { adapter: local, models: { qwen: { model: 'qwen2.5:7b' } } },
            },
        });
        const first = await llm.generate({ model: 'flash', messages });
        const second = await llm.generate({ model: 'qwen', messages });
        expect(first).toMatchObject({
            ok: true,
            data: { name: 'flash', provider: 'gemini', model: 'gemini-3.8-flash' },
        });
        expect(second).toMatchObject({ ok: true, data: { name: 'qwen', provider: 'local', model: 'qwen2.5:7b' } });
        expect(gemini.calls[0]?.request.model).toBe('gemini-3.8-flash');
        expect(local.calls[0]?.request.model).toBe('qwen2.5:7b');
        expect(llm.models.map((model) => model.name)).toEqual(['flash', 'qwen']);
        expect(await llm.listModels('local')).toEqual({ ok: true, data: [{ id: 'local-model' }] });
    });

    it('merges defaults and extra in order: shared, provider, model, request', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            defaults: { temperature: 0, top_p: 0.5, seed: 1 },
            providers: {
                p: {
                    adapter,
                    defaults: { top_p: 0.9, max_tokens: 100 },
                    extra: { a: 1, b: 1 },
                    models: { m: { model: 'id', defaults: { max_completion_tokens: 50 }, extra: { b: 2 } } },
                },
            },
        });
        await llm.generate({ model: 'm', messages, seed: 7 }, { extra: { c: 3 } });
        const call = adapter.calls[0];
        expect(call?.request).toEqual({ model: 'id', messages, temperature: 0, top_p: 0.9, max_tokens: 50, seed: 7 });
        expect(call?.extra).toEqual({ a: 1, b: 2, c: 3 });
    });

    it('returns unsupported for undeclared model capabilities even if adapter supports them', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({ providers: { p: { adapter, models: { small: 'small' } } } });
        const image = imagePart(new Uint8Array([1, 2, 3]), 'image/png');
        const tool: LlmTool = { type: 'function', function: { name: 'f' } };
        const cases: LlmRequest<'small'>[] = [
            { model: 'small', messages: [{ role: 'user', content: [image] }] },
            { model: 'small', messages, web_search: true },
            { model: 'small', messages, reasoning_effort: 'low' },
            { model: 'small', messages, batch: true },
            { model: 'small', messages, tools: [tool] },
            { model: 'small', messages, response_format: { type: 'json_object' } },
        ];
        for (const request of cases) {
            expect(await llm.generate(request)).toMatchObject({
                ok: false,
                error: { code: 'unsupported', retryable: false },
            });
        }
        expect(adapter.calls).toHaveLength(0);
        expect(
            await llm.generate({
                model: 'small',
                messages,
                reasoning_effort: 'none',
                web_search: false,
                batch: false,
                tool_choice: 'none',
            }),
        ).toMatchObject({ ok: true });
        expect(adapter.calls[0]?.request).toEqual({ model: 'small', messages });
        expect(llm.models[0]?.capabilities).toEqual({
            images: false,
            tools: false,
            json: false,
            webSearch: { supported: false, default: false },
            batch: { supported: false, default: false },
            reasoning: { supported: false },
            embeddings: { supported: false, maxInputs: 64 },
        });
    });

    it('applies capability defaults when omitted and forwards explicit off', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: {
                p: {
                    adapter,
                    models: {
                        m: {
                            model: 'id',
                            capabilities: {
                                images: true,
                                json: true,
                                webSearch: { supported: true, default: true },
                                reasoning: { supported: true, default: 'medium' },
                            },
                        },
                        n: { model: 'id', capabilities: { reasoning: { supported: true } } },
                    },
                },
            },
        });
        await llm.generate({ model: 'm', messages });
        await llm.generate({ model: 'm', messages, web_search: false, reasoning_effort: 'none' });
        await llm.generate({ model: 'n', messages });
        await llm.generate({ model: 'n', messages, reasoning_effort: 'max' });
        expect(adapter.calls.map((call) => call.request)).toEqual([
            { model: 'id', messages, web_search: true, reasoning_effort: 'medium' },
            { model: 'id', messages, reasoning_effort: 'none' },
            { model: 'id', messages },
            { model: 'id', messages, reasoning_effort: 'max' },
        ]);
        expect(llm.models[0]?.capabilities).toEqual({
            images: true,
            tools: false,
            json: 'native',
            webSearch: { supported: true, default: true },
            batch: { supported: false, default: false },
            reasoning: { supported: true, default: 'medium' },
            embeddings: { supported: false, maxInputs: 64 },
        });
    });

    it('maps a reasoning effort to the nearest declared level', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: {
                p: {
                    adapter,
                    models: {
                        oss: {
                            model: 'oss',
                            capabilities: { reasoning: { supported: true, levels: ['low', 'medium', 'high'] } },
                        },
                        onOff: {
                            model: 'onOff',
                            capabilities: { reasoning: { supported: true, levels: ['none', 'high'] } },
                        },
                        even: {
                            model: 'even',
                            capabilities: { reasoning: { supported: true, levels: ['low', 'high'] } },
                        },
                    },
                },
            },
        });
        const efforts = ['none', 'minimal', 'medium', 'xhigh', 'max'] as const;
        for (const effort of efforts) {
            await llm.generate({ model: 'oss', messages, reasoning_effort: effort });
        }
        await llm.generate({ model: 'onOff', messages, reasoning_effort: 'none' });
        await llm.generate({ model: 'onOff', messages, reasoning_effort: 'minimal' });
        await llm.generate({ model: 'even', messages, reasoning_effort: 'medium' });
        expect(adapter.calls.map((call) => call.request.reasoning_effort)).toEqual([
            'low',
            'low',
            'medium',
            'high',
            'high',
            'none',
            'high',
            'high',
        ]);
    });

    it('throws TypeError for unsupported adapter capabilities or invalid defaults', () => {
        const plain = createFakeProvider('plain', {});
        const full = createFakeProvider();
        const invalid: [unknown, unknown, RegExp][] = [
            [plain, { images: true }, /"providers\.p\.models\.m\.capabilities\.images": the plain adapter does not/],
            [plain, { tools: true }, /capabilities\.tools/],
            [plain, { json: 'prompt' }, /capabilities\.json/],
            [plain, { webSearch: { supported: true } }, /capabilities\.webSearch/],
            [plain, { batch: { supported: true } }, /capabilities\.batch/],
            [plain, { reasoning: { supported: true } }, /capabilities\.reasoning/],
            [full, { images: 'yes' }, /capabilities\.images/],
            [full, { json: 'xml' }, /capabilities\.json/],
            [full, { batch: true }, /capabilities\.batch/],
            [full, { batch: { default: true } }, /capabilities\.batch\.supported/],
            [full, { webSearch: { supported: false, default: true } }, /webSearch\.default": true needs supported/],
            [full, { reasoning: { supported: false, default: 'low' } }, /default and levels need supported/],
            [full, { reasoning: { supported: true, default: 'max', levels: ['low'] } }, /reasoning\.default": one of/],
            [full, { reasoning: { supported: true, default: 'huge' } }, /reasoning\.default/],
            [full, { reasoning: { supported: true, levels: ['none'] } }, /reasoning\.levels/],
            [full, { reasoning: { supported: true, levels: ['low', 'low'] } }, /reasoning\.levels/],
            [full, { jsonSchema: 'native' }, /capabilities\.jsonSchema" is not supported/],
        ];
        for (const [adapter, capabilities, message] of invalid) {
            expect(() =>
                createLlm({ providers: { p: { adapter, models: { m: { model: 'm', capabilities } } } } } as never),
            ).toThrow(message);
        }
        for (const supports of [{ images: false }, { json: 'xml' }, { voice: true }]) {
            expect(() =>
                createLlm({ providers: { p: { adapter: { ...full, supports } as never, models: { m: 'm' } } } }),
            ).toThrow(/providers\.p\.adapter\.supports/);
        }
        expect(full.calls).toHaveLength(0);
    });

    it('throws TypeError for options disallowed in defaults', () => {
        const adapter = createFakeProvider();
        const moved: [Record<string, unknown>, RegExp][] = [
            [
                { web_search: true },
                /"defaults\.web_search" is not supported: set it in the model's capabilities\.webSearch/,
            ],
            [{ batch: true }, /capabilities\.batch\.default, or batch per call/],
            [{ reasoning_effort: 'low' }, /capabilities\.reasoning\.default/],
            [
                { response_format: { type: 'json_object' } },
                /"defaults\.response_format" is not supported: set it in each/,
            ],
            [{ tools: [] }, /"defaults\.tools" is not supported/],
            [{ tool_choice: 'auto' }, /"defaults\.tool_choice" is not supported/],
        ];
        for (const [defaults, message] of moved) {
            expect(() => createLlm({ defaults, providers: { p: { adapter, models: { m: 'm' } } } } as never)).toThrow(
                message,
            );
        }
        expect(() =>
            createLlm({
                providers: { p: { adapter, models: { m: { model: 'm', defaults: { batch: true } as never } } } },
            }),
        ).toThrow(/providers\.p\.models\.m\.defaults\.batch/);
    });
});

describe('model functions', () => {
    it('sends input and options via llm.<name>(input, options)', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: {
                p: {
                    adapter,
                    defaults: { temperature: 0.1 },
                    models: { flash: 'gemini-flash', 'ds-flash': { model: 'deepseek', extra: { a: 1 } } },
                },
            },
        });
        const short = await llm.flash('hi');
        const full = await llm['ds-flash']('hi', { temperature: 0, seed: 3, extra: { b: 2 } });
        expect(short).toMatchObject({ ok: true, data: { name: 'flash', provider: 'p', model: 'gemini-flash' } });
        expect(full).toMatchObject({ ok: true, data: { name: 'ds-flash', provider: 'p', model: 'deepseek' } });
        expect(adapter.calls[0]?.request).toEqual({ model: 'gemini-flash', messages, temperature: 0.1 });
        expect(adapter.calls[1]?.request).toEqual({ model: 'deepseek', messages, temperature: 0, seed: 3 });
        expect(adapter.calls[1]?.extra).toEqual({ a: 1, b: 2 });
        expect(await llm.generate({ model: 'flash', messages })).toEqual(short);
        // @ts-expect-error only configured names are members
        expect(llm.pro).toBeUndefined();
    });

    it('takes a string, content parts or messages as input', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { images: true } } } } },
        });
        const image = imagePart(new Uint8Array([1, 2, 3]), 'image/png');
        const parts = [{ type: 'text' as const, text: 'what is this?' }, image];
        const conversation: LlmMessage[] = [
            { role: 'system', content: 'be short' },
            { role: 'user', content: parts },
            { role: 'assistant', content: 'a dot' },
            { role: 'user', content: 'sure?' },
        ];
        expect(await llm.m(parts)).toMatchObject({ ok: true });
        expect(await llm.m(conversation, { max_tokens: 10 })).toMatchObject({ ok: true });
        expect(adapter.calls[0]?.request.messages).toEqual([{ role: 'user', content: parts }]);
        expect(adapter.calls[1]?.request).toEqual({ model: 'm', messages: conversation, max_tokens: 10 });
    });

    it('works as a detached function', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({ providers: { p: { adapter, models: { a: 'model-a', b: 'model-b' } } } });
        function pick(call: LlmModelCall) {
            return call('hi', { seed: 1 });
        }
        const { a } = llm;
        function fixed(input: LlmModelInput) {
            return llm.b(input, { temperature: 0 });
        }
        expect(await a('hi')).toMatchObject({ ok: true, data: { name: 'a' } });
        expect(await pick(llm.b)).toMatchObject({ ok: true, data: { name: 'b' } });
        expect(await fixed('ho')).toMatchObject({ ok: true, data: { name: 'b' } });
        expect(adapter.calls.map((call) => call.request)).toEqual([
            { model: 'model-a', messages },
            { model: 'model-b', messages, seed: 1 },
            { model: 'model-b', messages: [{ role: 'user', content: 'ho' }], temperature: 0 },
        ]);
        expect(Object.keys(llm)).toEqual(['models', 'generate', 'embed', 'listModels', 'a', 'b']);
    });

    it('returns invalid_request with model and provider names instead of throwing', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({ providers: { p: { adapter, models: { m: 'm' } } } });
        const cases: [Promise<unknown>, RegExp][] = [
            [llm.m('hi', { model: 'other' } as never), /Option "model" is not supported/],
            [llm.m('hi', { messages } as never), /Option "messages" is not supported/],
            [llm.m(null as never), /a string, content parts or messages/],
            [llm.m(42 as never), /a string, content parts or messages/],
            [llm.m({ messages } as never), /a string, content parts or messages/],
            [llm.m([]), /a string, content parts or messages/],
            [llm.m([...messages, { type: 'text', text: 'x' }] as never), /a string, content parts or messages/],
            [llm.m('hi', null as never), /an object/],
            [llm.m('hi', { n: 2 } as never), /Option "n" is not supported/],
            [llm.m('hi', { timeoutMs: 1 } as never), /Option "timeoutMs"/],
            [llm.m('hi', { signal: 'x' } as never), /option "signal"/],
        ];
        for (const [result, message] of cases) {
            expect(await result).toMatchObject({
                ok: false,
                error: {
                    code: 'invalid_request',
                    message: expect.stringMatching(message),
                    retryable: false,
                    name: 'm',
                    provider: 'p',
                },
            });
        }
        expect(adapter.calls).toHaveLength(0);
    });

    it('preserves defaults when option is undefined or options object is empty', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            defaults: { temperature: 0.3, max_tokens: 100 },
            providers: { p: { adapter, models: { m: 'm' } } },
        });
        await llm.m('hi', { temperature: undefined, max_tokens: undefined, seed: 1 });
        await llm.m('hi', {});
        await llm.generate({ model: 'm', messages, temperature: undefined } as never);
        expect(adapter.calls.map((call) => call.request)).toEqual([
            { model: 'm', messages, temperature: 0.3, max_tokens: 100, seed: 1 },
            { model: 'm', messages, temperature: 0.3, max_tokens: 100 },
            { model: 'm', messages, temperature: 0.3, max_tokens: 100 },
        ]);
    });

    it('parses json when response_format is passed in options', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { json: true } } } } },
        });
        const format = { type: 'json_schema' as const, json_schema: { name: 'word', schema } };
        function classify(input: LlmModelInput) {
            return llm.m(input, { response_format: format });
        }
        adapter.reply(answer('{"kind":"noun"}'), answer('{"kind":"adjective"}'));
        expect(await classify('house')).toMatchObject({ ok: true, data: { json: { kind: 'noun' } } });
        expect(await classify('red')).toMatchObject({ ok: false, error: { code: 'invalid_json' } });
        expect(adapter.calls[0]?.request.response_format).toEqual(format);
    });

    it('supports tool loop with assistantMessage and toolMessage', async () => {
        const adapter = createFakeProvider();
        const call: LlmToolCall = { id: 'c1', type: 'function', function: { name: 'f', arguments: '{"x":1}' } };
        adapter.reply(
            (sent) => ({
                ok: true,
                data: {
                    text: '',
                    reasoning: 'need f',
                    toolCalls: [call],
                    model: sent.request.model,
                    usage: { inputTokens: 1, outputTokens: 1, reasoningTokens: 1 },
                    finishReason: 'tool_calls',
                    ms: 1,
                },
            }),
            answer('done'),
        );
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { tools: true } } } } },
        });
        const tools: LlmTool[] = [{ type: 'function', function: { name: 'f' } }];
        const conversation: LlmMessage[] = [{ role: 'user', content: 'go' }];
        const first = await llm.m(conversation, { tools });
        if (!first.ok) {
            throw new Error('first call failed');
        }
        conversation.push(
            assistantMessage(first.data),
            ...(first.data.toolCalls ?? []).map((toolCall) => toolMessage(toolCall, { y: 2 })),
        );
        expect(await llm.m(conversation, { tools })).toMatchObject({ ok: true, data: { text: 'done' } });
        expect(adapter.calls[1]?.request).toEqual({
            model: 'm',
            tools,
            messages: [
                { role: 'user', content: 'go' },
                { role: 'assistant', content: null, tool_calls: [call], reasoning: 'need f' },
                { role: 'tool', tool_call_id: 'c1', content: '{"y":2}' },
            ],
        });
    });

    it('rejects reserved model names and defines custom names as own properties', async () => {
        const adapter = createFakeProvider();
        const reserved = [
            'generate',
            'embed',
            'models',
            'listModels',
            'then',
            'toJSON',
            'toString',
            'valueOf',
            'constructor',
        ];
        for (const name of [...reserved, 'hasOwnProperty', '__proto__']) {
            const models = JSON.parse(JSON.stringify({ [name]: 'x' }));
            expect(() => createLlm({ providers: { p: { adapter, models } } })).toThrow(
                `Invalid value for option "providers.p.models.${name}": ${name} is reserved`,
            );
        }
        const models = JSON.parse('{"ds-flash": "dash-model", "flash.v2": "dot-model"}');
        const llm = createLlm({ providers: { p: { adapter, models } } }) as unknown as Record<string, LlmModelCall>;
        expect(Object.getPrototypeOf(llm)).toBe(Object.prototype);
        expect(await llm['ds-flash']?.('hi')).toMatchObject({ ok: true, data: { model: 'dash-model' } });
        expect(await llm['flash.v2']?.('hi')).toMatchObject({ ok: true, data: { model: 'dot-model' } });
        expect(JSON.parse(JSON.stringify({ llm })).llm.models).toHaveLength(2);
    });

    it('rejects empty provider and model names at creation', () => {
        const adapter = createFakeProvider();
        expect(() => createLlm({ providers: { '': { adapter, models: { m: 'm' } } } })).toThrow(
            new TypeError('Invalid value for option "providers": an empty provider name'),
        );
        expect(() => createLlm({ providers: { p: { adapter, models: { '': 'm' } } } })).toThrow(
            new TypeError('Invalid value for option "providers.p.models": an empty model name'),
        );
    });
});

describe('JSON answers', () => {
    const format = { type: 'json_schema' as const, json_schema: { name: 'word', schema } };

    it('native: sends the schema as it is and checks the answer', async () => {
        const adapter = createFakeProvider('p', { json: 'native' });
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { json: true } } } } },
        });
        adapter.reply(answer('{"kind":"noun"}'));
        expect(await llm.generate({ model: 'm', messages, response_format: format })).toMatchObject({
            ok: true,
            data: { json: { kind: 'noun' } },
        });
        expect(adapter.calls[0]?.request.response_format).toEqual(format);
        expect(adapter.calls[0]?.request.messages).toEqual(messages);
        adapter.reply(answer('{"kind":"adjective"}'));
        expect(await llm.generate({ model: 'm', messages, response_format: format })).toMatchObject({
            ok: false,
            error: { code: 'invalid_json', retryable: false, text: '{"kind":"adjective"}' },
        });
    });

    it('strict: sends a strict schema and removes the nulls of optional fields', async () => {
        const adapter = createFakeProvider('p', { json: 'strict' });
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { json: true } } } } },
        });
        adapter.reply(answer('{"kind":"verb","note":null}'));
        expect(await llm.generate({ model: 'm', messages, response_format: format })).toMatchObject({
            ok: true,
            data: { json: { kind: 'verb' } },
        });
        const sent = adapter.calls[0]?.request.response_format;
        expect(sent).toMatchObject({ type: 'json_schema', json_schema: { name: 'word', strict: true } });
        expect(sent?.type === 'json_schema' && sent.json_schema.schema).toMatchObject({
            required: ['kind', 'note'],
            additionalProperties: false,
        });
    });

    it('grammar: sends the schema and also puts it into the system prompt', async () => {
        const adapter = createFakeProvider('p', { json: 'grammar' });
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { json: true } } } } },
        });
        adapter.reply(answer('{"kind":"noun"}'));
        const system: LlmMessage = { role: 'system', content: 'Classify.' };
        await llm.generate({ model: 'm', messages: [system, ...messages], response_format: format });
        const sent = adapter.calls[0]?.request;
        expect(sent?.response_format).toEqual(format);
        expect(sent?.messages[0]).toEqual(system);
        expect(sent?.messages[1]).toMatchObject({ role: 'system' });
        expect(String(sent?.messages[1]?.content)).toContain('"enum": [');
        expect(sent?.messages[2]).toEqual(messages[0] as LlmMessage);
    });

    it('prompt: injects schema, sends json_object, and strips code fences', async () => {
        const adapter = createFakeProvider('p', { json: 'strict' });
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { json: 'prompt' } } } } },
        });
        adapter.reply(answer('```json\n{"kind":"noun"}\n```'));
        expect(await llm.generate({ model: 'm', messages, response_format: format })).toMatchObject({
            ok: true,
            data: { json: { kind: 'noun' } },
        });
        expect(adapter.calls).toHaveLength(1);
        expect(adapter.calls[0]?.request.response_format).toEqual({ type: 'json_object' });
        expect(String(adapter.calls[0]?.request.messages[0]?.content)).toContain('JSON Schema');
    });

    it('prompt: fails on schema mismatch without retrying in normal and batch calls', async () => {
        for (const batch of [false, true]) {
            const adapter = createFakeProvider();
            const llm = createLlm({
                limiterOptions: { maxRetries: 2, retryDelayMs: 1 },
                providers: {
                    p: {
                        adapter,
                        models: { m: { model: 'm', capabilities: { json: 'prompt', batch: { supported: true } } } },
                    },
                },
            });
            adapter.reply(answer('{"kind":"x"}'), answer('{"kind":"noun"}'));
            expect(await llm.generate({ model: 'm', messages, response_format: format, batch })).toMatchObject({
                ok: false,
                error: { code: 'invalid_json', retryable: false, attempts: 1 },
            });
            expect(adapter.calls).toHaveLength(1);
        }
    });

    it('json_object: rejects non-object JSON with invalid_json', async () => {
        const adapter = createFakeProvider('p', { json: 'prompt' });
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { json: true } } } } },
        });
        for (const text of ['123', 'null', '[1,2]', '"hi"', 'true']) {
            adapter.reply(answer(text));
            expect(
                await llm.generate({ model: 'm', messages, response_format: { type: 'json_object' } }),
            ).toMatchObject({
                ok: false,
                error: {
                    code: 'invalid_json',
                    message: 'The answer is not a JSON object',
                    retryable: false,
                    output: { json: JSON.parse(text) },
                },
            });
        }
        adapter.reply(answer('{"a":1}'));
        expect(await llm.generate({ model: 'm', messages, response_format: { type: 'json_object' } })).toMatchObject({
            ok: true,
            data: { json: { a: 1 } },
        });
    });

    it('json_schema without schema: rejects non-object JSON with invalid_json', async () => {
        const adapter = createFakeProvider('p', { json: 'native' });
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { json: true } } } } },
        });
        adapter.reply(answer('[1,2]'));
        const free = { type: 'json_schema' as const, json_schema: { name: 'free' } };
        expect(await llm.generate({ model: 'm', messages, response_format: free })).toMatchObject({
            ok: false,
            error: { code: 'invalid_json', message: 'The answer is not a JSON object', output: { json: [1, 2] } },
        });
        expect(adapter.calls[0]?.request.response_format).toEqual({ type: 'json_object' });
    });

    it('text: answers plain text and does not forward response_format', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({ providers: { p: { adapter, models: { m: 'm' } } } });
        adapter.reply(answer('[1]'));
        expect(await llm.generate({ model: 'm', messages, response_format: { type: 'text' } })).toMatchObject({
            ok: true,
            data: { text: '[1]' },
        });
        expect(adapter.calls[0]?.request).toEqual({ model: 'm', messages });
    });

    it('fails with invalid_request for a malformed response_format before any call', async () => {
        const adapter = createFakeProvider('p', { json: 'native', tools: true });
        const llm = createLlm({
            providers: {
                p: { adapter, models: { m: { model: 'm', capabilities: { json: true, tools: true } } } },
            },
        });
        /** Creates json_schema format with name `w` and given fields. */
        function spec(fields: Record<string, unknown>) {
            return { type: 'json_schema', json_schema: { name: 'w', ...fields } };
        }
        const invalid: [unknown, string][] = [
            [{ type: 'json_object', schema }, 'Option "response_format.schema" is not supported'],
            [{ type: 'text', strict: true }, 'Option "response_format.strict" is not supported'],
            [{ type: 'xml' }, 'Invalid value for option "response_format"'],
            [{ type: 'json_schema' }, 'Invalid value for option "response_format.json_schema"'],
            [spec({ foo: 1 }), 'Option "response_format.json_schema.foo" is not supported'],
            [spec({ name: '' }), 'Invalid value for option "response_format.json_schema.name"'],
            [spec({ name: 'a b' }), 'Invalid value for option "response_format.json_schema.name"'],
            [spec({ description: 1 }), 'Invalid value for option "response_format.json_schema.description"'],
            [spec({ strict: 'yes' }), 'Invalid value for option "response_format.json_schema.strict"'],
            [spec({ schema: [] }), 'Invalid value for option "response_format.json_schema.schema": a JSON Schema'],
            [
                spec({ schema: { type: 'object', properties: null } }),
                'Invalid value for option "response_format.json_schema.schema.properties"',
            ],
            [
                spec({ schema: { type: 'object', properties: { a: null } } }),
                'Invalid value for option "response_format.json_schema.schema.properties.a": a JSON Schema object',
            ],
            [
                spec({ schema: { type: 'array', items: { properties: { a: { items: 1 } } } } }),
                'Invalid value for option "response_format.json_schema.schema.items.properties.a.items"',
            ],
        ];
        for (const [format, message] of invalid) {
            expect(await llm.generate({ model: 'm', messages, response_format: format as never })).toMatchObject({
                ok: false,
                error: { code: 'invalid_request', message: expect.stringContaining(message), retryable: false },
            });
        }
        const parameters = { type: 'object', properties: { city: 'string' } };
        expect(
            await llm.generate({
                model: 'm',
                messages,
                tools: [{ type: 'function', function: { name: 'f', parameters } }],
            }),
        ).toMatchObject({
            ok: false,
            error: {
                code: 'invalid_request',
                message:
                    'Invalid value for option "tools.function.parameters.properties.city": a JSON Schema object or boolean',
            },
        });
        expect(adapter.calls).toHaveLength(0);
        const valid = spec({ schema, description: 'A word', strict: null });
        expect(await llm.generate({ model: 'm', messages, response_format: valid as never })).toMatchObject({
            ok: false,
            error: { code: 'invalid_json' },
        });
        expect(adapter.calls).toHaveLength(1);
    });

    it('accepts boolean/array sub-schemas and rejects invalid or circular schemas', async () => {
        const adapter = createFakeProvider('p', { json: 'native' });
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { json: true } } } } },
        });
        adapter.reply(answer('{}'));
        /** Generates with json_schema format of given schema. */
        function run(value: unknown) {
            const format = { type: 'json_schema', json_schema: { name: 'w', schema: value } };
            return llm.generate({ model: 'm', messages, response_format: format as never });
        }
        const object = { type: 'object', properties: { a: { type: 'string' } } };
        const valid: unknown[] = [
            { $defs: { node: object }, $ref: '#/$defs/node' },
            { anyOf: [object, { type: 'null' }] },
            { type: 'object', additionalProperties: object },
            { type: 'array', prefixItems: [object], items: object },
            { type: 'array', items: [object, { type: 'number' }] },
            { type: 'array', items: false },
            { type: 'object', properties: { p: true } },
            { type: 'object', properties: { a: object, b: object } },
        ];
        for (const value of valid) {
            // the fixed answer `{}` may not match the schema; only request validation is checked here
            const result = await run(value);
            expect(['ok', 'invalid_json']).toContain(result.ok ? 'ok' : result.error.code);
        }
        const cyclic: Record<string, unknown> = { type: 'object' };
        cyclic.properties = { self: cyclic };
        const path = 'Invalid value for option "response_format.json_schema.schema';
        const invalid: [unknown, string][] = [
            [{ properties: { p: null } }, `${path}.properties.p": a JSON Schema object or boolean`],
            [{ properties: { p: 1 } }, `${path}.properties.p": a JSON Schema object or boolean`],
            [{ items: 'string' }, `${path}.items": a JSON Schema object or boolean`],
            [{ items: [object, null] }, `${path}.items.1": a JSON Schema object or boolean`],
            [cyclic, `${path}.properties.self": a JSON Schema without circular references`],
        ];
        for (const [value, message] of invalid) {
            expect(await run(value)).toMatchObject({ ok: false, error: { code: 'invalid_request', message } });
        }
        expect(adapter.calls).toHaveLength(valid.length);
    });

    it('checks sub-schemas under combinator, conditional, map and tuple keywords', async () => {
        const adapter = createFakeProvider('p', { json: 'native' });
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { json: true } } } } },
        });
        adapter.reply(answer('{}'));
        /** Generates with json_schema format of given schema. */
        function run(value: unknown) {
            const format = { type: 'json_schema', json_schema: { name: 'w', schema: value } };
            return llm.generate({ model: 'm', messages, response_format: format as never });
        }
        const object = { type: 'object', properties: { a: { type: 'string' } } };
        const valid: unknown[] = [
            { type: 'object', anyOf: [object, true], oneOf: [object], allOf: [object, object] },
            { $defs: { A: object, B: false }, definitions: { C: object }, anyOf: [{ $ref: '#/$defs/A' }] },
            { type: 'object', patternProperties: { '^x': object }, propertyNames: { pattern: '^[a-z]+$' } },
            { type: 'array', contains: object, prefixItems: [object, true] },
            // biome-ignore lint/suspicious/noThenProperty: JSON Schema `then` keyword is the case under test
            { if: object, then: object, else: false, not: { type: 'null' } },
            { type: 'object', properties: { enum: { enum: [null] }, anyOf: { type: 'string' } } },
            { type: 'object', default: { anyOf: null }, examples: [{ items: null }] },
        ];
        for (const value of valid) {
            const result = await run(value);
            expect(['ok', 'invalid_json']).toContain(result.ok ? 'ok' : result.error.code);
        }
        const cyclic: Record<string, unknown> = { type: 'object' };
        cyclic.$defs = { R: { anyOf: [cyclic] } };
        const path = 'Invalid value for option "response_format.json_schema.schema';
        const schemaOrBoolean = 'a JSON Schema object or boolean';
        const invalid: [unknown, string][] = [
            [{ type: 'object', anyOf: [null] }, `${path}.anyOf.0": ${schemaOrBoolean}`],
            [{ $defs: { R: null } }, `${path}.$defs.R": ${schemaOrBoolean}`],
            [{ anyOf: {} }, `${path}.anyOf": an array of JSON Schemas`],
            [{ oneOf: [object, 1] }, `${path}.oneOf.1": ${schemaOrBoolean}`],
            [{ allOf: object }, `${path}.allOf": an array of JSON Schemas`],
            [{ prefixItems: { 0: object } }, `${path}.prefixItems": an array of JSON Schemas`],
            [{ definitions: [object] }, `${path}.definitions": an object of JSON Schemas`],
            [{ patternProperties: { '^x': 'string' } }, `${path}.patternProperties.^x": ${schemaOrBoolean}`],
            [{ properties: [object] }, `${path}.properties": an object of JSON Schemas`],
            [{ additionalProperties: null }, `${path}.additionalProperties": ${schemaOrBoolean}`],
            [{ not: [] }, `${path}.not": ${schemaOrBoolean}`],
            [{ contains: 'x' }, `${path}.contains": ${schemaOrBoolean}`],
            // biome-ignore lint/suspicious/noThenProperty: JSON Schema `then` keyword is the case under test
            [{ if: object, then: null }, `${path}.then": ${schemaOrBoolean}`],
            [{ else: 0 }, `${path}.else": ${schemaOrBoolean}`],
            [{ propertyNames: [] }, `${path}.propertyNames": ${schemaOrBoolean}`],
            [
                { anyOf: [{ properties: { a: { allOf: [null] } } }] },
                `${path}.anyOf.0.properties.a.allOf.0": ${schemaOrBoolean}`,
            ],
            [cyclic, `${path}.$defs.R.anyOf.0": a JSON Schema without circular references`],
        ];
        for (const [value, message] of invalid) {
            expect(await run(value)).toMatchObject({ ok: false, error: { code: 'invalid_request', message } });
        }
        const parameters = { type: 'object', properties: { a: { oneOf: [null] } } };
        const tools = [{ type: 'function' as const, function: { name: 'f', parameters } }];
        expect(await llm.generate({ model: 'm', messages, tools })).toMatchObject({
            ok: false,
            error: {
                code: 'invalid_request',
                message: `Invalid value for option "tools.function.parameters.properties.a.oneOf.0": ${schemaOrBoolean}`,
            },
        });
        expect(adapter.calls).toHaveLength(valid.length);
    });

    it('fails with unsupported for a model without JSON output', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { json: false } } } } },
        });
        expect(await llm.generate({ model: 'm', messages, response_format: { type: 'json_object' } })).toMatchObject({
            ok: false,
            error: { code: 'unsupported' },
        });
    });
});

describe('tool calls', () => {
    const tools: LlmTool[] = [
        {
            type: 'function',
            function: {
                name: 'get_weather',
                description: 'Weather of a city',
                parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
            },
        },
    ];
    const toolCall: LlmToolCall = {
        id: 'c1',
        type: 'function',
        function: { name: 'get_weather', arguments: '{"city":"Graz"}' },
    };
    const round: LlmMessage[] = [
        { role: 'user', content: 'Weather in Graz?' },
        { role: 'assistant', content: null, tool_calls: [toolCall] },
        { role: 'tool', tool_call_id: 'c1', content: '{"sky":"sunny"}' },
    ];

    it('accepts an assistantMessage with reasoning and extra_content followed by a toolMessage', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { tools: true } } } } },
        });
        const call: LlmToolCall = { ...toolCall, extra_content: { google: { thought_signature: 'sig' } } };
        const history: LlmMessage[] = [
            { role: 'user', content: 'Weather in Graz?' },
            assistantMessage({ text: '', toolCalls: [call], reasoning: 'look it up' }),
            toolMessage(call, { sky: 'sunny', id: 1n }),
        ];
        expect(await llm.generate({ model: 'm', messages: history, tools })).toMatchObject({ ok: true });
        expect(adapter.calls[0]?.request.messages).toEqual(history);
    });

    it('passes tools, tool_choice and tool turns through and returns the tool calls without checking JSON', async () => {
        const adapter = createFakeProvider();
        adapter.reply((call) => ({
            ok: true,
            data: {
                text: '',
                toolCalls: [toolCall],
                model: call.request.model,
                usage: { inputTokens: 10, outputTokens: 5, reasoningTokens: 0 },
                finishReason: 'tool_calls',
                ms: 1,
            },
        }));
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { tools: true, json: true } } } } },
        });
        const result = await llm.generate({
            model: 'm',
            messages: round,
            tools,
            tool_choice: { type: 'function', function: { name: 'get_weather' } },
            response_format: { type: 'json_schema', json_schema: { name: 'weather', schema } },
        });
        expect(result).toMatchObject({ ok: true, data: { text: '', toolCalls: [toolCall] } });
        expect(result.ok && 'json' in result.data).toBe(false);
        expect(adapter.calls[0]?.request).toMatchObject({
            messages: round,
            tools,
            tool_choice: { type: 'function', function: { name: 'get_weather' } },
        });
    });

    it('drops tool_choice without tools', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { tools: true } } } } },
        });
        await llm.generate({ model: 'm', messages, tool_choice: 'auto' });
        await llm.generate({ model: 'm', messages, tools, tool_choice: 'auto' });
        expect(adapter.calls[0]?.request).toEqual({ model: 'm', messages });
        expect(adapter.calls[1]?.request).toEqual({ model: 'm', messages, tools, tool_choice: 'auto' });
    });

    it('fails with unsupported for tools or a tool round trip on a model without tools', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { tools: false } } } } },
        });
        const cases: LlmRequest<'m'>[] = [
            { model: 'm', messages, tools },
            { model: 'm', messages: round },
        ];
        for (const request of cases) {
            expect(await llm.generate(request)).toMatchObject({
                ok: false,
                error: { code: 'unsupported', retryable: false },
            });
        }
        expect(await llm.generate({ model: 'm', messages, tool_choice: 'none' })).toMatchObject({ ok: true });
        expect(adapter.calls).toHaveLength(1);
    });

    it('fails with invalid_request for invalid tools, tool_choice and tool turns', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { tools: true } } } } },
        });
        const fn = tools[0]?.function as LlmTool['function'];
        const badArguments = { ...toolCall, function: { name: 'f', arguments: '[1]' } };
        const invalid: [Record<string, unknown>, RegExp][] = [
            [{ tools: [] }, /Invalid value for option "tools"/],
            [{ tools: [{ type: 'web', function: fn }] }, /"tools\.type"/],
            [{ tools: [{ type: 'function', function: { ...fn, name: 'get weather' } }] }, /tools\.function\.name/],
            [{ tools: [...tools, ...tools] }, /get_weather is declared twice/],
            [{ tools: [{ type: 'function', function: { ...fn, parameters: [] } }] }, /tools\.function\.parameters/],
            [{ tools: [{ type: 'function', function: { ...fn, extra: 1 } }] }, /"tools\.function\.extra" is not/],
            [{ tool_choice: 'any' }, /Invalid value for option "tool_choice"/],
            [{ tools, tool_choice: { type: 'function', function: { name: 'other' } } }, /tool_choice\.function\.name/],
            [{ messages: [{ role: 'assistant', content: null }] }, /content null only with tool_calls/],
            [
                { messages: [{ role: 'assistant', content: null, tool_calls: [badArguments] }] },
                /tool_calls\.function\.arguments/,
            ],
            [
                { messages: [{ role: 'assistant', content: null, tool_calls: [{ ...toolCall, type: 'x' }] }] },
                /tool_calls\.type/,
            ],
            [{ messages: [{ role: 'tool', tool_call_id: 'c9', content: 'x' }] }, /tool_call_id/],
            [{ messages: [{ role: 'user', content: 'x', tool_call_id: 'c1' }] }, /tool_call_id only in tool messages/],
            [{ messages: [{ role: 'user', content: 'x', reasoning: 'r' }] }, /reasoning only in assistant messages/],
        ];
        for (const [fields, message] of invalid) {
            const turns = [...messages, ...((fields.messages as unknown[] | undefined) ?? [])];
            expect(await llm.generate({ model: 'm', ...fields, messages: turns } as never)).toMatchObject({
                ok: false,
                error: { code: 'invalid_request', message: expect.stringMatching(message), retryable: false },
            });
        }
        expect(() =>
            createLlm({
                providers: { p: { adapter, models: { m: { model: 'm', capabilities: { tools: 'yes' } as never } } } },
            }),
        ).toThrow(/tools/);
        expect(adapter.calls).toHaveLength(0);
    });

    it('counts tool definitions, tool calls, tool results and assistant reasoning in the token estimate', () => {
        expect(estimateTokens({ model: 'm', messages, max_tokens: 1 }, 1)).toBe(3);
        expect(estimateTokens({ model: 'm', messages, tools, max_tokens: 1 }, 1)).toBe(
            3 + JSON.stringify(tools).length,
        );
        const turns: LlmMessage[] = [
            { role: 'assistant', content: null, reasoning: 'abc', tool_calls: [toolCall] },
            { role: 'tool', tool_call_id: 'c1', content: 'sunny' },
        ];
        expect(estimateTokens({ model: 'm', messages: turns, max_tokens: 1 }, 1)).toBe(
            'abc'.length + 'get_weather'.length + '{"city":"Graz"}'.length + 'sunny'.length + 1,
        );
    });
});

describe('limits', () => {
    it('retries a 429 after its retryAfterMs and reports the answer', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({ providers: { p: { adapter, models: { m: 'm' } } } });
        adapter.reply(
            failure({ code: 'rate_limited', message: '429', retryable: true, retryAfterMs: 20 }),
            answer('ok'),
        );
        expect(await llm.generate({ model: 'm', messages })).toMatchObject({ ok: true, data: { text: 'ok' } });
        expect(adapter.calls).toHaveLength(2);
    });

    it('fails immediately on non-retryable errors with model and provider names', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({ providers: { p: { adapter, models: { m: 'm' } } } });
        adapter.reply(failure({ code: 'http', status: 400, message: 'bad', retryable: false }));
        expect(await llm.generate({ model: 'm', messages })).toEqual({
            ok: false,
            error: {
                code: 'http',
                status: 400,
                message: 'bad',
                retryable: false,
                name: 'm',
                provider: 'p',
                settings: { name: 'm', provider: 'p', model: 'm' },
                attempts: 1,
            },
        });
        expect(adapter.calls).toHaveLength(1);
    });

    it('books tokens with charsPerToken and refuses a call that can never fit', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: {
                p: {
                    adapter,
                    sharedLimits: { tpm: 10 },
                    charsPerToken: 0.5,
                    models: { coarse: 'm', fine: { model: 'm', charsPerToken: 4 } },
                },
            },
        });
        const request = { messages: [{ role: 'user' as const, content: 'abcdefgh' }], max_tokens: 1 };
        expect(await llm.generate({ model: 'coarse', ...request })).toMatchObject({
            ok: false,
            error: { code: 'limiter', limiterCode: 'COST_EXCEEDS_LIMIT' },
        });
        // Fine books 3 estimated tokens; actual usage of 15 exceeds TPM but resolves immediately
        expect(await llm.generate({ model: 'fine', ...request })).toMatchObject({ ok: true });
        expect(adapter.calls).toHaveLength(1);
    });

    it('shares provider concurrency across models', async () => {
        let running = 0;
        let peak = 0;
        const adapter = createFakeProvider();
        const base = adapter.generate;
        adapter.generate = async (request, options) => {
            running += 1;
            peak = Math.max(peak, running);
            await Bun.sleep(10);
            running -= 1;
            return base(request, options);
        };
        const llm = createLlm({
            providers: { p: { adapter, sharedLimits: { concurrency: 1 }, models: { a: 'a', b: 'b' } } },
        });
        await Promise.all([
            llm.generate({ model: 'a', messages }),
            llm.generate({ model: 'b', messages }),
            llm.generate({ model: 'a', messages }),
        ]);
        expect(peak).toBe(1);
        expect(adapter.calls).toHaveLength(3);
        peak = 0;
        const unlimited = createLlm({ providers: { p: { adapter, models: { a: 'a' } } } });
        await Promise.all(Array.from({ length: 8 }, () => unlimited.generate({ model: 'a', messages })));
        expect(peak).toBe(8);
    });

    it('enforces both shared and model limits', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            limiterOptions: { maxWaitMs: 100 },
            providers: {
                p: {
                    adapter,
                    sharedLimits: { rpm: 2 },
                    models: { a: { model: 'a', modelLimits: { rpm: 1 } }, b: { model: 'b', modelLimits: { rpm: 5 } } },
                },
            },
        });
        expect(await llm.generate({ model: 'a', messages })).toMatchObject({ ok: true });
        expect(await llm.generate({ model: 'a', messages })).toMatchObject({
            ok: false,
            error: { code: 'limiter', limiterCode: 'MAX_WAIT_EXCEEDED' },
        });
        expect(await llm.generate({ model: 'b', messages })).toMatchObject({ ok: true });
        expect(await llm.generate({ model: 'b', messages })).toMatchObject({
            ok: false,
            error: { limiterCode: 'MAX_WAIT_EXCEEDED' },
        });
        expect(adapter.calls).toHaveLength(2);
    });

    it('spaces calls with intervalMs and passes limiterOptions', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            limiterOptions: { maxWaitMs: 100, maxRetries: 0 },
            providers: {
                p: { adapter, models: { m: { model: 'm', modelLimits: { intervalMs: 3000 } } } },
            },
        });
        expect(await llm.generate({ model: 'm', messages })).toMatchObject({ ok: true });
        expect(await llm.generate({ model: 'm', messages })).toMatchObject({
            ok: false,
            error: { limiterCode: 'MAX_WAIT_EXCEEDED' },
        });
        const once = createLlm({
            limiterOptions: { maxRetries: 0 },
            providers: { p: { adapter, models: { m: 'm' } } },
        });
        adapter.reply(failure({ code: 'timeout', message: 'slow', retryable: true }), answer('ok'));
        expect(await once.generate({ model: 'm', messages })).toMatchObject({ ok: false, error: { code: 'timeout' } });
    });

    it('returns provider error when 429 or 5xx backoff exceeds maxWaitMs', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            limiterOptions: { maxWaitMs: 100, maxRetries: 0 },
            providers: { p: { adapter, models: { m: 'm' } } },
        });
        adapter.reply(failure({ code: 'rate_limited', message: '429', retryable: true, retryAfterMs: 3_600_000 }));
        expect(await llm.m('hi')).toMatchObject({
            ok: false,
            error: { code: 'rate_limited', retryable: true, retryAfterMs: 3_600_000, attempts: 1 },
        });
        const outage = createLlm({
            limiterOptions: { maxWaitMs: 100, maxRetries: 0 },
            providers: { p: { adapter, models: { m: 'm' } } },
        });
        adapter.reply(failure({ code: 'http', status: 500, message: 'down', retryable: true, retryAfterMs: 1000 }));
        expect(await outage.m('hi')).toMatchObject({
            ok: false,
            error: { code: 'http', status: 500, message: 'down', retryable: true },
        });
    });

    it('aborts immediately when signal fires while waiting on limits or backoff', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', modelLimits: { intervalMs: 3000 } } } } },
        });
        expect(await llm.m('hi')).toMatchObject({ ok: true });
        const waiting = new AbortController();
        setTimeout(() => waiting.abort(), 50);
        const started = Date.now();
        expect(await llm.m('hi', { signal: waiting.signal })).toMatchObject({
            ok: false,
            error: { code: 'aborted', retryable: false, attempts: 0 },
        });
        expect(Date.now() - started).toBeLessThan(1000);
        const outage = createLlm({ providers: { p: { adapter, models: { m: 'm' } } } });
        adapter.reply(failure({ code: 'http', status: 503, message: 'down', retryable: true, retryAfterMs: 5000 }));
        const backoff = new AbortController();
        setTimeout(() => backoff.abort(), 50);
        expect(await outage.m('hi', { signal: backoff.signal })).toMatchObject({
            ok: false,
            error: { code: 'aborted', attempts: 1 },
        });
        expect(Date.now() - started).toBeLessThan(2000);
    });

    it('refunds token booking on 429 so retry fits tpm', async () => {
        const adapter = createFakeProvider();
        adapter.reply(
            failure({ code: 'rate_limited', message: '429', retryable: true, retryAfterMs: 50 }),
            answer('ok'),
        );
        const llm = createLlm({
            limiterOptions: { maxWaitMs: 1000 },
            providers: { p: { adapter, models: { m: { model: 'm', modelLimits: { tpm: 15 } } } } },
        });
        // Books 11 tokens; without refunding on 429, retry would require 22 of 15 TPM and stall
        expect(await llm.m('hi', { max_tokens: 10 })).toMatchObject({ ok: true, data: { attempts: 2 } });
    });

    it('dispatches calls for a model in FIFO order', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', modelLimits: { concurrency: 1 } } } } },
        });
        const texts = ['a', 'b', 'c', 'd', 'e'];
        await Promise.all(texts.map((text) => llm.m(text)));
        expect(adapter.calls.map((call) => call.request.messages[0]?.content)).toEqual(texts);
    });

    it('dispatches next call while retry is in backoff for normal and batch calls', async () => {
        for (const batch of [false, true]) {
            const adapter = createFakeProvider();
            adapter.reply(failure({ code: 'timeout', message: 'slow', retryable: true }), answer('ok'));
            const llm = createLlm({
                limiterOptions: { retryDelayMs: 200, maxRetryDelayMs: 200 },
                providers: {
                    p: {
                        adapter,
                        models: {
                            m: {
                                model: 'm',
                                modelLimits: { concurrency: 1 },
                                capabilities: { batch: { supported: true } },
                            },
                        },
                    },
                },
            });
            const done: string[] = [];
            await Promise.all(
                ['a', 'b'].map(async (text) => {
                    const result = await llm.generate({
                        model: 'm',
                        messages: [{ role: 'user', content: text }],
                        batch,
                    });
                    expect(result).toMatchObject({ ok: true });
                    done.push(text);
                }),
            );
            expect(done).toEqual(['b', 'a']);
        }
    });

    it('sends single recovery probe after 429 before releasing queued calls', async () => {
        let running = 0;
        const runningAtStart: number[] = [];
        const adapter = createFakeProvider();
        const base = adapter.generate;
        adapter.generate = async (request, options) => {
            runningAtStart.push(running);
            running += 1;
            await Bun.sleep(20);
            running -= 1;
            return base(request, options);
        };
        adapter.reply(
            failure({ code: 'rate_limited', message: '429', retryable: true, retryAfterMs: 100 }),
            answer('ok'),
        );
        const llm = createLlm({ providers: { p: { adapter, models: { m: 'm' } } } });
        const first = llm.m('a');
        await Bun.sleep(50);
        const results = await Promise.all([first, llm.m('b'), llm.m('c'), llm.m('d')]);
        expect(results.every((result) => result.ok)).toBe(true);
        // Initial 429, single recovery probe, then remaining three calls concurrently
        expect(runningAtStart).toEqual([0, 0, 0, 1, 2]);
    });

    it('rejects calls without dispatching while lock exceeds maxWaitMs', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            limiterOptions: { maxWaitMs: 100, maxRetries: 0 },
            providers: { p: { adapter, models: { m: 'm' } } },
        });
        adapter.reply(
            failure({ code: 'rate_limited', message: '429', retryable: true, retryAfterMs: 300 }),
            answer('ok'),
        );
        expect(await llm.m('hi')).toMatchObject({
            ok: false,
            error: { code: 'rate_limited', retryAfterMs: 300, attempts: 1 },
        });
        expect(await llm.m('hi')).toMatchObject({
            ok: false,
            error: { code: 'limiter', limiterCode: 'MAX_WAIT_EXCEEDED', attempts: 0 },
        });
        expect(adapter.calls).toHaveLength(1);
        await Bun.sleep(250);
        expect(await llm.m('hi')).toMatchObject({ ok: true, data: { attempts: 1 } });
        expect(adapter.calls).toHaveLength(2);
    });

    it('aborts queued call immediately without calling adapter', async () => {
        const adapter = createFakeProvider();
        const base = adapter.generate;
        adapter.generate = async (request, options) => {
            await Bun.sleep(50);
            return base(request, options);
        };
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', modelLimits: { concurrency: 1 } } } } },
        });
        const first = llm.m('first');
        const controller = new AbortController();
        const second = llm.m('second', { signal: controller.signal });
        controller.abort();
        expect(await second).toMatchObject({ ok: false, error: { code: 'aborted', attempts: 0 } });
        expect(await first).toMatchObject({ ok: true });
        expect(adapter.calls).toHaveLength(1);
    });
});

describe('batch calls', () => {
    it('resolves batch flag from call or capability default', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: {
                p: {
                    adapter,
                    models: {
                        flash: { model: 'id', capabilities: { batch: { supported: true } } },
                        flashBatch: { model: 'id', capabilities: { batch: { supported: true, default: true } } },
                    },
                },
            },
        });
        await llm.generate({ model: 'flash', messages });
        await llm.generate({ model: 'flashBatch', messages });
        await llm.generate({ model: 'flash', messages, batch: true });
        await llm.generate({ model: 'flashBatch', messages, batch: false });
        expect(adapter.calls.map((call) => call.request.batch)).toEqual([undefined, true, true, undefined]);
        expect(llm.models.map((model) => model.capabilities.batch)).toEqual([
            { supported: true, default: false },
            { supported: true, default: true },
        ]);
    });

    it('returns unsupported when batch is true on unsupported model', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: {
                p: { adapter, models: { m: 'm', b: { model: 'b', capabilities: { batch: { supported: false } } } } },
            },
        });
        for (const result of [
            await llm.generate({ model: 'm', messages, batch: true }),
            await llm.generate({ model: 'b', messages, batch: true }),
        ]) {
            expect(result).toMatchObject({
                ok: false,
                error: { code: 'unsupported', message: 'The model does not support batch calls', retryable: false },
            });
        }
        expect(await llm.generate({ model: 'm', messages, batch: false })).toMatchObject({ ok: true });
        expect(await llm.generate({ model: 'm', messages, batch: 'yes' } as never)).toMatchObject({
            ok: false,
            error: { code: 'invalid_request', message: expect.stringContaining('"batch"') },
        });
        expect(adapter.calls.map((call) => call.request)).toEqual([{ model: 'm', messages }]);
    });

    it('applies only model concurrency to batch calls, ignoring other limits and maxWaitMs', async () => {
        let running = 0;
        let peak = 0;
        const adapter = createFakeProvider();
        const base = adapter.generate;
        adapter.generate = async (request, options) => {
            running += 1;
            peak = Math.max(peak, running);
            await Bun.sleep(40);
            running -= 1;
            return base(request, options);
        };
        const llm = createLlm({
            limiterOptions: { maxWaitMs: 10 },
            providers: {
                p: {
                    adapter,
                    sharedLimits: { rpm: 1, tpm: 10 },
                    models: {
                        m: {
                            model: 'm',
                            capabilities: { batch: { supported: true } },
                            modelLimits: { concurrency: 1, rpd: 1, intervalMs: 60_000 },
                        },
                    },
                },
            },
        });
        const request = { model: 'm' as const, messages, batch: true };
        const results = await Promise.all([llm.generate(request), llm.generate(request), llm.generate(request)]);
        expect(results.every((result) => result.ok)).toBe(true);
        expect(peak).toBe(1);
        expect(await llm.generate({ model: 'm', messages })).toMatchObject({
            ok: false,
            error: { code: 'limiter', limiterCode: 'COST_EXCEEDS_LIMIT' },
        });
    });

    it('retries transient batch errors without global pause and fails fatal ones immediately', async () => {
        const adapter = createFakeProvider();
        adapter.reply(
            failure({ code: 'rate_limited', message: 'quota', retryable: true }),
            failure({ code: 'http', message: 'bad', retryable: false, status: 400 }),
        );
        const llm = createLlm({
            limiterOptions: { retryDelayMs: 1 },
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { batch: { supported: true } } } } } },
        });
        expect(await llm.generate({ model: 'm', messages, batch: true })).toEqual({
            ok: false,
            error: {
                code: 'http',
                message: 'bad',
                retryable: false,
                status: 400,
                name: 'm',
                provider: 'p',
                settings: { name: 'm', provider: 'p', model: 'm', batch: true },
                attempts: 2,
            },
        });
        expect(adapter.calls).toHaveLength(2);
        adapter.reply(answer('ok'));
        // Normal call is not blocked by batch 429; batch calls use separate queue
        expect(await llm.generate({ model: 'm', messages })).toMatchObject({ ok: true });
    });
});

describe('invalid configurations and requests', () => {
    it('throws TypeError on invalid config before any adapter call', () => {
        const adapter = createFakeProvider();
        const invalid: [unknown, RegExp][] = [
            [{ providers: {} }, /option "providers"/],
            [{ providers: { p: { adapter, models: {} } } }, /providers\.p\.models/],
            [{ providers: { p: { adapter: {}, models: { m: 'm' } } } }, /providers\.p\.adapter/],
            [
                { providers: { p: { adapter, models: { m: 'm' } }, q: { adapter, models: { m: 'n' } } } },
                /the name is taken/,
            ],
            [
                { providers: { p: { adapter, sharedLimits: { rpm: 0 }, models: { m: 'm' } } } },
                /providers\.p\.sharedLimits\.rpm/,
            ],
            [{ providers: { p: { adapter, limits: {}, models: { m: 'm' } } } }, /Option "providers\.p\.limits"/],
            [
                { providers: { p: { adapter, models: { m: { model: 'm', modelLimits: { maxRetries: 1 } } } } } },
                /Option "providers\.p\.models\.m\.modelLimits\.maxRetries" is not supported/,
            ],
            [
                {
                    providers: {
                        p: { adapter, models: { m: { model: 'm', modelLimits: { intervalMs: 0 } } } },
                    },
                },
                /providers\.p\.models\.m\.modelLimits\.intervalMs/,
            ],
            [
                { providers: { p: { adapter, models: { m: { model: 'm', charsPerToken: -1 } } } } },
                /providers\.p\.models\.m\.charsPerToken/,
            ],
            [
                { providers: { p: { adapter, models: { m: 'm' } } }, limiterOptions: { maxRetries: -1 } },
                /limiterOptions\.maxRetries/,
            ],
            [{ providers: { p: { adapter, models: { m: 'm' } } }, limiterOptions: { timeoutMs: 1 } }, /timeoutMs/],
            [{ providers: { p: { adapter, models: { m: 'm' } } }, limiter: {} }, /Option "limiter" is not supported/],
            [{ providers: { p: { adapter, extra: [], models: { m: 'm' } } } }, /providers\.p\.extra/],
            [{ providers: { p: { adapter: { ...adapter, mergeExtra: 1 }, models: { m: 'm' } } } }, /p\.adapter/],
            [{ providers: { p: { adapter, models: { m: 'm' } } }, defaults: { model: 'x' } }, /defaults\.model/],
            [
                { providers: { p: { adapter, sharedLimits: { buckets: [null] }, models: { m: 'm' } } } },
                /option "providers\.p\.sharedLimits\.buckets\.0": a smart-limit bucket with getWaitMs, reserve, settle/,
            ],
            [
                { providers: { p: { adapter, models: { m: { model: 'm', modelLimits: { buckets: [{}] } } } } } },
                /option "providers\.p\.models\.m\.modelLimits\.buckets\.0": a smart-limit bucket/,
            ],
            [
                {
                    providers: {
                        p: { adapter, models: { m: { model: 'm', modelLimits: { buckets: [createRPM(1), 'x'] } } } },
                    },
                },
                /modelLimits\.buckets\.1"/,
            ],
            [
                {
                    providers: {
                        p: {
                            adapter,
                            sharedLimits: { buckets: [{ ...createRPM(1), subscribe: 1 }] },
                            models: { m: 'm' },
                        },
                    },
                },
                /option "providers\.p\.sharedLimits\.buckets\.0\.subscribe": a function/,
            ],
            [
                { providers: { p: { adapter, sharedLimits: { buckets: [createRPM(0)] }, models: { m: 'm' } } } },
                /option "providers\.p\.sharedLimits\.buckets\.0": \S/,
            ],
            [
                {
                    providers: {
                        p: {
                            adapter,
                            sharedLimits: {
                                buckets: [
                                    Object.defineProperty({ ...createRPM(1) }, 'settle', {
                                        get() {
                                            throw new Error('getter failed');
                                        },
                                    }),
                                ],
                            },
                            models: { m: 'm' },
                        },
                    },
                },
                /option "providers\.p\.sharedLimits\.buckets\.0": a smart-limit bucket/,
            ],
        ];
        for (const [config, message] of invalid) {
            expect(() => createLlm(config as never)).toThrow(message);
        }
        expect(adapter.calls).toHaveLength(0);
    });

    it('accepts one smart-limit bucket instance shared by two providers and a model', async () => {
        const bucket = createRPM(100);
        const first = createFakeProvider('a');
        const second = createFakeProvider('b');
        const llm = createLlm({
            providers: {
                a: { adapter: first, sharedLimits: { buckets: [bucket] }, models: { x: 'x' } },
                b: {
                    adapter: second,
                    sharedLimits: { buckets: [bucket] },
                    models: { y: { model: 'y', modelLimits: { buckets: [bucket] } } },
                },
            },
        });
        expect(await llm.x('hi')).toMatchObject({ ok: true });
        expect(await llm.y('hi')).toMatchObject({ ok: true });
    });

    it('returns invalid_request instead of throwing on invalid generate and listModels calls', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({ defaults: { temperature: 0.5 }, providers: { p: { adapter, models: { m: 'm' } } } });
        const image = { type: 'image_url' as const, image_url: { url: 'https://x' } };
        const cyclicExtra: Record<string, unknown> = {};
        cyclicExtra.self = { parent: cyclicExtra };
        const defs: Record<string, unknown> = {};
        defs.node = { $defs: defs };
        const cyclicFormat = { type: 'json_schema' as const, json_schema: { name: 'w', schema: { $defs: defs } } };
        const cases: [Promise<unknown>, RegExp][] = [
            [
                llm.generate({ model: 'm', messages }, { extra: cyclicExtra }),
                /option "extra": a value without circular references/,
            ],
            [
                llm.generate({ model: 'm', messages, response_format: cyclicFormat }),
                /option "response_format\.json_schema\.schema\.\$defs\.node\.\$defs\.node": a JSON Schema without circular/,
            ],
            [llm.generate(null as never), /option "request"/],
            [llm.generate({ model: 'x' as 'm', messages }), /x is not configured/],
            [llm.generate({ model: 'm', messages, n: 2 } as never), /Option "n" is not supported/],
            [llm.generate({ model: 'm', messages: [{ role: 'user', content: [image] }] }), /base64 data URL/],
            [llm.generate({ model: 'm', messages }, { timeoutMs: 1 } as never), /Option "timeoutMs"/],
            [llm.generate({ model: 'm', messages }, { extra: [] } as never), /option "extra"/],
            [llm.generate({ model: 'm', messages }, { signal: 'x' } as never), /option "signal"/],
            [llm.listModels('q'), /q is not configured/],
            [llm.listModels('p', { signal: 'x' } as never), /option "signal"/],
            [llm.listModels('p', { foo: 1 } as never), /Option "foo" is not supported/],
            [llm.listModels('p', null as never), /option "options"/],
        ];
        for (const [result, message] of cases) {
            expect(await result).toMatchObject({
                ok: false,
                error: { code: 'invalid_request', message: expect.stringMatching(message), retryable: false },
            });
        }
        expect(adapter.calls).toHaveLength(0);
    });

    it('returns invalid_request for invalid messages, content parts, and image URLs', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { images: true, tools: true } } } } },
        });
        const url = 'data:image/png;base64,AQID';
        const call = { id: 'c1', type: 'function', function: { name: 'f', arguments: '{}' } };
        /** Creates history of one user message with given content. */
        function user(content: unknown) {
            return [{ role: 'user', content }];
        }
        const invalid: [unknown[], string][] = [
            [[{ role: 'user', content: 'x', name: 'bob' }], 'Option "messages.name" is not supported'],
            [[...messages, { role: 'assistant', content: 'x', refusal: null }], 'Option "messages.refusal"'],
            [user([]), 'Invalid value for option "messages.content": a string or one or more parts'],
            [[...messages, { role: 'assistant', content: [] }], 'option "messages.content"'],
            [
                [...messages, { role: 'assistant', content: 'hi', tool_calls: [] }],
                'Invalid value for option "messages.tool_calls": one or more tool calls',
            ],
            [
                [...messages, { role: 'assistant', content: null, tool_calls: [] }],
                'Invalid value for option "messages.tool_calls": one or more tool calls',
            ],
            [user([{ type: 'text', text: 'x', cache: true }]), 'Option "messages.content.cache" is not supported'],
            [
                user([{ type: 'image_url', image_url: { url }, text: 'x' }]),
                'Option "messages.content.text" is not supported',
            ],
            [
                user([{ type: 'image_url', image_url: { url, size: 1 } }]),
                'Option "messages.content.image_url.size" is not supported',
            ],
            [
                user([{ type: 'image_url', image_url: { url, detail: 'ultra' } }]),
                'Invalid value for option "messages.content.image_url.detail": auto, low or high',
            ],
            [user([{ type: 'image_url', image_url: { url: 'data:image/png;base64,' } }]), 'base64 data URL'],
            [user([{ type: 'image_url' }]), 'base64 data URL'],
        ];
        for (const [turns, message] of invalid) {
            expect(await llm.generate({ model: 'm', messages: turns as LlmMessage[] })).toMatchObject({
                ok: false,
                error: { code: 'invalid_request', message: expect.stringContaining(message), retryable: false },
            });
        }
        expect(adapter.calls).toHaveLength(0);
        // what assistantMessage and toolMessage build, with provider data in extra_content, stays valid
        const output = {
            text: '',
            reasoning: 'r',
            toolCalls: [{ ...call, extra_content: { anthropic: { content: [{ type: 'thinking' }] } } } as LlmToolCall],
        };
        const valid: LlmMessage[] = [
            {
                role: 'user',
                content: [
                    { type: 'text', text: 'x' },
                    { type: 'image_url', image_url: { url, detail: 'low' } },
                ],
            },
            assistantMessage(output),
            toolMessage(call, { ok: 1 }),
        ];
        expect(await llm.generate({ model: 'm', messages: valid })).toMatchObject({ ok: true });
    });

    it('turns a throwing adapter into unexpected', async () => {
        const adapter = createFakeProvider();
        adapter.generate = () => {
            throw new Error('boom');
        };
        adapter.listModels = () => Promise.reject(new Error('down'));
        const llm = createLlm({ providers: { p: { adapter, models: { m: 'm' } } } });
        expect(await llm.generate({ model: 'm', messages })).toEqual({
            ok: false,
            error: {
                code: 'unexpected',
                message: 'boom',
                retryable: false,
                name: 'm',
                provider: 'p',
                settings: { name: 'm', provider: 'p', model: 'm' },
                attempts: 1,
            },
        });
        expect(await llm.listModels('p')).toEqual({
            ok: false,
            error: { code: 'unexpected', message: 'down', retryable: false },
        });
    });

    it('maps malformed adapter results to unexpected in normal and batch calls', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { batch: { supported: true } } } } } },
        });
        const results: unknown[] = [
            undefined,
            { ok: true },
            { ok: true, data: {} },
            { ok: true, data: { text: 'x', model: 'm' } },
            { ok: false },
            { ok: false, error: {} },
            { ok: false, error: { code: 'http', message: 'x' } },
        ];
        for (const malformed of results) {
            adapter.generate = async () => malformed as never;
            for (const batch of [false, true]) {
                expect(await llm.generate({ model: 'm', messages, batch })).toMatchObject({
                    ok: false,
                    error: { code: 'unexpected', retryable: false, attempts: 1 },
                });
            }
        }
        const usage = { inputTokens: 0, outputTokens: null, reasoningTokens: 2 };
        const lenient = { text: '', model: 'm', usage, finishReason: null, ms: 1 };
        adapter.generate = async () => ({ ok: true, data: lenient }) as never;
        for (const batch of [false, true]) {
            expect(await llm.generate({ model: 'm', messages, batch })).toMatchObject({ ok: true, data: { text: '' } });
        }
    });

    it('maps invalid adapter usage counts to unexpected', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'm', capabilities: { batch: { supported: true } } } } } },
        });
        const full = { inputTokens: 1, outputTokens: 1, reasoningTokens: null };
        const usages: unknown[] = [
            {},
            { inputTokens: 1, outputTokens: 1 },
            { ...full, inputTokens: '5' },
            { ...full, outputTokens: -1 },
            { ...full, reasoningTokens: Number.NaN },
            { ...full, inputTokens: Infinity },
        ];
        const results: unknown[] = usages.flatMap((usage) => [
            { ok: true, data: { text: '{}', model: 'm', usage, ms: 1 } },
            { ok: false, error: { code: 'http', message: 'x', retryable: false, usage } },
        ]);
        for (const malformed of results) {
            adapter.generate = async () => malformed as never;
            for (const batch of [false, true]) {
                expect(await llm.generate({ model: 'm', messages, batch })).toMatchObject({
                    ok: false,
                    error: { code: 'unexpected', message: 'fake returned a malformed result', attempts: 1 },
                });
            }
        }
        const error = { code: 'http', message: 'x', retryable: false, usage: full };
        adapter.generate = async () => ({ ok: false, error }) as never;
        expect(await llm.generate({ model: 'm', messages })).toMatchObject({ ok: false, error: { code: 'http' } });
    });
});

describe('provider extra hooks', () => {
    /** Fake provider whose validateExtra allows only key 'a'. */
    function strictProvider() {
        return {
            ...createFakeProvider(),
            validateExtra(extra: Record<string, unknown>, option: string) {
                for (const key of Object.keys(extra)) {
                    if (key !== 'a') {
                        throw new TypeError(`Option "${option}.${key}" is not supported`);
                    }
                }
            },
        };
    }

    it('validateExtra checks provider and model extra in createLlm and the call extra in generate', async () => {
        const adapter = strictProvider();
        expect(() => createLlm({ providers: { p: { adapter, extra: { b: 1 }, models: { m: 'm' } } } })).toThrow(
            'Option "providers.p.extra.b" is not supported',
        );
        expect(() =>
            createLlm({ providers: { p: { adapter, models: { m: { model: 'm', extra: { c: 1 } } } } } }),
        ).toThrow('Option "providers.p.models.m.extra.c" is not supported');
        const llm = createLlm({ providers: { p: { adapter, extra: { a: 1 }, models: { m: 'm' } } } });
        expect(await llm.generate({ model: 'm', messages }, { extra: { d: 1 } })).toEqual({
            ok: false,
            error: {
                code: 'invalid_request',
                message: 'Option "extra.d" is not supported',
                retryable: false,
                name: 'm',
                provider: 'p',
            },
        });
        expect(await llm.generate({ model: 'm', messages }, { extra: { a: 2 } })).toMatchObject({ ok: true });
        expect(adapter.calls.map((call) => call.extra)).toEqual([{ a: 2 }]);
    });

    it('rejects extra with own __proto__ key', async () => {
        const adapter = createFakeProvider();
        const extra = JSON.parse('{"__proto__": {"num_ctx": 1}}');
        expect(() => createLlm({ providers: { p: { adapter, extra, models: { m: 'id' } } } })).toThrow(
            'Invalid value for option "providers.p.extra"',
        );
        expect(() => createLlm({ providers: { p: { adapter, models: { m: { model: 'id', extra } } } } })).toThrow(
            'Invalid value for option "providers.p.models.m.extra"',
        );
        const llm = createLlm({ providers: { p: { adapter, models: { m: 'id' } } } });
        expect(await llm.m('hi', { extra })).toMatchObject({
            ok: false,
            error: { code: 'invalid_request', message: 'Invalid value for option "extra"' },
        });
        expect(adapter.calls).toHaveLength(0);
    });

    it('mergeExtra gets provider, model and call extra in that order', async () => {
        const layers: Record<string, unknown>[][] = [];
        const adapter = {
            ...createFakeProvider(),
            mergeExtra(given: Record<string, unknown>[]) {
                layers.push(given);
                return { merged: given.length };
            },
        };
        const llm = createLlm({
            providers: { p: { adapter, extra: { a: 1 }, models: { m: { model: 'm', extra: { b: 2 } }, n: 'n' } } },
        });
        await llm.generate({ model: 'm', messages }, { extra: { c: 3 } });
        await llm.generate({ model: 'n', messages });
        expect(layers).toEqual([
            [{ a: 1 }, { b: 2 }, { c: 3 }],
            [{ a: 1 }, {}, {}],
        ]);
        expect(adapter.calls[0]?.extra).toEqual({ merged: 3 });
    });

    it('default merge: preserves configured extra keys when call sets undefined', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: { p: { adapter, extra: { k: 1 }, models: { m: { model: 'm', extra: { j: 2 } } } } },
        });
        const result = await llm.generate({ model: 'm', messages }, { extra: { k: undefined, j: undefined, n: 3 } });
        expect(adapter.calls[0]?.extra).toEqual({ k: 1, j: 2, n: 3 });
        expect(result.ok && Object.keys(result.data.settings.extra ?? {})).toEqual(['k', 'j', 'n']);
        expect(result.ok && result.data.settings.extra).toEqual({ k: 1, j: 2, n: 3 });
    });

    it('passes cloned extra layers to mergeExtra', async () => {
        const seen: unknown[][] = [];
        const adapter = {
            ...createFakeProvider(),
            mergeExtra(given: Record<string, unknown>[]) {
                const nested = given.map((layer) => layer.nested as { deep: number } | undefined);
                seen.push(nested.map((entry) => entry?.deep));
                for (const entry of nested) {
                    if (entry !== undefined) {
                        entry.deep = 99;
                    }
                }
                return {};
            },
        };
        const llm = createLlm({ providers: { p: { adapter, extra: { nested: { deep: 1 } }, models: { m: 'm' } } } });
        const own = { nested: { deep: 2 } };
        await llm.generate({ model: 'm', messages }, { extra: own });
        await llm.generate({ model: 'm', messages }, { extra: own });
        expect(seen).toEqual([
            [1, undefined, 2],
            [1, undefined, 2],
        ]);
        expect(own.nested.deep).toBe(2);
    });
});

describe('settings, attempts and answers of errors', () => {
    const format = { type: 'json_schema' as const, json_schema: { name: 'word', schema } };
    const tool: LlmTool = { type: 'function', function: { name: 'lookup' } };

    /** Creates fake reply with reasoning and fixed token usage. */
    function reasoned(text: string): FakeReply {
        return (call) => ({
            ok: true,
            data: {
                text,
                reasoning: 'thought',
                model: call.request.model,
                usage: { inputTokens: 10, outputTokens: 5, reasoningTokens: 3 },
                finishReason: 'stop',
                ms: 1,
            },
        });
    }

    it('reports the settings as sent: every default layer, mapped reasoning, JSON mode, tool names, merged extra', async () => {
        const adapter = createFakeProvider('p', { tools: true, reasoning: true, json: 'strict' });
        const llm = createLlm({
            defaults: { temperature: 0.2, seed: 1 },
            providers: {
                p: {
                    adapter,
                    defaults: { top_p: 0.9 },
                    extra: { a: 1 },
                    models: {
                        flash: {
                            model: 'flash-id',
                            defaults: { max_completion_tokens: 50 },
                            extra: { b: 2 },
                            capabilities: {
                                json: true,
                                tools: true,
                                reasoning: { supported: true, default: 'low', levels: ['low', 'high'] },
                            },
                        },
                    },
                },
            },
        });
        adapter.reply(reasoned('{"kind":"noun"}'));
        const call: LlmModelCall = llm.flash;
        const result = await call('hi', {
            response_format: format,
            tools: [tool],
            tool_choice: 'none',
            reasoning_effort: 'medium',
            stop: 'END',
            extra: { c: 3 },
        });
        expect(result).toEqual({
            ok: true,
            data: {
                text: '{"kind":"noun"}',
                json: { kind: 'noun' },
                reasoning: 'thought',
                model: 'flash-id',
                usage: { inputTokens: 10, outputTokens: 5, reasoningTokens: 3 },
                finishReason: 'stop',
                ms: 1,
                name: 'flash',
                provider: 'p',
                settings: {
                    name: 'flash',
                    provider: 'p',
                    model: 'flash-id',
                    temperature: 0.2,
                    top_p: 0.9,
                    max_tokens: 50,
                    seed: 1,
                    stop: 'END',
                    reasoning_effort: 'high',
                    tool_choice: 'none',
                    jsonMode: 'strict',
                    tools: ['lookup'],
                    extra: { a: 1, b: 2, c: 3 },
                },
                attempts: 1,
            },
        });
    });

    it('clones settings, models, and config to prevent external mutation', async () => {
        const adapter = createFakeProvider();
        const defaults = { stop: ['END'] };
        const extra = { options: { num_ctx: 2048 } };
        const llm = createLlm({
            defaults,
            providers: { p: { adapter, extra, models: { m: { model: 'id', capabilities: { images: false } } } } },
        });
        defaults.stop.push('CONFIG');
        extra.options.num_ctx = 1;
        const first = await llm.m('hi');
        if (!first.ok) {
            throw new Error('expected an answer');
        }
        (first.data.settings.stop as string[]).push('LEAK');
        (first.data.settings.extra as typeof extra).options.num_ctx = 99;
        (llm.models[0] as { capabilities: { images: boolean } }).capabilities.images = true;
        await llm.m('hi');
        expect(adapter.calls[1]?.request.stop).toEqual(['END']);
        expect(adapter.calls[1]?.extra).toEqual({ options: { num_ctx: 2048 } });
        expect(
            await llm.m([{ type: 'text', text: 'hi' }, imagePart(new Uint8Array([1, 2]), 'image/png')]),
        ).toMatchObject({
            ok: false,
            error: { code: 'unsupported' },
        });
    });

    it('snapshots queued request so caller mutations do not affect execution', async () => {
        const adapter = createFakeProvider('p', { json: 'native', tools: true });
        const llm = createLlm({
            providers: {
                p: {
                    adapter,
                    models: {
                        m: { model: 'm', modelLimits: { concurrency: 1 }, capabilities: { json: true, tools: true } },
                    },
                },
            },
        });
        adapter.reply(answer('{"kind":"noun"}'));
        const turns: LlmMessage[] = [{ role: 'user', content: 'original' }];
        const stop = ['S'];
        const tools: LlmTool[] = [{ type: 'function', function: { name: 'f', parameters: { type: 'object' } } }];
        const own = structuredClone(schema);
        const extra = { nested: { deep: 1 } };
        const first = llm.generate({ model: 'm', messages });
        const pending = llm.generate(
            {
                model: 'm',
                messages: turns,
                stop,
                tools,
                response_format: { type: 'json_schema', json_schema: { name: 'word', schema: own } },
            },
            { extra },
        );
        turns.push({ role: 'user', content: 'injected' });
        stop.push('T');
        (tools[0] as LlmTool).function.name = 'g';
        own.required.push('zzz');
        extra.nested.deep = 2;
        await first;
        const result = await pending;
        expect(result).toMatchObject({ ok: true, data: { json: { kind: 'noun' }, settings: { stop: ['S'] } } });
        const sent = adapter.calls[1];
        expect(sent?.request.messages).toEqual([{ role: 'user', content: 'original' }]);
        expect(sent?.request.stop).toEqual(['S']);
        expect(sent?.request.tools?.[0]?.function.name).toBe('f');
        expect(sent?.extra).toEqual({ nested: { deep: 1 } });
    });

    it('isolates stored defaults and extras from adapter mutations', async () => {
        const adapter = createFakeProvider();
        const seen: unknown[] = [];
        adapter.reply((call) => {
            seen.push(structuredClone({ stop: call.request.stop, extra: call.extra }));
            (call.request.stop as string[]).push('CORRUPT');
            (call.extra.nested as { deep: number }).deep = 99;
            return answer('ok')(call);
        });
        const llm = createLlm({
            defaults: { stop: ['X'] },
            providers: { p: { adapter, extra: { nested: { deep: 1 } }, models: { m: 'm' } } },
        });
        const first = await llm.m('hi');
        await llm.m('hi');
        const sent = { stop: ['X'], extra: { nested: { deep: 1 } } };
        expect(seen).toEqual([sent, sent]);
        expect(first.ok && first.data.settings).toMatchObject(sent);
    });

    it('omits unset or disabled fields from reported settings', async () => {
        const adapter = createFakeProvider();
        const capabilities = { batch: { supported: true }, webSearch: { supported: true } };
        const llm = createLlm({ providers: { p: { adapter, models: { m: { model: 'id', capabilities } } } } });
        const result = await llm.m('hi', { web_search: false, batch: false, reasoning_effort: 'none' });
        expect(result.ok && result.data.settings).toEqual({ name: 'm', provider: 'p', model: 'id' });
        const searched = await llm.m('hi', { web_search: true });
        expect(searched.ok && searched.data.settings).toEqual({
            name: 'm',
            provider: 'p',
            model: 'id',
            web_search: true,
        });
    });

    it('invalid_json: includes partial output with reasoning and parsed json', async () => {
        const adapter = createFakeProvider('p', { json: 'native' });
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'id', capabilities: { json: true } } } } },
        });
        adapter.reply(reasoned('{"kind":"adjective"}'));
        const settings = { name: 'm', provider: 'p', model: 'id', jsonMode: 'native' };
        const answered = {
            text: '{"kind":"adjective"}',
            reasoning: 'thought',
            model: 'id',
            usage: { inputTokens: 10, outputTokens: 5, reasoningTokens: 3 },
            finishReason: 'stop',
            ms: 1,
        };
        const mismatch = await llm.m('hi', { response_format: format });
        expect(mismatch).toMatchObject({
            ok: false,
            error: {
                code: 'invalid_json',
                retryable: false,
                text: answered.text,
                model: 'id',
                usage: answered.usage,
                ms: 1,
                name: 'm',
                provider: 'p',
                settings,
                attempts: 1,
            },
        });
        expect(!mismatch.ok && mismatch.error.output).toEqual({
            ...answered,
            json: { kind: 'adjective' },
            name: 'm',
            provider: 'p',
        });
        adapter.reply(reasoned('not json'));
        const notJson = await llm.m('hi', { response_format: format });
        expect(notJson).toMatchObject({ ok: false, error: { code: 'invalid_json', settings, attempts: 1 } });
        expect(!notJson.ok && notJson.error.output).toEqual({
            ...answered,
            text: 'not json',
            name: 'm',
            provider: 'p',
        });
    });

    it('prompt mode: attaches single-attempt output to invalid_json error', async () => {
        const adapter = createFakeProvider('p', { json: 'prompt' });
        const llm = createLlm({
            limiterOptions: { maxRetries: 2 },
            providers: { p: { adapter, models: { m: { model: 'id', capabilities: { json: true } } } } },
        });
        adapter.reply(reasoned('{"kind":"y"}'), reasoned('{"kind":"noun"}'));
        const failed = await llm.m('hi', { response_format: format });
        expect(failed).toMatchObject({
            ok: false,
            error: {
                code: 'invalid_json',
                retryable: false,
                text: '{"kind":"y"}',
                output: { text: '{"kind":"y"}', json: { kind: 'y' }, reasoning: 'thought' },
                settings: { jsonMode: 'prompt' },
                attempts: 1,
            },
        });
        expect(adapter.calls).toHaveLength(1);
    });

    it('enriches truncated and blocked adapter output with model and provider names', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({ providers: { p: { adapter, models: { m: 'id' } } } });
        for (const code of ['truncated', 'blocked'] as const) {
            const output = {
                text: 'par',
                reasoning: 'thought',
                model: 'id-001',
                usage: { inputTokens: 1, outputTokens: 2, reasoningTokens: 3 },
                finishReason: code === 'truncated' ? 'length' : 'content_filter',
                ms: 4,
            };
            adapter.reply(() => ({
                ok: false,
                error: { code, message: code, retryable: false, text: 'par', model: 'id-001', output },
            }));
            expect(await llm.m('hi')).toEqual({
                ok: false,
                error: {
                    code,
                    message: code,
                    retryable: false,
                    text: 'par',
                    model: 'id-001',
                    name: 'm',
                    provider: 'p',
                    output: { ...output, name: 'm', provider: 'p' },
                    settings: { name: 'm', provider: 'p', model: 'id' },
                    attempts: 1,
                },
            });
        }
    });

    it('includes settings on transport/limiter errors but omits them on pre-prepare errors', async () => {
        const adapter = createFakeProvider();
        const llm = createLlm({
            providers: { p: { adapter, models: { m: { model: 'id', modelLimits: { tpm: 10 } } } } },
        });
        adapter.reply(failure({ code: 'http', status: 400, message: 'bad', retryable: false }));
        const http = await llm.m('hi', { max_tokens: 1 });
        expect(http).toMatchObject({ ok: false, error: { code: 'http', settings: { model: 'id' }, attempts: 1 } });
        expect(!http.ok && http.error.output).toBeUndefined();
        const limited = await llm.m('x'.repeat(1000));
        expect(limited).toMatchObject({
            ok: false,
            error: { code: 'limiter', settings: { model: 'id' }, attempts: 0 },
        });
        const invalid = await llm.m('hi', { temperature: 'hot' as unknown as number });
        const unsupported = await llm.m('hi', { tools: [tool] });
        for (const result of [invalid, unsupported]) {
            expect(result.ok).toBe(false);
            expect(!result.ok && result.error.settings).toBeUndefined();
            expect(!result.ok && result.error.attempts).toBeUndefined();
        }
    });
});
