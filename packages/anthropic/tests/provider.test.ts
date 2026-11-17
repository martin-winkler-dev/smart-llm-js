// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'bun:test';
import Anthropic from '@anthropic-ai/sdk';
import { createLlm, type LlmProviderRequest } from '@smart-llm/core';
import {
    ANTHROPIC_API_KEY_ENV,
    ANTHROPIC_SUPPORT,
    type AnthropicExtra,
    createAnthropicProvider,
    DEFAULT_MAX_TOKENS,
} from '../src/index.js';
import { toMessageParams } from '../src/request.js';
import { apiError, fakeFetch, hanging, json, message, queued, type Route, text } from './fake.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

const call = { signal: new AbortController().signal, extra: {} };

const request: LlmProviderRequest = { model: 'claude-x', messages: [{ role: 'user', content: 'hi' }] };

const usage = { inputTokens: 10, outputTokens: 20, reasoningTokens: null };

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Returns route answering with Anthropic message. */
function reply(content: unknown[] = [text('hello')], overrides: Partial<Anthropic.Message> = {}): Route {
    return json(message(content, overrides));
}

/** Creates provider with key and fake fetch answering through route. */
function provider(route: Route = reply(), options: Record<string, unknown> = {}) {
    const fake = fakeFetch(route);
    return { ...fake, provider: createAnthropicProvider({ apiKey: 'key', fetch: fake.fetch, ...options }) };
}

describe('createAnthropicProvider options', () => {
    it('exports the key env name, default max tokens and support', () => {
        expect(ANTHROPIC_API_KEY_ENV).toBe('ANTHROPIC_API_KEY');
        expect(DEFAULT_MAX_TOKENS).toBe(16_000);
        expect(ANTHROPIC_SUPPORT).toEqual({
            images: true,
            tools: true,
            webSearch: true,
            reasoning: true,
            batch: true,
            json: 'native',
        });
    });

    it('exposes anthropic name, ANTHROPIC_SUPPORT, and no embed', () => {
        const { provider: anthropic } = provider();
        expect(anthropic.name).toBe('anthropic');
        expect(anthropic.supports).toEqual(ANTHROPIC_SUPPORT);
        expect(anthropic.embed).toBeUndefined();
        expect(anthropic.maxEmbedInputs).toBeUndefined();
        expect(typeof anthropic.listModels).toBe('function');
    });

    it('throws a TypeError without an API key in options or env', () => {
        const message = 'Invalid value for option "apiKey": set it or env ANTHROPIC_API_KEY';
        expect(() => createAnthropicProvider({ env: {} })).toThrow(TypeError);
        expect(() => createAnthropicProvider({ env: {} })).toThrow(message);
        expect(() => createAnthropicProvider({ apiKey: '', env: {} })).toThrow(message);
        expect(() => createAnthropicProvider({ env: { ANTHROPIC_API_KEY: '' } })).toThrow(message);
    });

    it('reads the API key from env, and an apiKey option wins over it', async () => {
        const fake = fakeFetch(reply());
        await createAnthropicProvider({ env: { ANTHROPIC_API_KEY: 'env-key' }, fetch: fake.fetch }).generate(
            request,
            call,
        );
        await createAnthropicProvider({
            apiKey: 'own-key',
            env: { ANTHROPIC_API_KEY: 'env-key' },
            fetch: fake.fetch,
        }).generate(request, call);
        expect(fake.sent.map((sent) => sent.headers.get('authorization'))).toEqual([
            'Bearer env-key',
            'Bearer own-key',
        ]);
    });

    it('rejects invalid options and unknown keys with TypeError', () => {
        const cases: [Record<string, unknown>, RegExp][] = [
            [{ timeoutMs: 0 }, /"timeoutMs"/],
            [{ timeoutMs: -1 }, /"timeoutMs"/],
            [{ timeoutMs: 1.5 }, /"timeoutMs"/],
            [{ timeoutMs: '1000' }, /"timeoutMs"/],
            [{ batchWaitSec: 0 }, /"batchWaitSec"/],
            [{ batchWaitSec: 601 }, /"batchWaitSec"/],
            [{ batchWaitSec: 1.5 }, /"batchWaitSec"/],
            [{ defaultMaxTokens: 0 }, /"defaultMaxTokens"/],
            [{ defaultMaxTokens: 2.5 }, /"defaultMaxTokens"/],
            [{ maxRetries: 2 }, /"maxRetries"/],
            [{ env: null }, /"env"/],
            [{ env: 'x' }, /"env"/],
            [{ env: [] }, /"env"/],
            [{ apiKey: null }, /"apiKey": a string/],
            [{ apiKey: 42 }, /"apiKey": a string/],
            [{ baseUrl: '' }, /"baseUrl"/],
            [{ baseUrl: 'not a url' }, /"baseUrl"/],
            [{ baseUrl: 42 }, /"baseUrl"/],
            [{ baseUrl: 'localhost:8080' }, /"baseUrl": an absolute http\(s\) URL/],
            [{ baseUrl: 'ftp://example.com' }, /"baseUrl"/],
            [{ headers: 'x' }, /"headers"/],
            [{ headers: { 'anthropic-beta': 1 } }, /"headers"/],
            [{ fetch: 'nope' }, /"fetch"/],
        ];
        for (const [options, pattern] of cases) {
            expect(() => createAnthropicProvider({ apiKey: 'key', ...options } as never)).toThrow(TypeError);
            expect(() => createAnthropicProvider({ apiKey: 'key', ...options } as never)).toThrow(pattern);
        }
        for (const options of [
            { batchWaitSec: 1 },
            { batchWaitSec: 600 },
            { timeoutMs: 1 },
            { defaultMaxTokens: 1 },
            { baseUrl: undefined, headers: undefined, fetch: undefined, env: undefined },
            { baseUrl: 'http://localhost:8080', headers: {}, env: {} },
        ]) {
            expect(() => createAnthropicProvider({ apiKey: 'key', ...options } as never)).not.toThrow();
        }
    });

    it('accepts env object with non-plain prototype chain', async () => {
        const env = Object.create(Object.create({ inherited: 'x' })) as Record<string, string>;
        env[ANTHROPIC_API_KEY_ENV] = 'proto-key';
        const fake = fakeFetch(reply());
        await createAnthropicProvider({ env, fetch: fake.fetch }).generate(request, call);
        expect(fake.sent[0]?.headers.get('authorization')).toBe('Bearer proto-key');
    });

    it('does not fall back to env key when apiKey is null', () => {
        const env = { ANTHROPIC_API_KEY: 'from-env' };
        expect(() => createAnthropicProvider({ apiKey: null, env } as never)).toThrow(TypeError);
    });
});

describe('createAnthropicProvider generate', () => {
    it('posts to /v1/messages with key, API version, custom headers and base URL', async () => {
        const fake = provider(reply(), { baseUrl: 'https://proxy.example.com', headers: { 'anthropic-beta': 'b-1' } });
        await fake.provider.generate(request, call);
        const sent = fake.sent[0];
        expect(fake.sent).toHaveLength(1);
        expect(sent?.method).toBe('POST');
        expect(sent?.url).toBe('https://proxy.example.com/v1/messages');
        expect(sent?.headers.get('authorization')).toBe('Bearer key');
        expect(sent?.headers.get('anthropic-version')).toBe('2026-10-21');
        expect(sent?.headers.get('anthropic-beta')).toBe('b-1');
    });

    it('uses default API URL, per-request timeout, and zero retries', async () => {
        const fake = provider();
        await fake.provider.generate(request, call);
        expect(fake.sent[0]?.url).toBe('https://api.anthropic.com/v1/messages');
        expect(fake.sent[0]?.headers.get('x-stainless-timeout')).toBe('600');
        expect(fake.sent[0]?.headers.get('x-stainless-retry-count')).toBe('0');
        const short = provider(reply(), { timeoutMs: 30_000 });
        await short.provider.generate(request, call);
        expect(short.sent[0]?.headers.get('x-stainless-timeout')).toBe('30');
    });

    it('sends the params of toMessageParams with the merged extra and reads the message', async () => {
        const fake = provider(
            reply([{ type: 'thinking', thinking: 'thought', signature: 's' }, text('{"a":1}')], {
                model: 'claude-x-001',
            }),
        );
        const full: LlmProviderRequest = {
            model: 'claude-x',
            messages: [
                { role: 'system', content: 'Be brief.' },
                { role: 'user', content: 'hi' },
            ],
            max_tokens: 4000,
            temperature: 0.5,
            stop: 'END',
            reasoning_effort: 'high',
            tools: [{ type: 'function', function: { name: 'now' } }],
            tool_choice: 'auto',
            response_format: { type: 'json_schema', json_schema: { name: 'a', schema: { type: 'object' } } },
        };
        const extra: AnthropicExtra = { thinkingDisplay: 'omitted', params: { metadata: { user_id: 'u' } } };
        const result = await fake.provider.generate(full, { ...call, extra });
        const mapped = toMessageParams(full, extra, DEFAULT_MAX_TOKENS);
        expect(mapped.ok).toBe(true);
        expect(fake.sent[0]?.body).toEqual(mapped.ok ? mapped.data : undefined);
        expect(fake.sent[0]?.body).toEqual({
            model: 'claude-x',
            max_tokens: 4000,
            system: [{ type: 'text', text: 'Be brief.' }],
            messages: [{ role: 'user', content: 'hi' }],
            stop_sequences: ['END'],
            tools: [{ name: 'now', input_schema: { type: 'object', properties: {} } }],
            tool_choice: { type: 'auto' },
            thinking: { type: 'adaptive', display: 'omitted' },
            output_config: {
                effort: 'high',
                format: {
                    type: 'json_schema',
                    schema: { type: 'object', properties: {}, additionalProperties: false },
                },
            },
            metadata: { user_id: 'u' },
        });
        expect(result).toEqual({
            ok: true,
            data: {
                text: '{"a":1}',
                reasoning: 'thought',
                model: 'claude-x-001',
                usage,
                finishReason: 'end_turn',
                ms: expect.any(Number),
            },
        });
    });

    it('uses defaultMaxTokens without max_tokens in the request', async () => {
        const fake = provider(reply(), { defaultMaxTokens: 2048 });
        await fake.provider.generate(request, call);
        expect(fake.sent[0]?.body).toEqual({
            model: 'claude-x',
            max_tokens: 2048,
            messages: [{ role: 'user', content: 'hi' }],
        });
    });

    it('sends a large max_tokens without the SDK streaming guard', async () => {
        const fake = provider();
        const result = await fake.provider.generate(
            { ...request, model: 'claude-opus-5-5', max_tokens: 128_000 },
            call,
        );
        expect(result.ok).toBe(true);
        expect(fake.sent).toHaveLength(1);
    });

    it('returns the mapping error without a request', async () => {
        const fake = provider();
        const result = await fake.provider.generate(
            {
                ...request,
                messages: [
                    { role: 'user', content: [{ type: 'image_url', image_url: { url: 'data:image/bmp;base64,AA' } }] },
                ],
            },
            call,
        );
        expect(result).toEqual({
            ok: false,
            error: {
                code: 'invalid_request',
                message: 'Anthropic accepts JPEG, PNG, GIF and WebP images, not image/bmp',
                retryable: false,
            },
        });
        expect(fake.sent).toHaveLength(0);
    });

    it('returns failed stop reasons with the answer as output', async () => {
        const fake = provider(reply([text('par')], { stop_reason: 'max_tokens' }));
        expect(await fake.provider.generate(request, call)).toMatchObject({
            ok: false,
            error: {
                code: 'truncated',
                retryable: false,
                text: 'par',
                output: { text: 'par', finishReason: 'max_tokens' },
            },
        });
    });

    it('maps HTTP errors with one fetch call each, without SDK retries', async () => {
        const cases: [Route, Record<string, unknown>][] = [
            [
                apiError(400, 'invalid_request_error', 'max_tokens: too large'),
                { code: 'http', status: 400, retryable: false, message: 'HTTP 400: max_tokens: too large' },
            ],
            [
                apiError(401, 'authentication_error', 'invalid x-api-key'),
                { code: 'http', status: 401, retryable: false, message: 'HTTP 401: invalid x-api-key' },
            ],
            [
                apiError(408, 'timeout_error', 'took too long'),
                { code: 'http', status: 408, retryable: true, message: 'HTTP 408: took too long' },
            ],
            [
                apiError(409, 'conflict_error', 'conflict'),
                { code: 'http', status: 409, retryable: true, message: 'HTTP 409: conflict' },
            ],
            [
                apiError(429, 'rate_limit_error', 'slow down', { 'retry-after': '3' }),
                {
                    code: 'rate_limited',
                    status: 429,
                    retryable: true,
                    retryAfterMs: 3000,
                    message: 'HTTP 429: slow down',
                },
            ],
            [
                apiError(429, 'rate_limit_error', 'slow down', { 'retry-after-ms': '250', 'retry-after': '3' }),
                {
                    code: 'rate_limited',
                    status: 429,
                    retryable: true,
                    retryAfterMs: 250,
                    message: 'HTTP 429: slow down',
                },
            ],
            [
                apiError(500, 'api_error', 'Internal server error'),
                { code: 'http', status: 500, retryable: true, message: 'HTTP 500: Internal server error' },
            ],
            [
                apiError(529, 'overloaded_error', 'Overloaded'),
                { code: 'http', status: 529, retryable: true, message: 'HTTP 529: Overloaded' },
            ],
        ];
        for (const [route, error] of cases) {
            const fake = provider(route);
            expect(await fake.provider.generate(request, call)).toEqual({ ok: false, error } as never);
            expect(fake.sent).toHaveLength(1);
        }
    });

    it('maps connection failures to retryable network and timeouts to retryable timeout', async () => {
        const failing = provider(() => {
            throw new TypeError('fetch failed');
        });
        expect(await failing.provider.generate(request, call)).toMatchObject({
            ok: false,
            error: { code: 'network', retryable: true },
        });
        expect(failing.sent).toHaveLength(1);
        const slow = provider(hanging(), { timeoutMs: 5 });
        expect(await slow.provider.generate(request, call)).toMatchObject({
            ok: false,
            error: { code: 'timeout', retryable: true },
        });
        expect(slow.sent).toHaveLength(1);
    });

    it('reports an abort before and during the request as aborted', async () => {
        const before = new AbortController();
        before.abort();
        const fake = provider(hanging());
        expect(await fake.provider.generate(request, { signal: before.signal, extra: {} })).toMatchObject({
            ok: false,
            error: { code: 'aborted', retryable: false },
        });
        const during = new AbortController();
        const pending = fake.provider.generate(request, { signal: during.signal, extra: {} });
        setTimeout(() => during.abort(), 5);
        expect(await pending).toMatchObject({ ok: false, error: { code: 'aborted', retryable: false } });
    });

    it('uses an existing Anthropic client as is, still without retries', async () => {
        const fake = fakeFetch(queued(apiError(500, 'api_error', 'Internal server error'), reply()));
        const client = new Anthropic({ apiKey: 'own-key', fetch: fake.fetch });
        const anthropic = createAnthropicProvider(client);
        expect(await anthropic.generate(request, call)).toMatchObject({
            ok: false,
            error: { code: 'http', status: 500 },
        });
        expect(fake.sent).toHaveLength(1);
        expect(fake.sent[0]?.headers.get('authorization')).toBe('Bearer own-key');
        expect(fake.sent[0]?.headers.get('x-stainless-retry-count')).toBe('0');
        expect(await anthropic.generate(request, call)).toMatchObject({ ok: true, data: { text: 'hello' } });
    });

    it('sends the timeout of an existing Anthropic client per request', async () => {
        const fake = fakeFetch(reply());
        const anthropic = createAnthropicProvider(new Anthropic({ apiKey: 'key', fetch: fake.fetch, timeout: 5000 }));
        await anthropic.generate(request, call);
        expect(fake.sent[0]?.headers.get('x-stainless-timeout')).toBe('5');
        const large = { ...request, max_tokens: 64_000 };
        expect(await anthropic.generate(large, call)).toMatchObject({ ok: true });
    });

    it('uses default timeout for client without timeout', async () => {
        const fake = fakeFetch(reply());
        const anthropic = createAnthropicProvider(new Anthropic({ apiKey: 'key', fetch: fake.fetch }));
        expect(await anthropic.generate({ ...request, max_tokens: 64_000 }, call)).toMatchObject({ ok: true });
        expect(fake.sent[0]?.headers.get('x-stainless-timeout')).toBe('600');
    });

    it('uses default timeout for duck-typed client without timeout', async () => {
        const fake = fakeFetch(reply());
        const real = new Anthropic({ apiKey: 'key', fetch: fake.fetch });
        const wrapper = { messages: real.messages, models: real.models } as unknown as Anthropic;
        const anthropic = createAnthropicProvider(wrapper);
        expect(await anthropic.generate({ ...request, max_tokens: 64_000 }, call)).toMatchObject({ ok: true });
        expect(fake.sent[0]?.headers.get('x-stainless-timeout')).toBe('600');
    });

    it('rejects empty, 204, HTML, and non-message 2xx responses as retryable unexpected', async () => {
        const error = { code: 'unexpected', message: 'The response is not an Anthropic message', retryable: true };
        const routes: Route[] = [
            () => new Response(null, { status: 204 }),
            () => new Response('', { headers: { 'content-type': 'application/json', 'content-length': '0' } }),
            () => new Response('<html>proxy</html>', { headers: { 'content-type': 'text/html' } }),
            json({}),
            json({ content: 'x', stop_reason: 'end_turn' }),
        ];
        for (const route of routes) {
            expect(await provider(route).provider.generate(request, call)).toEqual({ ok: false, error } as never);
        }
    });

    it('works through createLlm with a native JSON schema', async () => {
        const fake = fakeFetch(reply([text('{"a":1}')]));
        const llm = createLlm({
            providers: {
                anthropic: {
                    adapter: createAnthropicProvider({ apiKey: 'key', fetch: fake.fetch }),
                    models: { claude: { model: 'claude-x', capabilities: { json: true } } },
                },
            },
        });
        const schema = { type: 'object', properties: { a: { type: 'number' } }, required: ['a'] };
        const result = await llm.claude('hi', {
            response_format: { type: 'json_schema', json_schema: { name: 'a', schema } },
        });
        expect(result).toMatchObject({ ok: true, data: { json: { a: 1 }, provider: 'anthropic', attempts: 1 } });
        expect((fake.sent[0]?.body as Record<string, unknown> | undefined)?.output_config).toEqual({
            format: { type: 'json_schema', schema: { ...schema, additionalProperties: false } },
        });
    });
});

describe('createAnthropicProvider listModels', () => {
    it('pages through /v1/models with limit 1000 and maps the model info', async () => {
        function model(id: string, extra: Record<string, unknown>) {
            return {
                type: 'model',
                id,
                display_name: id.toUpperCase(),
                created_at: '2026-01-01T00:00:00Z',
                max_input_tokens: null,
                max_tokens: null,
                ...extra,
            };
        }
        const fake = provider(
            queued(
                json({
                    data: [model('a', { max_input_tokens: 1_000_000, max_tokens: 128_000 }), model('b', {})],
                    has_more: true,
                    first_id: 'a',
                    last_id: 'b',
                }),
                json({ data: [model('c', { max_tokens: 64_000 })], has_more: false, first_id: 'c', last_id: 'c' }),
            ),
        );
        expect(await fake.provider.listModels?.()).toEqual({
            ok: true,
            data: [
                { id: 'a', displayName: 'A', inputTokenLimit: 1_000_000, outputTokenLimit: 128_000 },
                { id: 'b', displayName: 'B' },
                { id: 'c', displayName: 'C', outputTokenLimit: 64_000 },
            ],
        });
        expect(fake.sent.map((sent) => [sent.method, sent.path])).toEqual([
            ['GET', '/v1/models'],
            ['GET', '/v1/models'],
        ]);
        const queries = fake.sent.map((sent) => Object.fromEntries(new URL(sent.url).searchParams));
        expect(queries).toEqual([{ limit: '1000' }, { limit: '1000', after_id: 'b' }]);
        expect(fake.sent[0]?.headers.get('x-stainless-retry-count')).toBe('0');
    });

    it('returns errors without output and an abort as aborted', async () => {
        const fake = provider(apiError(401, 'authentication_error', 'invalid x-api-key'));
        expect(await fake.provider.listModels?.()).toEqual({
            ok: false,
            error: { code: 'http', status: 401, retryable: false, message: 'HTTP 401: invalid x-api-key' },
        });
        expect(fake.sent).toHaveLength(1);
        const controller = new AbortController();
        controller.abort();
        expect(await provider(hanging()).provider.listModels?.({ signal: controller.signal })).toMatchObject({
            ok: false,
            error: { code: 'aborted', retryable: false },
        });
    });
});

describe('createAnthropicProvider extra', () => {
    const { provider: anthropic } = provider();

    it('accepts every AnthropicExtra key with valid values', () => {
        const extra = {
            params: { metadata: { user_id: 'u' }, thinking: null },
            thinkingBudgets: { minimal: 1024, low: 2048, medium: 4096, high: 8192, xhigh: 16_384, max: 32_768 },
            thinkingDisplay: 'omitted',
            thinkingOff: 'between_tools',
            webSearchTool: { max_uses: 3 },
        };
        expect(() => anthropic.validateExtra?.(extra, 'extra')).not.toThrow();
        expect(() => anthropic.validateExtra?.({}, 'extra')).not.toThrow();
        for (const thinkingOff of ['disabled', 'between_tools', 'omit']) {
            expect(() => anthropic.validateExtra?.({ thinkingOff }, 'extra')).not.toThrow();
        }
        expect(() => anthropic.validateExtra?.({ thinkingDisplay: 'summarized' }, 'extra')).not.toThrow();
    });

    it('rejects unknown keys and invalid values with TypeErrors naming the option', () => {
        expect(() => anthropic.validateExtra?.({ config: {} }, 'providers.a.extra')).toThrow(
            'Option "providers.a.extra.config" is not supported by the Anthropic provider (only params, thinkingBudgets, thinkingDisplay, thinkingOff, webSearchTool)',
        );
        expect(() => anthropic.validateExtra?.({ thinkingBudgets: { high: 1000 } }, 'extra')).toThrow(
            'Invalid value for option "extra.thinkingBudgets.high": an integer of at least 1024',
        );
        const cases: [Record<string, unknown>, RegExp][] = [
            [{ params: [] }, /"extra\.params/],
            [{ params: null }, /"extra\.params/],
            [{ params: 'x' }, /"extra\.params/],
            [{ params: { model: 'm' } }, /"extra\.params/],
            [{ params: { messages: [] } }, /"extra\.params/],
            [{ params: { stream: true } }, /"extra\.params/],
            [{ thinkingBudgets: [] }, /"extra\.thinkingBudgets/],
            [{ thinkingBudgets: { high: 2048.5 } }, /"extra\.thinkingBudgets\.high"/],
            [{ thinkingBudgets: { high: '2048' } }, /"extra\.thinkingBudgets\.high"/],
            [{ thinkingBudgets: { none: 2048 } }, /"extra\.thinkingBudgets\.none"/],
            [{ thinkingDisplay: 'full' }, /"extra\.thinkingDisplay"/],
            [{ thinkingOff: 'off' }, /"extra\.thinkingOff"/],
            [{ webSearchTool: [] }, /"extra\.webSearchTool"/],
            [{ webSearchTool: null }, /"extra\.webSearchTool"/],
        ];
        for (const [extra, pattern] of cases) {
            expect(() => anthropic.validateExtra?.(extra, 'extra')).toThrow(TypeError);
            expect(() => anthropic.validateExtra?.(extra, 'extra')).toThrow(pattern);
        }
    });

    it('merges layers with later ones winning, deep-merging params, thinkingBudgets and webSearchTool', () => {
        const merged = anthropic.mergeExtra?.([
            {
                params: { metadata: { user_id: 'a' }, output_config: { effort: 'low' }, stop_sequences: ['a', 'b'] },
                thinkingBudgets: { low: 2048 },
                thinkingDisplay: 'omitted',
                webSearchTool: { max_uses: 2, allowed_domains: ['a.com'] },
            },
            {
                params: { output_config: { effort: 'high' }, stop_sequences: ['x'], thinking: { type: 'adaptive' } },
                thinkingBudgets: { high: 8192 },
                thinkingDisplay: 'summarized',
                thinkingOff: 'omit',
                webSearchTool: { allowed_domains: ['b.com'] },
            },
            { params: { thinking: null } },
        ]);
        expect(merged).toEqual({
            params: {
                metadata: { user_id: 'a' },
                output_config: { effort: 'high' },
                stop_sequences: ['x'],
                thinking: null,
            },
            thinkingBudgets: { low: 2048, high: 8192 },
            thinkingDisplay: 'summarized',
            thinkingOff: 'omit',
            webSearchTool: { max_uses: 2, allowed_domains: ['b.com'] },
        });
        expect(anthropic.mergeExtra?.([{}, {}, {}])).toEqual({});
    });

    it('applies the merged extra through createLlm', async () => {
        const fake = fakeFetch(reply());
        const llm = createLlm({
            providers: {
                anthropic: {
                    adapter: createAnthropicProvider({ apiKey: 'key', fetch: fake.fetch }),
                    extra: { params: { metadata: { user_id: 'u' } }, thinkingDisplay: 'omitted' },
                    models: {
                        claude: {
                            model: 'claude-x',
                            extra: { thinkingOff: 'between_tools' },
                            capabilities: { reasoning: { supported: true } },
                        },
                    },
                },
            },
        });
        const result = await llm.claude('hi', {
            reasoning_effort: 'none',
            extra: { params: { service_tier: 'auto' } },
        });
        expect(result.ok).toBe(true);
        expect(fake.sent[0]?.body).toEqual({
            model: 'claude-x',
            max_tokens: 16_000,
            messages: [{ role: 'user', content: 'hi' }],
            thinking: { type: 'between_tools' },
            metadata: { user_id: 'u' },
            service_tier: 'auto',
        });
    });
});
