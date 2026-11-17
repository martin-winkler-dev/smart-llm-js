// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'bun:test';
import type Anthropic from '@anthropic-ai/sdk';
import {
    AnthropicError,
    APIConnectionError,
    APIConnectionTimeoutError,
    APIError,
    APIUserAbortError,
} from '@anthropic-ai/sdk';
import { readMessage, toProviderError } from '../src/response.js';
import { message, text, USAGE } from './fake.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

const usage = { inputTokens: 10, outputTokens: 20, reasoningTokens: null };

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Creates SDK status error like the client does for an error response. */
function statusError(status: number, body: unknown, headers: Record<string, string> = {}): APIError {
    return APIError.generate(status, body as object, undefined, new Headers(headers));
}

/** Creates Anthropic error body `{type:'error', error:{type, message}}`. */
function errorBody(type: string, value: string) {
    return { type: 'error', error: { type, message: value }, request_id: 'req_1' };
}

describe('readMessage', () => {
    it('concatenates text blocks, joins non-empty thinking texts and adds cache tokens to the input tokens', () => {
        const read = readMessage(
            message(
                [
                    { type: 'thinking', thinking: 'plan', signature: 's1' },
                    { type: 'thinking', thinking: '', signature: 's2' },
                    { type: 'redacted_thinking', data: 'R' },
                    { type: 'thinking', thinking: 'more', signature: 's3' },
                    text('Hel'),
                    text('lo'),
                ],
                {
                    usage: {
                        ...USAGE,
                        input_tokens: 10,
                        cache_creation_input_tokens: 5,
                        cache_read_input_tokens: 20,
                        output_tokens: 50,
                        output_tokens_details: { thinking_tokens: 30 },
                    },
                },
            ),
            'claude-requested',
            12.4,
        );
        expect(read).toEqual({
            ok: true,
            data: {
                text: 'Hello',
                reasoning: 'plan\n\nmore',
                model: 'claude-served',
                usage: { inputTokens: 35, outputTokens: 20, reasoningTokens: 30 },
                finishReason: 'end_turn',
                ms: 12,
            },
        });
    });

    it('defaults reasoningTokens to null and falls back to requested model', () => {
        expect(readMessage(message([text('hi')], { model: '' }), 'claude-requested', 12.6)).toEqual({
            ok: true,
            data: { text: 'hi', model: 'claude-requested', usage, finishReason: 'end_turn', ms: 13 },
        });
        expect(readMessage(message([]), 'm', 0)).toEqual({
            ok: true,
            data: { text: '', model: 'claude-served', usage, finishReason: 'end_turn', ms: 0 },
        });
    });

    it('clamps output tokens to 0 when thinking tokens exceed output tokens', () => {
        const read = readMessage(
            message([text('hi')], {
                usage: { ...USAGE, output_tokens: 10, output_tokens_details: { thinking_tokens: 30 } },
            }),
            'm',
            0,
        );
        expect(read.ok ? read.data.usage : undefined).toEqual({
            inputTokens: 10,
            outputTokens: 0,
            reasoningTokens: 30,
        });
    });

    it('reads tool calls and puts all raw content blocks verbatim and in order on the first call', () => {
        const content = [
            { type: 'thinking', thinking: 'A', signature: 'sA' },
            text('Checking'),
            {
                type: 'tool_use',
                id: 't1',
                name: 'get_weather',
                input: { city: 'Graz' },
                caller: { type: 'direct' },
            },
            { type: 'redacted_thinking', data: 'R' },
            { type: 'tool_use', id: 't2', name: 'now', input: {}, caller: { type: 'direct' } },
            { type: 'tool_use', id: 't3', name: 'later', caller: { type: 'direct' } },
        ];
        const read = readMessage(message(content, { stop_reason: 'tool_use' }), 'm', 1);
        expect(read).toEqual({
            ok: true,
            data: {
                text: 'Checking',
                reasoning: 'A',
                toolCalls: [
                    {
                        id: 't1',
                        type: 'function',
                        function: { name: 'get_weather', arguments: '{"city":"Graz"}' },
                        extra_content: { anthropic: { content } },
                    },
                    { id: 't2', type: 'function', function: { name: 'now', arguments: '{}' } },
                    { id: 't3', type: 'function', function: { name: 'later', arguments: '{}' } },
                ],
                model: 'claude-served',
                usage,
                finishReason: 'tool_use',
                ms: 1,
            },
        });
    });

    it('attaches raw content on first tool call even without thinking, omits without tool calls', () => {
        const content = [{ type: 'tool_use', id: 't1', name: 'f', input: { a: 1 }, caller: { type: 'direct' } }];
        const read = readMessage(message(content, { stop_reason: 'tool_use' }), 'm', 1);
        expect(read.ok && read.data.toolCalls).toEqual([
            {
                id: 't1',
                type: 'function',
                function: { name: 'f', arguments: '{"a":1}' },
                extra_content: { anthropic: { content } },
            },
        ]);
        const plain = readMessage(message([{ type: 'thinking', thinking: 'A', signature: 's' }, text('a')]), 'm', 1);
        expect(plain.ok && plain.data).not.toHaveProperty('toolCalls');
    });

    it('reads sources from web search citations, deduped by url, ignoring server tool use', () => {
        const location = { type: 'web_search_result_location', cited_text: 'x', encrypted_index: 'e' };
        const read = readMessage(
            message([
                {
                    type: 'server_tool_use',
                    id: 's1',
                    name: 'web_search',
                    input: { query: 'q' },
                    caller: { type: 'direct' },
                },
                {
                    type: 'web_search_tool_result',
                    tool_use_id: 's1',
                    caller: { type: 'direct' },
                    content: [
                        {
                            type: 'web_search_result',
                            url: 'https://b.example',
                            title: 'B',
                            encrypted_content: 'e',
                            page_age: null,
                        },
                    ],
                },
                text('X', [
                    { ...location, url: 'https://a.example', title: 'A' },
                    { ...location, url: 'https://c.example', title: null },
                ]),
                text('Y', [
                    { ...location, url: 'https://a.example', title: 'A again' },
                    {
                        type: 'char_location',
                        cited_text: 'y',
                        document_index: 0,
                        document_title: null,
                        start_char_index: 0,
                        end_char_index: 1,
                        file_id: null,
                    },
                ]),
            ]),
            'm',
            1,
        );
        expect(read).toEqual({
            ok: true,
            data: {
                text: 'XY',
                sources: [{ url: 'https://a.example', title: 'A' }, { url: 'https://c.example' }],
                model: 'claude-served',
                usage,
                finishReason: 'end_turn',
                ms: 1,
            },
        });
    });

    it('falls back to web search results without citations and ignores search errors', () => {
        function result(url: string, title: string) {
            return {
                type: 'web_search_result',
                url,
                title,
                encrypted_content: 'e',
                page_age: null,
            };
        }
        const read = readMessage(
            message([
                {
                    type: 'web_search_tool_result',
                    tool_use_id: 's1',
                    caller: { type: 'direct' },
                    content: [result('https://a.example', 'A'), result('https://b.example', 'B')],
                },
                {
                    type: 'web_search_tool_result',
                    tool_use_id: 's2',
                    caller: { type: 'direct' },
                    content: { type: 'web_search_tool_result_error', error_code: 'unavailable' },
                },
                text('X'),
            ]),
            'm',
            1,
        );
        expect(read.ok && read.data.sources).toEqual([
            { url: 'https://a.example', title: 'A' },
            { url: 'https://b.example', title: 'B' },
        ]);
        const errorOnly = readMessage(
            message([
                {
                    type: 'web_search_tool_result',
                    tool_use_id: 's2',
                    caller: { type: 'direct' },
                    content: { type: 'web_search_tool_result_error', error_code: 'unavailable' },
                },
                text('X'),
            ]),
            'm',
            1,
        );
        expect(errorOnly.ok && errorOnly.data).not.toHaveProperty('sources');
    });

    it('rejects non-message responses without content array as retryable unexpected', () => {
        const error = { code: 'unexpected', message: 'The response is not an Anthropic message', retryable: true };
        const responses = [
            null,
            undefined,
            '<html>proxy</html>',
            '',
            [],
            {},
            { content: 'x', stop_reason: 'end_turn' },
            { content: [null], stop_reason: 'end_turn' },
            { content: [text('a'), 'b'], stop_reason: 'end_turn' },
        ];
        for (const response of responses) {
            expect(readMessage(response as never, 'm', 1)).toEqual({ ok: false, error } as never);
        }
    });

    it('rejects non-object tool_use input as retryable unexpected with output', () => {
        for (const input of ['abc', [1, 2], 42]) {
            const content = [text('a'), { type: 'tool_use', id: 't1', name: 'f', input, caller: { type: 'direct' } }];
            expect(readMessage(message(content, { stop_reason: 'tool_use' }), 'm', 1)).toMatchObject({
                ok: false,
                error: {
                    code: 'unexpected',
                    message: 'The arguments of tool call f are not a JSON object',
                    retryable: true,
                    text: 'a',
                    output: { text: 'a', finishReason: 'tool_use' },
                },
            });
        }
        const truncated = [{ type: 'tool_use', id: 't1', name: 'f', input: 'abc', caller: { type: 'direct' } }];
        expect(readMessage(message(truncated, { stop_reason: 'max_tokens' }), 'm', 1)).toMatchObject({
            ok: false,
            error: { code: 'truncated' },
        });
    });
});

describe('readMessage stop reasons', () => {
    it('succeeds for end_turn, stop_sequence and null', () => {
        for (const reason of ['end_turn', 'stop_sequence', null] as const) {
            expect(readMessage(message([text('a')], { stop_reason: reason }), 'm', 1)).toEqual({
                ok: true,
                data: { text: 'a', model: 'claude-served', usage, finishReason: reason, ms: 1 },
            });
        }
    });

    it('carries the read answer with reasoning as output on failed stop reasons', () => {
        const read = readMessage(
            message([{ type: 'thinking', thinking: 'thought', signature: 's' }, text('par')], {
                stop_reason: 'max_tokens',
            }),
            'm',
            7,
        );
        expect(read).toEqual({
            ok: false,
            error: {
                code: 'truncated',
                message: 'The answer hit the output token limit',
                retryable: false,
                text: 'par',
                model: 'claude-served',
                usage,
                ms: 7,
                output: {
                    text: 'par',
                    reasoning: 'thought',
                    model: 'claude-served',
                    usage,
                    finishReason: 'max_tokens',
                    ms: 7,
                },
            },
        });
    });

    it('maps each failing stop reason to code, message and retryable', () => {
        function refusal(category: string | null, explanation: string | null): never {
            return { type: 'refusal', category, explanation } as never;
        }
        const cases: [Partial<Anthropic.Message>, Record<string, unknown>][] = [
            [
                { stop_reason: 'tool_use' },
                { code: 'unexpected', message: 'Stop reason tool_use without a tool call', retryable: true },
            ],
            [
                { stop_reason: 'model_context_window_exceeded' },
                { code: 'truncated', message: 'The conversation exceeds the context window', retryable: false },
            ],
            [
                { stop_reason: 'refusal', stop_details: refusal('cyber', 'Exploit code') },
                { code: 'blocked', message: 'The model refused the request (cyber): Exploit code', retryable: false },
            ],
            [
                { stop_reason: 'refusal', stop_details: refusal('bio', null) },
                { code: 'blocked', message: 'The model refused the request (bio)', retryable: false },
            ],
            [
                { stop_reason: 'refusal', stop_details: refusal(null, 'No reason') },
                { code: 'blocked', message: 'The model refused the request: No reason', retryable: false },
            ],
            [
                { stop_reason: 'refusal' },
                { code: 'blocked', message: 'The model refused the request', retryable: false },
            ],
            [
                { stop_reason: 'pause_turn' },
                {
                    code: 'unexpected',
                    message: 'The turn paused (pause_turn): continuing a paused server tool turn is not supported',
                    retryable: false,
                },
            ],
            [
                { stop_reason: 'compaction' as never },
                { code: 'unexpected', message: 'Stop reason compaction', retryable: false },
            ],
        ];
        for (const [overrides, error] of cases) {
            const read = readMessage(message([text('a')], overrides), 'm', 1);
            expect(read).toEqual({
                ok: false,
                error: {
                    ...error,
                    text: 'a',
                    model: 'claude-served',
                    usage,
                    ms: 1,
                    output: {
                        text: 'a',
                        model: 'claude-served',
                        usage,
                        finishReason: overrides.stop_reason ?? null,
                        ms: 1,
                    },
                },
            } as never);
        }
    });
});

describe('toProviderError', () => {
    it('maps aborts, also of other errors with an aborted signal', () => {
        expect(toProviderError(new APIUserAbortError())).toMatchObject({ code: 'aborted', retryable: false });
        const controller = new AbortController();
        controller.abort();
        expect(toProviderError(new Error('x'), controller.signal)).toMatchObject({
            code: 'aborted',
            retryable: false,
        });
        expect(toProviderError(new APIConnectionError({}), controller.signal)).toMatchObject({ code: 'aborted' });
    });

    it('maps connection timeouts to timeout and other connection errors to network, both retryable', () => {
        expect(toProviderError(new APIConnectionTimeoutError())).toMatchObject({ code: 'timeout', retryable: true });
        expect(toProviderError(new APIConnectionError({ message: 'Connection error.' }))).toMatchObject({
            code: 'network',
            retryable: true,
        });
        expect(toProviderError(new APIConnectionError({}), new AbortController().signal)).toMatchObject({
            code: 'network',
        });
    });

    it('maps HTTP statuses with the API message', () => {
        const cases: [number, string, string, Record<string, unknown>][] = [
            [400, 'invalid_request_error', 'max_tokens: too large', { code: 'http', retryable: false }],
            [401, 'authentication_error', 'invalid x-api-key', { code: 'http', retryable: false }],
            [403, 'permission_error', 'no access', { code: 'http', retryable: false }],
            [404, 'not_found_error', 'model: claude-x', { code: 'http', retryable: false }],
            [408, 'timeout_error', 'took too long', { code: 'http', retryable: true }],
            [409, 'conflict_error', 'conflict', { code: 'http', retryable: true }],
            [413, 'request_too_large', 'too big', { code: 'http', retryable: false }],
            [429, 'rate_limit_error', 'slow down', { code: 'rate_limited', retryable: true }],
            [500, 'api_error', 'Internal server error', { code: 'http', retryable: true }],
            [503, 'api_error', 'unavailable', { code: 'http', retryable: true }],
            [529, 'overloaded_error', 'Overloaded', { code: 'http', retryable: true }],
        ];
        for (const [status, type, value, error] of cases) {
            expect(toProviderError(statusError(status, errorBody(type, value)))).toEqual({
                ...error,
                status,
                message: `HTTP ${status}: ${value}`,
            } as never);
        }
    });

    it('reports only the status without an API message', () => {
        const error = statusError(500, undefined);
        expect(toProviderError(error)).toEqual({
            code: 'http',
            status: 500,
            retryable: true,
            message: 'HTTP 500',
        });
    });

    it('reads retry-after-ms before retry-after, which is seconds or an HTTP date', () => {
        const body = errorBody('rate_limit_error', 'slow down');
        expect(toProviderError(statusError(429, body, { 'retry-after': '2' }))).toEqual({
            code: 'rate_limited',
            status: 429,
            retryable: true,
            retryAfterMs: 2000,
            message: 'HTTP 429: slow down',
        });
        expect(toProviderError(statusError(429, body, { 'retry-after-ms': '1500', 'retry-after': '9' }))).toMatchObject(
            { retryAfterMs: 1500 },
        );
        const later = new Date(Date.now() + 60_000).toUTCString();
        const dated = toProviderError(
            statusError(529, errorBody('overloaded_error', 'Overloaded'), { 'retry-after': later }),
        );
        expect(dated).toMatchObject({ code: 'http', status: 529, retryable: true });
        expect((dated.retryAfterMs ?? 0) > 50_000).toBe(true);
        expect(toProviderError(statusError(503, body))).not.toHaveProperty('retryAfterMs');
    });

    it('maps other SDK errors and TypeErrors to invalid_request, anything else to unexpected', () => {
        expect(toProviderError(new AnthropicError('Streaming is required'))).toEqual({
            code: 'invalid_request',
            message: 'Streaming is required',
            retryable: false,
        });
        expect(toProviderError(new TypeError('bad value'))).toEqual({
            code: 'invalid_request',
            message: 'bad value',
            retryable: false,
        });
        expect(toProviderError(new Error('boom'))).toEqual({ code: 'unexpected', message: 'boom', retryable: false });
        expect(toProviderError('odd')).toMatchObject({ code: 'unexpected', retryable: false });
    });
});
