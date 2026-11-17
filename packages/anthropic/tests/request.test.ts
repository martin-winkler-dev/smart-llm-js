// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'bun:test';
import type { LlmMessage, LlmProviderRequest, LlmReasoningEffort, LlmToolCall } from '@smart-llm/core';
import { assistantMessage, schemaInstruction } from '@smart-llm/core';
import type { AnthropicExtra } from '../src/index.js';
import { toMessageParams } from '../src/request.js';
import { readMessage } from '../src/response.js';
import { message, text } from './fake.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

const base = { model: 'claude-x', max_tokens: 16_000, messages: [{ role: 'user', content: 'hi' }] };

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Maps request with defaults (model claude-x, user hi) and returns params, failing the test on error. */
function mapped(request: Partial<LlmProviderRequest>, extra: AnthropicExtra = {}, defaultMaxTokens = 16_000) {
    const result = toMessageParams(
        { model: 'claude-x', messages: [{ role: 'user', content: 'hi' }], ...request },
        extra,
        defaultMaxTokens,
    );
    if (!result.ok) {
        throw new Error(`Expected params, got ${result.error.code}: ${result.error.message}`);
    }
    return result.data;
}

/** Maps request with defaults and returns whole result. */
function result(request: Partial<LlmProviderRequest>, extra: AnthropicExtra = {}, defaultMaxTokens = 16_000) {
    return toMessageParams(
        { model: 'claude-x', messages: [{ role: 'user', content: 'hi' }], ...request },
        extra,
        defaultMaxTokens,
    );
}

/** Creates tool call with arguments and optional extra_content. */
function toolCall(id: string, name: string, args: string, extraContent?: Record<string, unknown>): LlmToolCall {
    return {
        id,
        type: 'function',
        function: { name, arguments: args },
        ...(extraContent === undefined ? {} : { extra_content: extraContent }),
    };
}

describe('toMessageParams basics', () => {
    it('sends model, messages and default max_tokens only', () => {
        expect(mapped({})).toEqual(base as never);
        expect(mapped({}, {}, 2048)).toEqual({ ...base, max_tokens: 2048 } as never);
        expect(mapped({ max_tokens: 64 })).toEqual({ ...base, max_tokens: 64 } as never);
    });

    it('drops temperature, top_p and seed', () => {
        expect(mapped({ temperature: 0.2, top_p: 0.9, seed: 7 })).toEqual(base as never);
    });

    it('maps stop to stop_sequences and omits an empty list', () => {
        expect(mapped({ stop: 'END' })).toEqual({ ...base, stop_sequences: ['END'] } as never);
        expect(mapped({ stop: ['a', 'b'] })).toEqual({ ...base, stop_sequences: ['a', 'b'] } as never);
        expect(mapped({ stop: [] })).toEqual(base as never);
    });
});

describe('toMessageParams system', () => {
    it('hoists system and developer messages in order, skipping empty texts', () => {
        const params = mapped({
            messages: [
                { role: 'system', content: 'A' },
                {
                    role: 'developer',
                    content: [
                        { type: 'text', text: 'B' },
                        { type: 'text', text: '' },
                        { type: 'text', text: 'C' },
                    ],
                },
                { role: 'user', content: 'hi' },
                { role: 'assistant', content: 'hello' },
                { role: 'system', content: 'D' },
                { role: 'developer', content: '' },
                { role: 'user', content: 'bye' },
            ],
        });
        expect(params).toEqual({
            model: 'claude-x',
            max_tokens: 16_000,
            system: [
                { type: 'text', text: 'A' },
                { type: 'text', text: 'B' },
                { type: 'text', text: 'C' },
                { type: 'text', text: 'D' },
            ],
            messages: [
                { role: 'user', content: 'hi' },
                { role: 'assistant', content: 'hello' },
                { role: 'user', content: 'bye' },
            ],
        } as never);
    });

    it('omits system when all system texts are empty', () => {
        expect(
            mapped({
                messages: [
                    { role: 'system', content: '' },
                    { role: 'user', content: 'hi' },
                ],
            }),
        ).toEqual(base as never);
    });

    it('merges the user turns around a hoisted mid-conversation system message', () => {
        expect(
            mapped({
                messages: [
                    { role: 'user', content: 'hi' },
                    { role: 'developer', content: 'mid' },
                    { role: 'user', content: 'bye' },
                ],
            }).messages,
        ).toEqual([
            {
                role: 'user',
                content: [
                    { type: 'text', text: 'hi' },
                    { type: 'text', text: 'bye' },
                ],
            },
        ]);
    });

    it('appends JSON instruction after system texts for json_object or schemaless json_schema', () => {
        const instruction = { type: 'text', text: schemaInstruction(undefined) };
        const messages: LlmMessage[] = [
            { role: 'system', content: 'Be brief.' },
            { role: 'user', content: 'hi' },
        ];
        const expected = { ...base, system: [{ type: 'text', text: 'Be brief.' }, instruction] };
        expect(mapped({ messages, response_format: { type: 'json_object' } })).toEqual(expected as never);
        expect(mapped({ messages, response_format: { type: 'json_schema', json_schema: { name: 'a' } } })).toEqual(
            expected as never,
        );
        expect(mapped({ response_format: { type: 'json_object' } })).toEqual({
            ...base,
            system: [instruction],
        } as never);
    });

    it('adds no second JSON instruction when core already injected it', () => {
        const params = mapped({
            messages: [
                { role: 'system', content: schemaInstruction(undefined) },
                { role: 'user', content: 'hi' },
            ],
            response_format: { type: 'json_object' },
        });
        expect(params).toEqual({ ...base, system: [{ type: 'text', text: schemaInstruction(undefined) }] } as never);
    });

    it('sends no JSON instruction for text format', () => {
        expect(mapped({ response_format: { type: 'text' } })).toEqual(base as never);
    });
});

describe('toMessageParams user content', () => {
    it('maps text parts and base64 image data URLs, ignoring detail', () => {
        const params = mapped({
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: 'look' },
                        { type: 'image_url', image_url: { url: 'data:image/png;base64,AAAA', detail: 'high' } },
                        { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,BBBB' } },
                        { type: 'image_url', image_url: { url: 'data:image/gif;base64,CCCC' } },
                        { type: 'image_url', image_url: { url: 'data:image/webp;base64,DDDD' } },
                    ],
                },
            ],
        });
        expect(params.messages).toEqual([
            {
                role: 'user',
                content: [
                    { type: 'text', text: 'look' },
                    { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'AAAA' } },
                    { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: 'BBBB' } },
                    { type: 'image', source: { type: 'base64', media_type: 'image/gif', data: 'CCCC' } },
                    { type: 'image', source: { type: 'base64', media_type: 'image/webp', data: 'DDDD' } },
                ],
            },
        ]);
    });

    it('skips empty text parts', () => {
        const params = mapped({
            messages: [
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: '' },
                        { type: 'text', text: 'a' },
                        { type: 'text', text: '' },
                    ],
                },
            ],
        });
        expect(params.messages).toEqual([{ role: 'user', content: [{ type: 'text', text: 'a' }] }]);
    });

    it('rejects empty user turn with invalid_request and merges empty into non-empty turn', () => {
        const error = {
            ok: false,
            error: {
                code: 'invalid_request',
                message: 'A user message has no content (Anthropic rejects empty text)',
                retryable: false,
            },
        } as const;
        expect(result({ messages: [{ role: 'user', content: '' }] })).toEqual(error);
        expect(result({ messages: [{ role: 'user', content: [{ type: 'text', text: '' }] }] })).toEqual(error);
        expect(
            result({
                messages: [
                    { role: 'user', content: 'hi' },
                    { role: 'assistant', content: 'hello' },
                    { role: 'user', content: [] },
                ],
            }),
        ).toEqual(error);
        expect(
            mapped({
                messages: [
                    { role: 'user', content: '' },
                    { role: 'user', content: 'hi' },
                ],
            }).messages,
        ).toEqual([{ role: 'user', content: [{ type: 'text', text: 'hi' }] }]);
    });

    it('rejects unsupported image media types with invalid_request', () => {
        expect(
            result({
                messages: [
                    {
                        role: 'user',
                        content: [{ type: 'image_url', image_url: { url: 'data:image/bmp;base64,AAAA' } }],
                    },
                ],
            }),
        ).toEqual({
            ok: false,
            error: {
                code: 'invalid_request',
                message: 'Anthropic accepts JPEG, PNG, GIF and WebP images, not image/bmp',
                retryable: false,
            },
        });
    });
});

describe('toMessageParams assistant and tool turns', () => {
    it('sends lone assistant string directly, text parts as blocks, and omits reasoning', () => {
        const params = mapped({
            messages: [
                { role: 'user', content: 'hi' },
                { role: 'assistant', content: 'hello', reasoning: 'greet' },
                { role: 'user', content: 'more' },
                {
                    role: 'assistant',
                    content: [
                        { type: 'text', text: 'a' },
                        { type: 'text', text: '' },
                    ],
                },
                { role: 'user', content: 'bye' },
            ],
        });
        expect(params.messages).toEqual([
            { role: 'user', content: 'hi' },
            { role: 'assistant', content: 'hello' },
            { role: 'user', content: 'more' },
            { role: 'assistant', content: [{ type: 'text', text: 'a' }] },
            { role: 'user', content: 'bye' },
        ]);
    });

    it('omits an assistant message without blocks and merges the user turns around it', () => {
        expect(
            mapped({
                messages: [
                    { role: 'user', content: 'hi' },
                    { role: 'assistant', content: '' },
                    { role: 'assistant', content: null },
                    { role: 'user', content: 'again' },
                ],
            }).messages,
        ).toEqual([
            {
                role: 'user',
                content: [
                    { type: 'text', text: 'hi' },
                    { type: 'text', text: 'again' },
                ],
            },
        ]);
    });

    it('maps tool calls to tool_use blocks after the text and groups tool results before the user text', () => {
        const params = mapped({
            messages: [
                { role: 'user', content: 'weather?' },
                {
                    role: 'assistant',
                    content: 'Let me check',
                    tool_calls: [toolCall('t1', 'get_weather', '{"city":"Graz"}'), toolCall('t2', 'now', '')],
                },
                { role: 'tool', tool_call_id: 't1', content: 'sunny' },
                {
                    role: 'tool',
                    tool_call_id: 't2',
                    content: [
                        { type: 'text', text: '12' },
                        { type: 'text', text: 'UTC' },
                    ],
                },
                { role: 'user', content: 'thanks' },
            ],
        });
        expect(params.messages).toEqual([
            { role: 'user', content: 'weather?' },
            {
                role: 'assistant',
                content: [
                    { type: 'text', text: 'Let me check' },
                    { type: 'tool_use', id: 't1', name: 'get_weather', input: { city: 'Graz' } },
                    { type: 'tool_use', id: 't2', name: 'now', input: {} },
                ],
            },
            {
                role: 'user',
                content: [
                    { type: 'tool_result', tool_use_id: 't1', content: 'sunny' },
                    {
                        type: 'tool_result',
                        tool_use_id: 't2',
                        content: [
                            { type: 'text', text: '12' },
                            { type: 'text', text: 'UTC' },
                        ],
                    },
                    { type: 'text', text: 'thanks' },
                ],
            },
        ]);
    });

    it('maps empty tool results without content and skips empty text parts', () => {
        const params = mapped({
            messages: [
                { role: 'user', content: 'go' },
                {
                    role: 'assistant',
                    content: null,
                    tool_calls: [toolCall('t1', 'f', '{}'), toolCall('t2', 'g', '{}')],
                },
                { role: 'tool', tool_call_id: 't1', content: '' },
                {
                    role: 'tool',
                    tool_call_id: 't2',
                    content: [
                        { type: 'text', text: '' },
                        { type: 'text', text: 'x' },
                    ],
                },
                { role: 'tool', tool_call_id: 't3', content: [{ type: 'text', text: '' }] },
                { role: 'user', content: '' },
            ],
        });
        expect(params.messages.at(-1)).toEqual({
            role: 'user',
            content: [
                { type: 'tool_result', tool_use_id: 't1' },
                { type: 'tool_result', tool_use_id: 't2', content: [{ type: 'text', text: 'x' }] },
                { type: 'tool_result', tool_use_id: 't3' },
            ],
        });
    });

    it('puts tool results first when user content precedes them in a merged turn', () => {
        const params = mapped({
            messages: [
                { role: 'user', content: 'go' },
                {
                    role: 'assistant',
                    content: null,
                    tool_calls: [toolCall('t1', 'f', '{}'), toolCall('t2', 'g', '{}')],
                },
                {
                    role: 'user',
                    content: [
                        { type: 'text', text: 'note' },
                        { type: 'image_url', image_url: { url: 'data:image/png;base64,AAAA' } },
                    ],
                },
                { role: 'tool', tool_call_id: 't1', content: 'r1' },
                { role: 'assistant', content: null },
                { role: 'tool', tool_call_id: 't2', content: 'r2' },
            ],
        });
        expect(params.messages.at(-1)).toEqual({
            role: 'user',
            content: [
                { type: 'tool_result', tool_use_id: 't1', content: 'r1' },
                { type: 'tool_result', tool_use_id: 't2', content: 'r2' },
                { type: 'text', text: 'note' },
                { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'AAAA' } },
            ],
        });
    });

    it('fails with invalid_request for tool call arguments that are not a JSON object', () => {
        for (const args of ['"x"', '[1]', '42', 'null']) {
            expect(
                result({
                    messages: [
                        { role: 'user', content: 'go' },
                        { role: 'assistant', content: null, tool_calls: [toolCall('t1', 'f', args)] },
                    ],
                }),
            ).toEqual({
                ok: false,
                error: {
                    code: 'invalid_request',
                    message: 'The arguments of tool call f are not a JSON object',
                    retryable: false,
                },
            });
        }
    });
});

describe('toMessageParams assistant replay', () => {
    const location = {
        type: 'web_search_result_location',
        cited_text: 'Sunny',
        encrypted_index: 'e1',
        url: 'https://w.example',
        title: 'W',
    };
    /** Interleaved answer: thinking, web search, cited text, tool use, more thinking, tool use. */
    const blocks = [
        { type: 'thinking', thinking: 'S1', signature: 'sig1' },
        {
            type: 'server_tool_use',
            id: 'srv1',
            name: 'web_search',
            input: { query: 'Graz' },
            caller: { type: 'direct' },
        },
        {
            type: 'web_search_tool_result',
            tool_use_id: 'srv1',
            caller: { type: 'direct' },
            content: [
                {
                    type: 'web_search_result',
                    url: 'https://w.example',
                    title: 'W',
                    encrypted_content: 'enc',
                    page_age: null,
                },
            ],
        },
        text('Sunny in Graz', [location]),
        { type: 'tool_use', id: 't1', name: 'f', input: { a: 1 }, caller: { type: 'direct' } },
        { type: 'thinking', thinking: 'S2', signature: 'sig2' },
        { type: 'redacted_thinking', data: 'R' },
        { type: 'tool_use', id: 't2', name: 'g', input: {}, caller: { type: 'direct' } },
    ];
    const read = readMessage(message(blocks, { stop_reason: 'tool_use' }), 'claude-x', 1);
    const answer = assistantMessage(read.ok ? read.data : { text: '' });
    const results: LlmMessage[] = [
        { role: 'tool', tool_call_id: 't1', content: 'r1' },
        { role: 'tool', tool_call_id: 't2', content: 'r2' },
    ];
    const toolResults = {
        role: 'user',
        content: [
            { type: 'tool_result', tool_use_id: 't1', content: 'r1' },
            { type: 'tool_result', tool_use_id: 't2', content: 'r2' },
        ],
    };

    it('replays an interleaved answer of the current tool loop verbatim and in order', () => {
        const params = mapped({ messages: [{ role: 'user', content: 'weather?' }, answer, ...results] });
        expect(params.messages).toEqual([
            { role: 'user', content: 'weather?' },
            { role: 'assistant', content: blocks },
            toolResults,
        ] as never);
    });

    it('strips only the thinking blocks of an answer before the last user message', () => {
        const params = mapped({
            messages: [
                { role: 'user', content: 'weather?' },
                answer,
                ...results,
                { role: 'assistant', content: 'done' },
                { role: 'user', content: 'thanks' },
            ],
        });
        expect(params.messages).toEqual([
            { role: 'user', content: 'weather?' },
            { role: 'assistant', content: [blocks[1], blocks[2], blocks[3], blocks[4], blocks[7]] },
            toolResults,
            { role: 'assistant', content: 'done' },
            { role: 'user', content: 'thanks' },
        ] as never);
    });

    it('keeps the thinking of the current tool loop when a user message accompanies the tool results', () => {
        const after = mapped({
            messages: [{ role: 'user', content: 'weather?' }, answer, ...results, { role: 'user', content: 'note' }],
        });
        const before = mapped({
            messages: [{ role: 'user', content: 'weather?' }, answer, { role: 'user', content: 'note' }, ...results],
        });
        const merged = { role: 'user', content: [...toolResults.content, { type: 'text', text: 'note' }] };
        for (const params of [after, before]) {
            expect(params.messages).toEqual([
                { role: 'user', content: 'weather?' },
                { role: 'assistant', content: blocks },
                merged,
            ] as never);
        }
    });

    it('drops empty text blocks of the raw content and does not mutate the message', () => {
        const raw = [text(''), { type: 'tool_use', id: 't1', name: 'f', input: {} }];
        const call = toolCall('t1', 'f', '{}', { anthropic: { content: raw } });
        const before = structuredClone(call);
        const params = mapped({
            messages: [
                { role: 'user', content: 'go' },
                { role: 'assistant', content: null, tool_calls: [call] },
            ],
        });
        expect(params.messages[1]).toEqual({ role: 'assistant', content: [raw[1]] } as never);
        expect(call).toEqual(before);
    });

    it('rebuilds text and tool_use without thinking when ids differ or the raw content is invalid', () => {
        const rebuilt = {
            role: 'assistant',
            content: [
                { type: 'text', text: 'Sunny in Graz' },
                { type: 'tool_use', id: 't1', name: 'f', input: { a: 1 } },
                { type: 'tool_use', id: 't9', name: 'g', input: {} },
            ],
        };
        const variants: unknown[] = [blocks, [...blocks, { type: 'tool_use', id: 't9' }], 'x', [...blocks, 'x'], [{}]];
        for (const content of variants) {
            const params = mapped({
                messages: [
                    { role: 'user', content: 'weather?' },
                    {
                        role: 'assistant',
                        content: 'Sunny in Graz',
                        tool_calls: [
                            toolCall('t1', 'f', '{"a":1}', { anthropic: { content } }),
                            toolCall('t9', 'g', '{}'),
                        ],
                    },
                ],
            });
            expect(params.messages[1]).toEqual(rebuilt as never);
        }
        const withoutRaw = mapped({
            messages: [
                { role: 'user', content: 'weather?' },
                { role: 'assistant', content: 'Sunny in Graz', tool_calls: [toolCall('t1', 'f', '{"a":1}')] },
            ],
        });
        expect(withoutRaw.messages[1]).toEqual({ role: 'assistant', content: rebuilt.content.slice(0, 2) } as never);
    });
});

describe('toMessageParams tools', () => {
    const tools = [
        {
            type: 'function' as const,
            function: {
                name: 'get_weather',
                description: 'Weather of a city',
                parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
                strict: true,
            },
        },
        { type: 'function' as const, function: { name: 'now' } },
        { type: 'function' as const, function: { name: 'loose', strict: false } },
    ];
    const mappedTools = [
        {
            name: 'get_weather',
            description: 'Weather of a city',
            input_schema: {
                type: 'object',
                properties: { city: { type: 'string' } },
                required: ['city'],
                additionalProperties: false,
            },
            strict: true,
        },
        { name: 'now', input_schema: { type: 'object', properties: {} } },
        { name: 'loose', input_schema: { type: 'object', properties: {} } },
    ];

    /** Maps one tool with parameters and strict flag; returns whole result. */
    function toolResult(parameters: Record<string, unknown> | undefined, strict?: boolean) {
        return result({
            tools: [
                {
                    type: 'function',
                    function: {
                        name: 'f',
                        ...(parameters === undefined ? {} : { parameters }),
                        ...(strict === undefined ? {} : { strict }),
                    },
                },
            ],
        });
    }

    it('maps function tools to input_schema with strict only when true', () => {
        expect(mapped({ tools })).toEqual({ ...base, tools: mappedTools } as never);
    });

    it('defaults missing root type to object, preserves non-strict schemas, and avoids mutation', () => {
        const parameters = { properties: { a: { type: 'string', minLength: 1 } }, additionalProperties: true };
        const before = structuredClone(parameters);
        const loose = toolResult(parameters);
        expect(loose.ok && loose.data.tools).toEqual([
            {
                name: 'f',
                input_schema: {
                    type: 'object',
                    properties: { a: { type: 'string', minLength: 1 } },
                    additionalProperties: true,
                },
            },
        ]);
        expect(toolResult(parameters, true)).toMatchObject({ ok: false, error: { code: 'invalid_request' } });
        const strict = toolResult({ properties: parameters.properties }, true);
        expect(strict.ok && strict.data.tools).toEqual([
            {
                name: 'f',
                input_schema: {
                    type: 'object',
                    properties: { a: { type: 'string', description: '{minLength: 1}' } },
                    additionalProperties: false,
                },
                strict: true,
            },
        ]);
        expect(parameters).toEqual(before);
        const empty = toolResult(undefined, true);
        expect(empty.ok && empty.data.tools).toEqual([
            { name: 'f', input_schema: { type: 'object', properties: {}, additionalProperties: false }, strict: true },
        ]);
    });

    it('keeps type object without object rules on a strict root anyOf or $ref schema', () => {
        const branch = { type: 'object', properties: { a: { type: 'string' } }, required: ['a'] };
        const strictBranch = { ...branch, additionalProperties: false };
        const read = toolResult({ anyOf: [branch] }, true);
        expect(read.ok && read.data.tools).toEqual([
            { name: 'f', input_schema: { type: 'object', anyOf: [strictBranch] }, strict: true },
        ]);
        const ref = toolResult({ $ref: '#/$defs/R', $defs: { R: branch } }, true);
        expect(ref.ok && ref.data.tools).toEqual([
            {
                name: 'f',
                input_schema: { type: 'object', $ref: '#/$defs/R', $defs: { R: strictBranch } },
                strict: true,
            },
        ]);
        const loose = toolResult({ anyOf: [branch] });
        expect(loose.ok && loose.data.tools).toEqual([
            { name: 'f', input_schema: { type: 'object', anyOf: [branch] } },
        ]);
    });

    it('rejects boolean sub-schemas and tuple items only in strict tool schemas', () => {
        const cases = [
            [{ a: true }, 'boolean schemas'],
            [{ a: { type: 'array', items: false } }, 'boolean schemas'],
            [{ a: { type: 'array', items: [{ type: 'string' }] } }, 'tuple items'],
        ] as const;
        for (const [properties, cause] of cases) {
            const parameters = { type: 'object' as const, properties };
            expect(toolResult(parameters, true)).toEqual({
                ok: false,
                error: {
                    code: 'invalid_request',
                    message: expect.stringContaining(`${cause} `),
                    retryable: false,
                },
            });
            const loose = toolResult(parameters);
            expect(loose.ok && loose.data.tools).toEqual([{ name: 'f', input_schema: parameters }]);
        }
        expect(toolResult({ type: 'object', properties: { a: true } }, true)).toMatchObject({
            error: {
                message:
                    'The JSON schema of tool f cannot be converted for Anthropic: boolean schemas (true/false) are not supported in Anthropic strict mode',
            },
        });
    });

    it('keeps a typeless enum inside an anyOf branch of a strict tool schema', () => {
        const properties = { a: { anyOf: [{ enum: ['x', 'y'] }, { type: 'null' }] } };
        const read = toolResult({ type: 'object', properties }, true);
        expect(read.ok && read.data.tools).toEqual([
            { name: 'f', input_schema: { type: 'object', properties, additionalProperties: false }, strict: true },
        ]);
    });

    it('keeps enum and const in a strict tool schema, also without type', () => {
        const properties = { unit: { type: 'string', enum: ['C', 'F'] }, kind: { const: 'temp' } };
        const read = toolResult({ type: 'object', properties }, true);
        expect(read.ok && read.data.tools).toEqual([
            { name: 'f', input_schema: { type: 'object', properties, additionalProperties: false }, strict: true },
        ]);
    });

    it('fails with invalid_request for non-object parameters and strict schemas the transform rejects', () => {
        expect(toolResult({ type: 'array', items: { type: 'string' } })).toEqual({
            ok: false,
            error: {
                code: 'invalid_request',
                message: 'The parameters of tool f must be an object schema, not type "array"',
                retryable: false,
            },
        });
        expect(toolResult({ type: ['object', 'null'] })).toMatchObject({
            ok: false,
            error: { code: 'invalid_request' },
        });
        expect(toolResult({ type: 'object', properties: { a: {} } }, true)).toEqual({
            ok: false,
            error: {
                code: 'invalid_request',
                message:
                    'The JSON schema of tool f cannot be converted for Anthropic: JSON schema must have a type defined if anyOf/oneOf/allOf are not used',
                retryable: false,
            },
        });
        for (const property of [{ enum: null }, { enum: 'a' }, { anyOf: [{ enum: null }] }]) {
            expect(toolResult({ type: 'object', properties: { a: property } }, true)).toMatchObject({
                ok: false,
                error: { code: 'invalid_request', message: expect.stringContaining('must have a type defined') },
            });
        }
        expect(toolResult({ type: 'object', properties: { a: {} } }, false).ok).toBe(true);
    });

    it('appends the web search server tool with or without function tools, merged with extra.webSearchTool', () => {
        const search = { type: 'web_search_20260209', name: 'web_search' };
        expect(mapped({ tools, web_search: true })).toEqual({ ...base, tools: [...mappedTools, search] } as never);
        expect(mapped({ web_search: true })).toEqual({ ...base, tools: [search] } as never);
        expect(
            mapped({ web_search: true }, { webSearchTool: { max_uses: 3, allowed_domains: ['example.com'] } }),
        ).toEqual({ ...base, tools: [{ ...search, max_uses: 3, allowed_domains: ['example.com'] }] } as never);
        expect(mapped({ web_search: false })).toEqual(base as never);
    });

    it('maps tool_choice', () => {
        const cases = [
            ['auto', { type: 'auto' }],
            ['none', { type: 'none' }],
            ['required', { type: 'any' }],
            [
                { type: 'function', function: { name: 'now' } },
                { type: 'tool', name: 'now' },
            ],
        ] as const;
        for (const [choice, expected] of cases) {
            expect(mapped({ tools, tool_choice: choice }).tool_choice).toEqual(expected as never);
        }
    });
});

describe('toMessageParams reasoning', () => {
    it('sends adaptive thinking with output_config.effort, mapping minimal to low', () => {
        const cases: [LlmReasoningEffort, string][] = [
            ['minimal', 'low'],
            ['low', 'low'],
            ['medium', 'medium'],
            ['high', 'high'],
            ['xhigh', 'xhigh'],
            ['max', 'max'],
        ];
        for (const [effort, sent] of cases) {
            expect(mapped({ reasoning_effort: effort })).toEqual({
                ...base,
                thinking: { type: 'adaptive', display: 'summarized' },
                output_config: { effort: sent },
            } as never);
        }
        expect(mapped({ reasoning_effort: 'high' }, { thinkingDisplay: 'omitted' })).toEqual({
            ...base,
            thinking: { type: 'adaptive', display: 'omitted' },
            output_config: { effort: 'high' },
        } as never);
    });

    it('sends nothing without reasoning effort', () => {
        expect(mapped({}, { thinkingDisplay: 'omitted', thinkingBudgets: { high: 2048 } })).toEqual(base as never);
    });

    it('maps none by extra.thinkingOff', () => {
        expect(mapped({ reasoning_effort: 'none' })).toEqual({ ...base, thinking: { type: 'disabled' } } as never);
        expect(mapped({ reasoning_effort: 'none' }, { thinkingOff: 'disabled' })).toEqual({
            ...base,
            thinking: { type: 'disabled' },
        } as never);
        expect(mapped({ reasoning_effort: 'none' }, { thinkingOff: 'between_tools' })).toEqual({
            ...base,
            thinking: { type: 'between_tools' },
        } as never);
        expect(mapped({ reasoning_effort: 'none' }, { thinkingOff: 'omit' })).toEqual(base as never);
        expect(mapped({ reasoning_effort: 'none' }, { thinkingBudgets: { high: 2048 } })).toEqual({
            ...base,
            thinking: { type: 'disabled' },
        } as never);
    });

    it('sends enabled thinking with the budget of the effort and no output_config.effort', () => {
        const thinkingBudgets = { minimal: 1024, high: 8000 };
        expect(mapped({ reasoning_effort: 'high' }, { thinkingBudgets })).toEqual({
            ...base,
            thinking: { type: 'enabled', budget_tokens: 8000, display: 'summarized' },
        } as never);
        expect(mapped({ reasoning_effort: 'minimal' }, { thinkingBudgets, thinkingDisplay: 'omitted' })).toEqual({
            ...base,
            thinking: { type: 'enabled', budget_tokens: 1024, display: 'omitted' },
        } as never);
    });

    it('fails with invalid_request for a missing budget or a budget not below max_tokens', () => {
        expect(result({ reasoning_effort: 'low' }, { thinkingBudgets: { high: 8000 } })).toEqual({
            ok: false,
            error: {
                code: 'invalid_request',
                message:
                    'extra.thinkingBudgets has no budget for reasoning effort low; declare capabilities.reasoning.levels matching the budgets',
                retryable: false,
            },
        });
        expect(result({ reasoning_effort: 'high', max_tokens: 8000 }, { thinkingBudgets: { high: 8000 } })).toEqual({
            ok: false,
            error: {
                code: 'invalid_request',
                message: 'The thinking budget 8000 must be below max_tokens 8000',
                retryable: false,
            },
        });
        expect(result({ reasoning_effort: 'high' }, { thinkingBudgets: { high: 20_000 } }, 16_000)).toEqual({
            ok: false,
            error: {
                code: 'invalid_request',
                message: 'The thinking budget 20000 must be below max_tokens 16000',
                retryable: false,
            },
        });
    });
});

describe('toMessageParams JSON schema', () => {
    const schema = {
        type: 'object',
        properties: {
            name: { type: 'string', description: 'Full name', minLength: 2 },
            age: { type: 'integer', minimum: 0 },
            tags: { type: 'array', items: { type: 'string', format: 'email' }, minItems: 3 },
            choice: { anyOf: [{ type: 'object', properties: { d: { type: 'string' } } }, { type: 'null' }] },
            ref: { $ref: '#/$defs/Item' },
        },
        required: ['name'],
        $defs: { Item: { type: 'object', properties: { f: { type: 'string', pattern: '^a' } } } },
    };
    const strict = {
        type: 'object',
        properties: {
            name: { type: 'string', description: 'Full name\n\n{minLength: 2}' },
            age: { type: 'integer', description: '{minimum: 0}' },
            tags: { type: 'array', items: { type: 'string', format: 'email' }, description: '{minItems: 3}' },
            choice: {
                anyOf: [
                    { type: 'object', properties: { d: { type: 'string' } }, additionalProperties: false },
                    { type: 'null' },
                ],
            },
            ref: { $ref: '#/$defs/Item' },
        },
        required: ['name'],
        $defs: {
            Item: {
                type: 'object',
                properties: { f: { type: 'string', description: '{pattern: "^a"}' } },
                additionalProperties: false,
            },
        },
        additionalProperties: false,
    };

    it('sends the SDK-transformed schema as output_config.format, input unchanged', () => {
        const before = structuredClone(schema);
        const params = mapped({ response_format: { type: 'json_schema', json_schema: { name: 'a', schema } } });
        expect(params).toEqual({
            ...base,
            output_config: { format: { type: 'json_schema', schema: strict } },
        } as never);
        expect(schema).toEqual(before);
    });

    /** Maps response format schema; returns whole result. */
    function formatResult(formatSchema: Record<string, unknown>) {
        return result({ response_format: { type: 'json_schema', json_schema: { name: 'a', schema: formatSchema } } });
    }

    /** Maps response format schema; returns sent schema, failing the test on error. */
    function sentSchema(formatSchema: Record<string, unknown>) {
        const read = formatResult(formatSchema);
        if (!read.ok) {
            throw new Error(`Expected params, got ${read.error.code}: ${read.error.message}`);
        }
        return (read.data.output_config?.format as { schema?: unknown } | undefined)?.schema;
    }

    it('keeps enum and const, typed and typeless', () => {
        const properties = {
            kind: { type: 'string', enum: ['a', 'b'] },
            dog: { type: 'string', const: 'dog' },
            any: { enum: ['x', 1, null] },
            none: { const: null },
            mixed: { enum: [1, 2], minimum: 1 },
        };
        expect(sentSchema({ type: 'object', properties })).toEqual({
            type: 'object',
            properties: { ...properties, mixed: { enum: [1, 2], description: '{minimum: 1}' } },
            additionalProperties: false,
        });
    });

    it('applies object, array and string rules to type arrays', () => {
        const properties = {
            obj: {
                type: ['object', 'null'],
                properties: { a: { type: 'string', minLength: 1 } },
                required: ['a'],
                additionalProperties: false,
            },
            list: { type: ['array', 'null'], items: { type: 'string' }, minItems: 1 },
            mail: { type: ['string', 'null'], format: 'email' },
            port: { type: ['integer', 'null'], format: 'int32' },
        };
        expect(sentSchema({ type: 'object', properties })).toEqual({
            type: 'object',
            properties: {
                obj: {
                    type: ['object', 'null'],
                    properties: { a: { type: 'string', description: '{minLength: 1}' } },
                    required: ['a'],
                    additionalProperties: false,
                },
                list: { type: ['array', 'null'], items: { type: 'string' }, minItems: 1 },
                mail: { type: ['string', 'null'], format: 'email' },
                port: { type: ['integer', 'null'], description: '{format: "int32"}' },
            },
            additionalProperties: false,
        });
    });

    it('transforms definitions like $defs, also next to a root $ref', () => {
        const item = { type: 'object', properties: { f: { type: 'string', pattern: '^a' } } };
        const strictItem = {
            type: 'object',
            properties: { f: { type: 'string', description: '{pattern: "^a"}' } },
            additionalProperties: false,
        };
        expect(
            sentSchema({
                type: 'object',
                properties: { item: { $ref: '#/definitions/Item' } },
                definitions: { Item: item },
            }),
        ).toEqual({
            type: 'object',
            properties: { item: { $ref: '#/definitions/Item' } },
            definitions: { Item: strictItem },
            additionalProperties: false,
        });
        expect(sentSchema({ $ref: '#/definitions/Item', definitions: { Item: item } })).toEqual({
            type: 'object',
            $ref: '#/definitions/Item',
            definitions: { Item: strictItem },
        });
    });

    it('validates root schema like tool parameters', () => {
        for (const [root, type] of [
            [{ type: 'array', items: { type: 'string' } }, '"array"'],
            [{ type: ['object', 'null'], properties: {} }, '["object","null"]'],
        ] as const) {
            expect(formatResult(root)).toEqual({
                ok: false,
                error: {
                    code: 'invalid_request',
                    message: `The JSON schema of response format a must be an object schema, not type ${type}`,
                    retryable: false,
                },
            });
        }
        expect(sentSchema({ properties: { a: { type: 'string' } } })).toEqual({
            type: 'object',
            properties: { a: { type: 'string' } },
            additionalProperties: false,
        });
        const anyOf = [{ type: 'object', properties: { a: { type: 'string' } }, required: ['a'] }];
        const strictAnyOf = [{ ...anyOf[0], additionalProperties: false }];
        for (const root of [{ type: 'object', anyOf }, { anyOf }]) {
            expect(sentSchema(root)).toEqual({ type: 'object', anyOf: strictAnyOf });
        }
        expect(sentSchema({ $ref: '#/$defs/R', $defs: { R: anyOf[0] } })).toEqual({
            type: 'object',
            $ref: '#/$defs/R',
            $defs: { R: strictAnyOf[0] },
        });
    });

    it('fails with invalid_request naming boolean sub-schemas and tuple items', () => {
        expect(formatResult({ type: 'object', properties: { a: true } })).toEqual({
            ok: false,
            error: {
                code: 'invalid_request',
                message:
                    'The JSON schema of response format a cannot be converted for Anthropic: boolean schemas (true/false) are not supported in Anthropic strict mode',
                retryable: false,
            },
        });
        expect(
            formatResult({ type: 'object', properties: { a: { type: 'array', items: [{ type: 'string' }] } } }),
        ).toEqual({
            ok: false,
            error: {
                code: 'invalid_request',
                message:
                    'The JSON schema of response format a cannot be converted for Anthropic: tuple items (array of schemas) are not supported in Anthropic strict mode',
                retryable: false,
            },
        });
    });

    it('rejects open and map objects instead of closing them silently', () => {
        for (const open of [true, { type: 'number' }]) {
            const mapSchema = {
                type: 'object',
                properties: { scores: { type: 'object', additionalProperties: open } },
            };
            expect(
                result({ response_format: { type: 'json_schema', json_schema: { name: 'a', schema: mapSchema } } }),
            ).toMatchObject({ ok: false, error: { code: 'invalid_request', retryable: false } });
        }
    });

    it('accepts a property named additionalProperties', () => {
        const named = { type: 'object', properties: { additionalProperties: { type: 'string' } } };
        expect(
            result({ response_format: { type: 'json_schema', json_schema: { name: 'a', schema: named } } }),
        ).toMatchObject({ ok: true });
    });

    it('fails with invalid_request for a schema the transform rejects', () => {
        expect(
            result({
                response_format: {
                    type: 'json_schema',
                    json_schema: { name: 'a', schema: { type: 'object', properties: { x: { description: 'any' } } } },
                },
            }),
        ).toEqual({
            ok: false,
            error: {
                code: 'invalid_request',
                message:
                    'The JSON schema of response format a cannot be converted for Anthropic: JSON schema must have a type defined if anyOf/oneOf/allOf are not used',
                retryable: false,
            },
        });
    });

    it('merges format and effort into one output_config without a JSON instruction', () => {
        const params = mapped({
            reasoning_effort: 'high',
            response_format: { type: 'json_schema', json_schema: { name: 'a', schema: { type: 'object' } } },
        });
        expect(params).toEqual({
            ...base,
            thinking: { type: 'adaptive', display: 'summarized' },
            output_config: {
                effort: 'high',
                format: {
                    type: 'json_schema',
                    schema: { type: 'object', properties: {}, additionalProperties: false },
                },
            },
        } as never);
    });
});

describe('toMessageParams extra.params', () => {
    it('deep-merges params last, replacing arrays and deleting keys set to null', () => {
        const params = mapped(
            { reasoning_effort: 'high', stop: ['a', 'b'] },
            {
                params: {
                    output_config: { effort: 'max' },
                    thinking: null,
                    metadata: { user_id: 'u1' },
                    stop_sequences: ['z'],
                    max_tokens: 99,
                },
            },
        );
        expect(params).toEqual({
            ...base,
            max_tokens: 99,
            stop_sequences: ['z'],
            output_config: { effort: 'max' },
            metadata: { user_id: 'u1' },
        } as never);
    });

    it('deletes a nested key set to null and keeps its siblings', () => {
        const params = mapped(
            {
                reasoning_effort: 'high',
                response_format: { type: 'json_schema', json_schema: { name: 'a', schema: { type: 'object' } } },
            },
            { params: { output_config: { effort: null } } },
        );
        expect(params.output_config).toEqual({
            format: { type: 'json_schema', schema: { type: 'object', properties: {}, additionalProperties: false } },
        } as never);
    });
});
