// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

// Live: Anthropic Messages API through createLlm with the key in env ANTHROPIC_API_KEY.
// Run with `bun run test:live:anthropic` from the repository root; never part of `bun test`.
// Requests: 12 HTTP requests (A0 to A11, 11 billed); local check L1 sends none.

import { describe, expect, it } from 'bun:test';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
    assistantMessage,
    createLlm,
    imageFromFile,
    imagePart,
    type LlmError,
    type LlmErrorCode,
    type LlmJsonSchema,
    type LlmMessage,
    type LlmOutput,
    type LlmTool,
    type Result,
    toolMessage,
} from '@smart-llm/core';
import { createAnthropicProvider } from '../../src/index.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Default model: cheapest current Claude. */
const MODEL = 'claude-haiku-5-5';

/** Model for web search and `thinkingOff: 'between_tools'` (one request). */
const SONNET = 'claude-sonnet-5-5';

/** Model for manual thinking budget (only current model accepting `budget_tokens`). */
const HAIKU_45 = 'claude-haiku-4-5';

/** API key from env; suite skipped without it. */
const API_KEY = process.env.ANTHROPIC_API_KEY;

/** Image fixture: the word PELICAN (supplied by the user). */
const FIXTURE = fileURLToPath(new URL('../../../core/tests/fixtures/word-pelican.png', import.meta.url));

/** Word class of invented word. */
const KIND_SCHEMA: LlmJsonSchema = {
    type: 'object',
    properties: { kind: { type: 'string', enum: ['noun', 'verb', 'adjective'] } },
    required: ['kind'],
};

/** Word written in image. */
const WORD_SCHEMA: LlmJsonSchema = { type: 'object', properties: { word: { type: 'string' } }, required: ['word'] };

/** Function tools: `get_weather` strict, `get_time` not. */
const TOOLS: LlmTool[] = [
    {
        type: 'function',
        function: {
            name: 'get_weather',
            description: 'Current weather of a city.',
            parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
            strict: true,
        },
    },
    {
        type: 'function',
        function: {
            name: 'get_time',
            description: 'Current local time of a time zone.',
            parameters: { type: 'object', properties: { zone: { type: 'string' } }, required: ['zone'] },
        },
    },
];

/** Tool result for `get_weather`. */
const WEATHER_RESULT = { city: 'Rome', celsius: 17, sky: 'light rain' };

/** Messages of classification request (A1). */
const CLASSIFY_MESSAGES: LlmMessage[] = [
    { role: 'system', content: 'You classify invented English words by their role in a sentence.' },
    { role: 'developer', content: 'Answer only with the JSON object.' },
    { role: 'user', content: 'Sentence: "The children found a shiny blorf under the bridge."' },
];

if (API_KEY === undefined || API_KEY === '') {
    console.warn('[live] ANTHROPIC_API_KEY not set: Anthropic live tests skipped');
}

/** Test LLM instance: Haiku 5.5, Sonnet 5.5 (web search), Haiku 4.5 (thinking budget), unknown model. */
const llm = API_KEY
    ? createLlm({
          limiterOptions: { maxRetries: 1, maxWaitMs: 120_000 },
          providers: {
              anthropic: {
                  adapter: createAnthropicProvider({ timeoutMs: 60_000 }),
                  sharedLimits: { intervalMs: 1_000 },
                  models: {
                      haiku: {
                          model: MODEL,
                          capabilities: {
                              images: true,
                              tools: true,
                              json: true,
                              reasoning: { supported: true, levels: ['none', 'low', 'medium', 'high'] },
                          },
                      },
                      sonnet: {
                          model: SONNET,
                          capabilities: {
                              json: true,
                              webSearch: { supported: true },
                              reasoning: { supported: true, levels: ['none', 'low', 'medium', 'high'] },
                          },
                          extra: { thinkingOff: 'between_tools', webSearchTool: { max_uses: 1 } },
                      },
                      haiku45: {
                          model: HAIKU_45,
                          capabilities: { reasoning: { supported: true, levels: ['none', 'low'] } },
                          extra: { thinkingBudgets: { low: 1024 } },
                      },
                      missing: { model: 'claude-nope-0' },
                  },
              },
          },
      })
    : undefined;

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Replaces API key value with `<masked>`. */
function mask(text: string): string {
    return API_KEY ? text.split(API_KEY).join('<masked>') : text;
}

/** Formats error as `FAIL <code> status=<s> retryable=<b> <message>`. */
function describeError(error: LlmError): string {
    const text = error.output === undefined ? '' : ` text=${JSON.stringify(error.output.text.slice(0, 300))}`;
    return `FAIL ${error.code} status=${error.status ?? '-'} retryable=${error.retryable} ${error.message}${text}`;
}

/** Logs one masked line per request: answer summary or error. */
function logResult(label: string, result: Result<LlmOutput, LlmError>): void {
    if (!result.ok) {
        console.log(mask(`[live] ${label}: ${describeError(result.error)}`));
        return;
    }
    const { text, json, toolCalls, reasoning, finishReason, usage, ms, attempts } = result.data;
    const tools = (toolCalls ?? []).map((call) => call.function.name).join(',');
    const line =
        `[live] ${label}: ${ms} ms ok text=${JSON.stringify(text.slice(0, 80))} json=${JSON.stringify(json)} ` +
        `tools=${tools} reasoning=${reasoning?.length ?? 0}ch finish=${finishReason} ` +
        `usage=${usage.inputTokens}/${usage.outputTokens}/${usage.reasoningTokens} attempts=${attempts}`;
    console.log(mask(line));
}

/**
 * Logs result and returns answer.
 * @throws {Error} If result failed (masked error description).
 */
function expectOk(label: string, result: Result<LlmOutput, LlmError>): LlmOutput {
    logResult(label, result);
    if (!result.ok) {
        throw new Error(mask(`${label}: ${describeError(result.error)}`));
    }
    return result.data;
}

/**
 * Logs result and returns error.
 * @throws {Error} If result ok, or code or status differ.
 */
function expectError(
    label: string,
    result: Result<LlmOutput, LlmError>,
    code: LlmErrorCode,
    status?: number,
): LlmError {
    logResult(label, result);
    if (result.ok) {
        throw new Error(`${label}: expected error ${code}, got ok`);
    }
    if (result.error.code !== code || (status !== undefined && result.error.status !== status)) {
        throw new Error(mask(`${label}: expected ${code} ${status ?? ''}, got ${describeError(result.error)}`));
    }
    return result.error;
}

/** Reads raw content blocks of first tool call (`extra_content.anthropic.content`). */
function rawContent(output: LlmOutput): unknown {
    const anthropic = output.toolCalls?.[0]?.extra_content?.anthropic as { content?: unknown } | undefined;
    return anthropic?.content;
}

describe.skipIf(llm === undefined)('Anthropic Messages API (live)', () => {
    const live = llm as NonNullable<typeof llm>;

    /** A4 request and answer, replayed by A5. */
    let toolTurn: { messages: LlmMessage[]; output: LlmOutput } | undefined;

    it('A0 listModels: both model ids listed with token limits', async () => {
        const result = await live.listModels('anthropic');
        if (!result.ok) {
            console.log(mask(`[live] A0: ${describeError(result.error)}`));
            throw new Error(mask(`A0: ${describeError(result.error)}`));
        }
        const ids = result.data.map((model) => model.id);
        console.log(`[live] A0: ok ${ids.length} models: ${ids.join(' ')}`);
        expect(ids).toContain(MODEL);
        expect(ids).toContain(SONNET);
        const haiku = result.data.find((model) => model.id === MODEL);
        expect(haiku?.inputTokenLimit).toBeGreaterThan(0);
        expect(haiku?.outputTokenLimit).toBeGreaterThan(0);
    }, 60_000);

    it('A1 system and developer messages, JSON schema, usage, thinking disabled, extra.params', async () => {
        const result = await live.haiku(CLASSIFY_MESSAGES, {
            response_format: { type: 'json_schema', json_schema: { name: 'answer', schema: KIND_SCHEMA } },
            max_tokens: 200,
            reasoning_effort: 'none',
            extra: { params: { metadata: { user_id: 'live-test' } } },
        });
        const data = expectOk('A1', result);
        expect(data.json).toEqual({ kind: 'noun' });
        expect(data.finishReason).toBe('end_turn');
        expect(data.usage.inputTokens).toBeGreaterThan(0);
        expect(data.usage.outputTokens).toBeGreaterThan(0);
        expect([null, 0]).toContain(data.usage.reasoningTokens);
        expect(data.reasoning).toBeUndefined();
        expect(data.settings).toMatchObject({
            name: 'haiku',
            model: MODEL,
            jsonMode: 'native',
            reasoning_effort: 'none',
            max_tokens: 200,
        });
        expect(data.attempts).toBe(1);
        expect(data.model.startsWith(MODEL)).toBe(true);
    }, 60_000);

    it('A2 image input with JSON schema', async () => {
        if (!existsSync(FIXTURE)) {
            throw new Error(`A2: image fixture missing at ${FIXTURE}`);
        }
        const image = await imageFromFile(FIXTURE);
        const result = await live.haiku([{ type: 'text', text: 'Which word is written in this image?' }, image], {
            response_format: { type: 'json_schema', json_schema: { name: 'word', schema: WORD_SCHEMA } },
            max_tokens: 300,
        });
        const data = expectOk('A2', result);
        expect(String((data.json as { word?: unknown }).word).toUpperCase()).toBe('PELICAN');
        expect(data.usage.inputTokens).toBeGreaterThan(0);
        expect(data.settings.reasoning_effort).toBeUndefined();
    }, 60_000);

    it('A3 json_object with adaptive thinking', async () => {
        const result = await live.haiku(
            'Return a JSON object with the integers "a" and "b": a = (4817 * 293) mod 97, b = the digit sum of 4817 * 293.',
            { response_format: { type: 'json_object' }, max_tokens: 3000, reasoning_effort: 'medium' },
        );
        const data = expectOk('A3', result);
        const json = data.json as { a?: unknown; b?: unknown };
        expect(Number(json.a)).toBe(31);
        expect(Number(json.b)).toBe(19);
        expect(data.settings.reasoning_effort).toBe('medium');
        expect(typeof data.reasoning === 'string' && data.reasoning.length > 0).toBe(true);
        expect(data.usage.reasoningTokens).toBeGreaterThan(0);
    }, 60_000);

    it('A4 tool round trip: tool call with thinking', async () => {
        const messages: LlmMessage[] = [
            {
                role: 'user',
                content:
                    'Compute (4817 * 293) mod 97. If the result is odd, get the current weather for Rome; if it is even, for Oslo. Use the tool.',
            },
        ];
        const result = await live.haiku(messages, {
            tools: TOOLS,
            tool_choice: 'auto',
            max_tokens: 2000,
            reasoning_effort: 'medium',
        });
        const data = expectOk('A4', result);
        const [call] = data.toolCalls ?? [];
        expect(call?.function.name).toBe('get_weather');
        expect(String((JSON.parse(call?.function.arguments ?? '{}') as { city?: unknown }).city)).toContain('Rome');
        expect(data.finishReason).toBe('tool_use');
        const raw = rawContent(data);
        expect(Array.isArray(raw)).toBe(true);
        const blocks = (Array.isArray(raw) ? raw : []) as { type?: unknown }[];
        expect(blocks.some((block) => block.type === 'tool_use')).toBe(true);
        expect(data.settings.tools).toEqual(['get_weather', 'get_time']);
        expect(blocks.some((block) => block.type === 'thinking')).toBe(true);
        toolTurn = { messages, output: data };
    }, 60_000);

    it('A5 tool round trip: answer from tool result', async () => {
        if (toolTurn === undefined) {
            throw new Error('A5: needs the tool call answer of A4, which failed or did not run');
        }
        const { messages, output } = toolTurn;
        const results = (output.toolCalls ?? []).map((call) => toolMessage(call, WEATHER_RESULT));
        const result = await live.haiku([...messages, assistantMessage(output), ...results], {
            tools: TOOLS,
            tool_choice: 'none',
            max_tokens: 500,
            reasoning_effort: 'medium',
        });
        const data = expectOk('A5', result);
        expect(data.text).toContain('17');
        expect(data.toolCalls).toBeUndefined();
    }, 60_000);

    it('A6 forced tool choice: required', async () => {
        const result = await live.haiku('Hello!', {
            tools: TOOLS,
            tool_choice: 'required',
            max_tokens: 200,
            reasoning_effort: 'none',
        });
        const data = expectOk('A6', result);
        expect(data.toolCalls?.length ?? 0).toBeGreaterThan(0);
        expect(data.settings.tool_choice).toBe('required');
    }, 60_000);

    it('A7 stop sequence', async () => {
        const result = await live.haiku('Count from 1 to 10 in digits, comma separated, nothing else.', {
            max_tokens: 60,
            reasoning_effort: 'none',
            stop: ['6'],
        });
        const data = expectOk('A7', result);
        expect(data.text).toContain('5');
        expect(data.text).not.toContain('6');
        expect(data.finishReason).toBe('stop_sequence');
    }, 60_000);

    it('A8 max_tokens too small: truncated with output', async () => {
        const result = await live.haiku('Write 300 words about bibliographies.', {
            max_tokens: 5,
            reasoning_effort: 'none',
        });
        const error = expectError('A8', result, 'truncated');
        expect(error.output?.finishReason).toBe('max_tokens');
        expect(error.settings?.max_tokens).toBe(5);
        expect(error.retryable).toBe(false);
    }, 60_000);

    it('A9 unknown model: HTTP 404', async () => {
        const result = await live.missing('x', { max_tokens: 5 });
        const error = expectError('A9', result, 'http', 404);
        expect(error.retryable).toBe(false);
        expect(error.message.startsWith('HTTP 404')).toBe(true);
    }, 60_000);

    it('A10 web search with sources and thinking between tools', async () => {
        const result = await live.sonnet('Search the web: which organisation develops Zotero? One sentence.', {
            web_search: true,
            reasoning_effort: 'none',
            max_tokens: 600,
        });
        const data = expectOk('A10', result);
        expect(data.text.length).toBeGreaterThan(0);
        expect(data.sources?.length ?? 0).toBeGreaterThan(0);
        expect(data.sources?.[0]?.url.startsWith('http')).toBe(true);
        expect(data.settings.web_search).toBe(true);
    }, 60_000);

    it('A11 manual thinking budget on Haiku 4.5', async () => {
        const result = await live.haiku45('What is 17 * 23? Answer with the number only.', {
            reasoning_effort: 'low',
            max_tokens: 1500,
        });
        const data = expectOk('A11', result);
        expect(data.text).toContain('391');
        expect(data.usage.reasoningTokens).toBeGreaterThan(0);
        expect(data.settings).toMatchObject({ reasoning_effort: 'low', extra: { thinkingBudgets: { low: 1024 } } });
    }, 60_000);

    it('L1 local validation: invalid_request without a request', async () => {
        const bmp = await live.haiku(
            [
                { type: 'text', text: 'Describe this image.' },
                imagePart(new Uint8Array([0x42, 0x4d, 0, 0]), 'image/bmp'),
            ],
            { max_tokens: 5 },
        );
        const tool = await live.haiku('Hello!', {
            tools: [{ type: 'function', function: { name: 'echo', parameters: { type: 'string' } } }],
            max_tokens: 5,
        });
        const open = await live.haiku('Hello!', {
            response_format: {
                type: 'json_schema',
                json_schema: {
                    name: 'open',
                    schema: { type: 'object', properties: { a: { type: 'string' } }, additionalProperties: true },
                },
            },
            max_tokens: 5,
        });
        const cases = [
            ['L1 bmp', bmp, 'image/bmp'],
            ['L1 tool', tool, 'object schema'],
            ['L1 open schema', open, 'additionalProperties'],
        ] as const;
        for (const [label, result, reason] of cases) {
            const error = expectError(label, result, 'invalid_request');
            expect(error.message).toContain(reason);
            expect(error.status).toBeUndefined();
        }
    }, 60_000);
});
