// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

// Live: chat completions on the DeepSeek API with the key in env DEEPSEEK_API_KEY.
// Run with `bun run test:live:deepseek` from the repository root; never part of `bun test`.
// Requests: 9 HTTP requests (D0 to D8, D0 free), at most 18 with retries; L1 sends none.

import { describe, expect, it } from 'bun:test';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { LlmError, LlmErrorCode, LlmMessage, LlmOutput, LlmTool, Result } from '@smart-llm/core';
import { assistantMessage, createLlm, imageFromFile, toolMessage } from '@smart-llm/core';
import { createDeepseekProvider } from '../../src/index.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** DeepSeek V4.1 Flash: id from DeepSeek API docs, checked by D0. */
const MODEL = 'deepseek-flash';

/** API key from env; suite skipped without it. */
const API_KEY = process.env.DEEPSEEK_API_KEY;

/** Shared image fixture with the word `PELICAN` (supplied by the user). */
const FIXTURE = fileURLToPath(new URL('../../../core/tests/fixtures/word-pelican.png', import.meta.url));

/** Schema: word class of an invented word. */
const KIND_SCHEMA = {
    type: 'object',
    properties: { kind: { type: 'string', enum: ['noun', 'verb', 'adjective'] } },
    required: ['kind'],
};

/** Schema: word read from the image. */
const WORD_SCHEMA = { type: 'object', properties: { word: { type: 'string' } }, required: ['word'] };

/** Tools: strict `get_weather` and non-strict `get_time`. */
const TOOLS: LlmTool[] = [
    {
        type: 'function',
        function: {
            name: 'get_weather',
            description: 'Current weather of a city.',
            parameters: {
                type: 'object',
                properties: { city: { type: 'string' } },
                required: ['city'],
                additionalProperties: false,
            },
            strict: true,
        },
    },
    {
        type: 'function',
        function: {
            name: 'get_time',
            description: 'Current time in a time zone.',
            parameters: { type: 'object', properties: { zone: { type: 'string' } }, required: ['zone'] },
        },
    },
];

/** Fake `get_weather` result for D5. */
const WEATHER_RESULT = { city: 'Rome', celsius: 17, sky: 'light rain' };

/** User message of the tool round trip (D4, D5). */
const WEATHER_MESSAGES: LlmMessage[] = [
    {
        role: 'user',
        content:
            'Compute (4817 * 293) mod 97. If the result is odd, get the current weather for Rome; if it is even, for Oslo. Use the tool.',
    },
];

if (API_KEY === undefined || API_KEY === '') {
    console.warn('[live] DEEPSEEK_API_KEY not set: DeepSeek live tests skipped');
}

/** Test LLM instance; undefined without key. */
const llm = API_KEY
    ? createLlm({
          limiterOptions: { maxRetries: 1, maxWaitMs: 120_000 },
          providers: {
              deepseek: {
                  adapter: createDeepseekProvider({ apiKey: API_KEY, timeoutMs: 60_000 }),
                  sharedLimits: { intervalMs: 1_000 },
                  models: {
                      flash: {
                          model: MODEL,
                          capabilities: { images: true, tools: true, json: true, reasoning: { supported: true } },
                      },
                      missing: { model: 'deepseek-nope-0' },
                  },
              },
          },
      })
    : undefined;

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Replaces the API key with `<masked>`. */
function mask(text: string): string {
    return API_KEY ? text.replaceAll(API_KEY, '<masked>') : text;
}

/** Logs one line per request: timing, text, JSON, tools, reasoning, finish, usage, attempts or failure. */
function logResult(label: string, result: Result<LlmOutput, LlmError>): void {
    if (!result.ok) {
        const { code, status, retryable, message } = result.error;
        console.log(mask(`[live] ${label}: FAIL ${code} status=${status} retryable=${retryable} ${message}`));
        return;
    }
    const { ms, text, json, toolCalls, reasoning, finishReason, usage, attempts } = result.data;
    const tools = toolCalls?.map((call) => call.function.name).join(',') ?? '';
    const tokens = `${usage.inputTokens}/${usage.outputTokens}/${usage.reasoningTokens}`;
    const parts = [
        `${ms} ms ok text=${JSON.stringify(text.slice(0, 80))}`,
        `json=${JSON.stringify(json)}`,
        `tools=${tools}`,
        `reasoning=${reasoning?.length ?? 0}ch`,
        `finish=${finishReason}`,
        `usage=${tokens}`,
        `attempts=${attempts}`,
    ];
    console.log(mask(`[live] ${label}: ${parts.join(' ')}`));
}

/**
 * Logs result and returns its data.
 * @throws {Error} When the result is not ok (masked error description).
 */
function expectOk(label: string, result: Result<LlmOutput, LlmError>): LlmOutput {
    logResult(label, result);
    if (!result.ok) {
        throw new Error(mask(`${label}: expected ok, got ${JSON.stringify(result.error)}`));
    }
    return result.data;
}

/**
 * Logs result and returns its error.
 * @throws {Error} When the result is ok or code/status differ.
 */
function expectError(
    label: string,
    result: Result<LlmOutput, LlmError>,
    code: LlmErrorCode,
    status?: number,
): LlmError {
    logResult(label, result);
    if (result.ok) {
        throw new Error(mask(`${label}: expected ${code}, got ok ${JSON.stringify(result.data.text)}`));
    }
    if (result.error.code !== code || (status !== undefined && result.error.status !== status)) {
        throw new Error(mask(`${label}: expected ${code} ${status ?? ''}, got ${JSON.stringify(result.error)}`));
    }
    return result.error;
}

/**
 * Returns the configured LLM instance.
 * @throws {Error} When the key is missing (suite skipped then).
 */
function live(): NonNullable<typeof llm> {
    if (llm === undefined) {
        throw new Error('DEEPSEEK_API_KEY not set');
    }
    return llm;
}

describe.skipIf(llm === undefined)('DeepSeek chat completions (live)', () => {
    /** D4 answer, replayed by D5. */
    let toolTurn: LlmOutput | undefined;

    it('D0 listModels: the model id is listed', async () => {
        const result = await live().listModels('deepseek');
        if (!result.ok) {
            throw new Error(mask(`D0: listModels failed ${JSON.stringify(result.error)}`));
        }
        const ids = result.data.map((model) => model.id);
        if (!ids.includes(MODEL)) {
            console.log(`[live] D0: ${MODEL} not listed; available ids: ${ids.join(', ')}`);
        }
        expect(ids).toContain(MODEL);
    }, 60_000);

    it('D1 system and developer messages, JSON schema in prompt mode, usage, thinking off', async () => {
        const result = await live().flash(
            [
                { role: 'system', content: 'You classify invented English words by their role in a sentence.' },
                { role: 'developer', content: 'Answer only with the JSON object.' },
                { role: 'user', content: 'Sentence: "The children found a shiny blorf under the bridge."' },
            ],
            {
                response_format: { type: 'json_schema', json_schema: { name: 'answer', schema: KIND_SCHEMA } },
                max_tokens: 200,
                reasoning_effort: 'none',
            },
        );
        const data = expectOk('D1', result);
        expect(data.json).toEqual({ kind: 'noun' });
        expect(data.finishReason).toBe('stop');
        expect(data.usage.inputTokens ?? 0).toBeGreaterThan(0);
        expect(data.reasoning).toBeUndefined();
        expect(data.usage.reasoningTokens ?? 0).toBe(0);
        expect(data.usage.outputTokens ?? 0).toBeGreaterThan(0);
        expect(data.settings).toMatchObject({
            name: 'flash',
            model: MODEL,
            jsonMode: 'prompt',
            reasoning_effort: 'none',
        });
        expect(data.attempts).toBe(1);
    }, 60_000);

    it('D2 image input with JSON schema', async () => {
        if (!existsSync(FIXTURE)) {
            throw new Error(`D2: image fixture missing at ${FIXTURE}`);
        }
        const result = await live().flash(
            [{ type: 'text', text: 'Which word is written in this image?' }, await imageFromFile(FIXTURE)],
            {
                response_format: { type: 'json_schema', json_schema: { name: 'answer', schema: WORD_SCHEMA } },
                max_tokens: 200,
                reasoning_effort: 'none',
            },
        );
        const data = expectOk('D2', result);
        expect(String((data.json as { word?: unknown }).word).toUpperCase()).toBe('PELICAN');
    }, 60_000);

    it('D3 json_object with thinking: reasoning_content and reasoning tokens', async () => {
        const result = await live().flash(
            'Return a JSON object with the number "a" and the string "b": a = 17 * 23, b = the string "zotero" reversed.',
            {
                response_format: { type: 'json_object' },
                max_tokens: 3000,
                reasoning_effort: 'low',
            },
        );
        const data = expectOk('D3', result);
        const json = data.json as { a?: unknown; b?: unknown };
        expect(Number(json.a)).toBe(391);
        expect(json.b).toBe('oretoz');
        expect(data.reasoning?.length ?? 0).toBeGreaterThan(0);
        expect(data.usage.reasoningTokens ?? 0).toBeGreaterThan(0);
        expect(data.settings.reasoning_effort).toBe('low');
    }, 60_000);

    it('D4 tool round trip in thinking mode: tool call', async () => {
        const result = await live().flash(WEATHER_MESSAGES, {
            tools: TOOLS,
            tool_choice: 'auto',
            max_tokens: 2000,
            reasoning_effort: 'low',
        });
        const data = expectOk('D4', result);
        expect(data.reasoning ?? '').not.toBe('');
        const [call] = data.toolCalls ?? [];
        expect(call?.function.name).toBe('get_weather');
        expect(call?.id ?? '').not.toBe('');
        expect(String((JSON.parse(call?.function.arguments ?? '{}') as { city?: unknown }).city)).toContain('Rome');
        expect(data.finishReason).toBe('tool_calls');
        expect(data.settings.tools).toEqual(['get_weather', 'get_time']);
        toolTurn = data;
    }, 60_000);

    it('D5 tool round trip in thinking mode: answer from tool result', async () => {
        if (toolTurn === undefined) {
            throw new Error('D5: needs the tool call of D4, which failed or did not run');
        }
        const results = (toolTurn.toolCalls ?? []).map((call) => toolMessage(call, WEATHER_RESULT));
        const result = await live().flash([...WEATHER_MESSAGES, assistantMessage(toolTurn), ...results], {
            tools: TOOLS,
            max_tokens: 2000,
            reasoning_effort: 'low',
        });
        const data = expectOk('D5', result);
        expect(data.text).toContain('17');
    }, 60_000);

    it('D6 stop sequence and seed', async () => {
        const result = await live().flash('Count from 1 to 10 in digits, comma separated, nothing else.', {
            max_tokens: 60,
            reasoning_effort: 'none',
            stop: ['6'],
            seed: 7,
            temperature: 0,
        });
        const data = expectOk('D6', result);
        expect(data.text).toContain('5');
        expect(data.text).not.toContain('6');
        expect(data.finishReason).toBe('stop');
        expect(data.settings).toMatchObject({ seed: 7, temperature: 0, stop: ['6'] });
    }, 60_000);

    it('D7 max_tokens too small: truncated with output', async () => {
        const result = await live().flash('Write 300 words about bibliographies.', {
            max_tokens: 5,
            reasoning_effort: 'none',
        });
        const error = expectError('D7', result, 'truncated');
        expect(error.output?.finishReason).toBe('length');
    }, 60_000);

    it('D8 unknown model: HTTP error', async () => {
        const result = await live().missing('x', { max_tokens: 5 });
        const error = expectError('D8', result, 'http');
        console.log(`[live] D8: status ${error.status}`);
        expect([400, 404]).toContain(error.status ?? 0);
        expect(error.retryable).toBe(false);
    }, 60_000);

    it('L1 local checks: web search and batch unsupported', async () => {
        const search = await live().flash('x', { web_search: true });
        const batch = await live().flash('x', { batch: true });
        expect(search).toMatchObject({ ok: false, error: { code: 'unsupported' } });
        expect(batch).toMatchObject({ ok: false, error: { code: 'unsupported' } });
    }, 60_000);
});
