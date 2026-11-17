// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

// Live: generate and listModels on the Gemini Developer API with the free key in env GEMINI_API_KEY.
// Run with `bun run test:live:gemini` from the repository root; never part of `bun test`.
// Free key: 9 requests (G0 to G8), 7 on gemini-4-hydrogen (20 free calls per day); G0 is listModels, G8 an unknown model.

import { describe, expect, it } from 'bun:test';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
    assistantMessage,
    createLlm,
    imageFromFile,
    type LlmError,
    type LlmErrorCode,
    type LlmMessage,
    type LlmOutput,
    type LlmTool,
    type Result,
    toolMessage,
} from '@smart-llm/core';
import { createGeminiProvider } from '../../src/index.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** API key from env; suite skipped without it. */
const API_KEY = process.env.GEMINI_API_KEY;

/** Model with 20 free requests per day. */
const MODEL = 'gemini-4-hydrogen';

/** Model id unknown to the API (G8). */
const MISSING_MODEL = 'gemini-0-nope';

/** Per-test timeout: above `maxWaitMs` (120 s) plus one request timeout (60 s), so 503 backoff finishes inside test. */
const TEST_TIMEOUT_MS = 200_000;

/** Image with the single word PELICAN; supplied by the user. */
const PELICAN_PATH = fileURLToPath(new URL('../../../core/tests/fixtures/word-pelican.png', import.meta.url));

/** Matches runs of whitespace, collapsed in log lines. */
const MATCH_WHITESPACE = /\s+/g;

/** Classification schema for the invented word "blorf" (a noun). */
const KIND_SCHEMA = {
    type: 'object',
    properties: { kind: { type: 'string', enum: ['noun', 'verb', 'adjective'] } },
    required: ['kind'],
};

/** Schema of the word shown in the image fixture. */
const WORD_SCHEMA = { type: 'object', properties: { word: { type: 'string' } }, required: ['word'] };

/** Function tools with different parameter shapes. */
const TOOLS: LlmTool[] = [
    {
        type: 'function',
        function: {
            name: 'get_weather',
            description: 'Current weather of a city',
            parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
        },
    },
    {
        type: 'function',
        function: {
            name: 'get_time',
            description: 'Current local time in an IANA time zone',
            parameters: { type: 'object', properties: { zone: { type: 'string' } }, required: ['zone'] },
        },
    },
];

/** Weather result returned for every `get_weather` call. */
const WEATHER_RESULT = { city: 'Rome', celsius: 17, sky: 'light rain' };

if (API_KEY === undefined || API_KEY === '') {
    console.warn('[live] GEMINI_API_KEY not set: Gemini generate live tests skipped');
}

/** Test LLM instance; undefined without key. */
const llm = API_KEY
    ? createLlm({
          limiterOptions: { maxRetries: 1, maxWaitMs: 120_000 },
          providers: {
              gemini: {
                  adapter: createGeminiProvider({ timeoutMs: 60_000 }),
                  sharedLimits: { intervalMs: 15_000 },
                  models: {
                      hydrogen: {
                          model: MODEL,
                          capabilities: { images: true, tools: true, json: true, reasoning: { supported: true } },
                      },
                      missing: { model: MISSING_MODEL },
                  },
              },
          },
      })
    : undefined;

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Replaces API key with `<masked>` and collapses whitespace. */
function mask(text: string): string {
    const masked = API_KEY ? text.replaceAll(API_KEY, '<masked>') : text;
    return masked.replace(MATCH_WHITESPACE, ' ');
}

/** Logs one masked line. */
function log(label: string, text: string): void {
    console.log(mask(`[live] ${label}: ${text}`));
}

/** Compact description of an error. */
function describeError(error: LlmError): string {
    const output = error.output === undefined ? '' : ` finish=${error.output.finishReason}`;
    return `${error.code} status=${error.status ?? '-'} retryable=${error.retryable} ${error.message.slice(0, 240)}${output}`;
}

/** Compact description of a generate answer. */
function describeOutput(data: LlmOutput): string {
    const usage = `${data.usage.inputTokens}/${data.usage.outputTokens}/${data.usage.reasoningTokens ?? 0}`;
    return [
        `text=${JSON.stringify(data.text.slice(0, 80))}`,
        data.json === undefined ? '' : `json=${JSON.stringify(data.json).slice(0, 120)}`,
        data.toolCalls === undefined ? '' : `tools=${data.toolCalls.map((call) => call.function.name).join(',')}`,
        data.reasoning === undefined ? '' : `reasoning=${data.reasoning.length}ch`,
        `finish=${data.finishReason} usage=${usage} attempts=${data.attempts}`,
    ]
        .filter((part) => part !== '')
        .join(' ');
}

/** Awaits one live request and logs label, ms and compact result. */
async function send<T>(
    label: string,
    request: () => Promise<Result<T, LlmError>>,
    describeData: (data: T) => string,
): Promise<Result<T, LlmError>> {
    const startedAt = performance.now();
    const result = await request();
    const ms = Math.round(performance.now() - startedAt);
    log(label, `${ms} ms ${result.ok ? `ok ${describeData(result.data)}` : `FAIL ${describeError(result.error)}`}`);
    return result;
}

/**
 * Returns the data of a successful result.
 * @throws {Error} With the masked error when the result failed.
 */
function expectOk<T>(label: string, result: Result<T, LlmError>): T {
    if (!result.ok) {
        throw new Error(mask(`${label}: expected ok, got ${describeError(result.error)}`));
    }
    return result.data;
}

/**
 * Returns the error of a failed result with the expected code (and status).
 * @throws {Error} With a masked description when the result differs.
 */
function expectError<T>(label: string, result: Result<T, LlmError>, code: LlmErrorCode, status?: number): LlmError {
    if (result.ok) {
        throw new Error(`${label}: expected ${code}, got ok`);
    }
    const { error } = result;
    if (error.code !== code || (status !== undefined && error.status !== status)) {
        throw new Error(mask(`${label}: expected ${code} ${status ?? ''}, got ${describeError(error)}`));
    }
    return error;
}

describe.skipIf(llm === undefined)('Gemini generate through createLlm (live, free key)', () => {
    const live = llm as NonNullable<typeof llm>;

    /** History of the tool round trip: set by G4, continued by G5. */
    let toolHistory: LlmMessage[] | undefined;

    it('G0 listModels: model listed with token limits', async () => {
        const label = 'G0 listModels';
        const listed = expectOk(
            label,
            await send(
                label,
                () => live.listModels('gemini'),
                (data) => `${data.length} models`,
            ),
        );
        const model = listed.find((info) => info.id === MODEL);
        expect(model).toBeDefined();
        expect(model?.inputTokenLimit ?? 0).toBeGreaterThan(0);
        expect(model?.outputTokenLimit ?? 0).toBeGreaterThan(0);
    });

    it(
        'G1 system and developer messages, JSON schema, usage, reasoning none',
        async () => {
            const label = 'G1 json schema';
            const data = expectOk(
                label,
                await send(
                    label,
                    () =>
                        live.generate({
                            model: 'hydrogen',
                            messages: [
                                { role: 'system', content: 'You classify invented English words by their role.' },
                                { role: 'developer', content: 'Answer with the JSON object only.' },
                                { role: 'user', content: 'Sentence: "The children found a shiny blorf."' },
                            ],
                            response_format: {
                                type: 'json_schema',
                                json_schema: { name: 'kind', schema: KIND_SCHEMA },
                            },
                            reasoning_effort: 'none',
                            max_tokens: 1000,
                        }),
                    describeOutput,
                ),
            );
            expect(data.json).toEqual({ kind: 'noun' });
            expect(data.usage.inputTokens ?? 0).toBeGreaterThan(0);
            expect(data.usage.outputTokens ?? 0).toBeGreaterThan(0);
            expect(data.settings).toMatchObject({ model: MODEL, jsonMode: 'native', reasoning_effort: 'none' });
        },
        TEST_TIMEOUT_MS,
    );

    it(
        'G2 image input with JSON schema',
        async () => {
            if (!existsSync(PELICAN_PATH)) {
                throw new Error(`G2: image fixture missing at ${PELICAN_PATH}`);
            }
            const image = await imageFromFile(PELICAN_PATH);
            const label = 'G2 image';
            const data = expectOk(
                label,
                await send(
                    label,
                    () =>
                        live.generate({
                            model: 'hydrogen',
                            messages: [
                                {
                                    role: 'user',
                                    content: [{ type: 'text', text: 'Which single word does this image show?' }, image],
                                },
                            ],
                            response_format: {
                                type: 'json_schema',
                                json_schema: { name: 'word', schema: WORD_SCHEMA },
                            },
                            reasoning_effort: 'none',
                            max_tokens: 1000,
                        }),
                    describeOutput,
                ),
            );
            const json = data.json as { word?: unknown };
            expect(String(json.word).toUpperCase()).toBe('PELICAN');
            expect(data.settings.jsonMode).toBe('native');
        },
        TEST_TIMEOUT_MS,
    );

    it(
        'G3 json_object with reasoning',
        async () => {
            const label = 'G3 reasoning';
            const data = expectOk(
                label,
                await send(
                    label,
                    () =>
                        live.generate({
                            model: 'hydrogen',
                            messages: [
                                {
                                    role: 'user',
                                    content:
                                        'Return a JSON object with the number "a" and the string "b": a = 17 * 23, b = the string "zotero" reversed.',
                                },
                            ],
                            response_format: { type: 'json_object' },
                            reasoning_effort: 'high',
                            max_tokens: 4000,
                        }),
                    describeOutput,
                ),
            );
            const json = data.json as { a?: unknown; b?: unknown };
            expect(Number(json.a)).toBe(391);
            expect(json.b).toBe('oretoz');
            expect(data.reasoning?.length ?? 0).toBeGreaterThan(0);
            expect(data.usage.reasoningTokens ?? 0).toBeGreaterThan(0);
            expect(data.settings.reasoning_effort).toBe('high');
        },
        TEST_TIMEOUT_MS,
    );

    it(
        'G4 tool round trip: tool call',
        async () => {
            const label = 'G4 tool call';
            const messages: LlmMessage[] = [
                { role: 'user', content: 'What is the weather in Rome right now? Use the tool.' },
            ];
            const data = expectOk(
                label,
                await send(
                    label,
                    () => live.generate({ model: 'hydrogen', messages, tools: TOOLS, max_tokens: 2000 }),
                    describeOutput,
                ),
            );
            const [call] = data.toolCalls ?? [];
            expect(call?.function.name).toBe('get_weather');
            expect(JSON.parse(call?.function.arguments ?? '{}').city).toContain('Rome');
            const google = call?.extra_content?.google as { thought_signature?: unknown } | undefined;
            expect(typeof google?.thought_signature).toBe('string');
            expect(data.settings.tools).toEqual(['get_weather', 'get_time']);
            messages.push(assistantMessage(data));
            for (const toolCall of data.toolCalls ?? []) {
                messages.push(toolMessage(toolCall, WEATHER_RESULT));
            }
            toolHistory = messages;
        },
        TEST_TIMEOUT_MS,
    );

    it(
        'G5 tool round trip: answer from tool result',
        async () => {
            if (toolHistory === undefined) {
                throw new Error('G5: needs the tool call of G4, which failed or did not run');
            }
            const messages = toolHistory;
            const label = 'G5 tool answer';
            const data = expectOk(
                label,
                await send(
                    label,
                    () => live.generate({ model: 'hydrogen', messages, tools: TOOLS, max_tokens: 2000 }),
                    describeOutput,
                ),
            );
            expect(data.text).toContain('17');
        },
        TEST_TIMEOUT_MS,
    );

    it(
        'G6 sampling: temperature, seed and stop',
        async () => {
            const label = 'G6 sampling';
            const data = expectOk(
                label,
                await send(
                    label,
                    () =>
                        live.generate({
                            model: 'hydrogen',
                            messages: [
                                {
                                    role: 'user',
                                    content: 'Count from 1 to 10 in digits, comma separated, nothing else.',
                                },
                            ],
                            reasoning_effort: 'none',
                            temperature: 0,
                            seed: 7,
                            stop: ['6'],
                            max_tokens: 1000,
                        }),
                    describeOutput,
                ),
            );
            expect(data.text).toContain('5');
            expect(data.text).not.toContain('6');
            expect(data.finishReason).toBe('STOP');
            expect(data.settings).toMatchObject({ reasoning_effort: 'none', temperature: 0, seed: 7, stop: ['6'] });
        },
        TEST_TIMEOUT_MS,
    );

    it(
        'G7 max_tokens too small: truncated with output',
        async () => {
            const label = 'G7 truncated';
            const result = await send(
                label,
                () =>
                    live.generate({
                        model: 'hydrogen',
                        messages: [{ role: 'user', content: 'Write 300 words about bibliographies.' }],
                        reasoning_effort: 'none',
                        max_tokens: 20,
                    }),
                describeOutput,
            );
            const error = expectError(label, result, 'truncated');
            expect(error.output?.finishReason).toBe('MAX_TOKENS');
            expect(error.settings?.max_tokens).toBe(20);
        },
        TEST_TIMEOUT_MS,
    );

    it(
        'G8 unknown model: HTTP 404',
        async () => {
            const label = 'G8 unknown model';
            const result = await send(
                label,
                () => live.generate({ model: 'missing', messages: [{ role: 'user', content: 'x' }] }),
                describeOutput,
            );
            const error = expectError(label, result, 'http', 404);
            expect(error.retryable).toBe(false);
            expect(error.message.trim().length).toBeGreaterThan(0);
        },
        TEST_TIMEOUT_MS,
    );
});
