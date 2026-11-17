// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

// Live: Anthropic Message Batches through createLlm with the key in env ANTHROPIC_API_KEY (opt-in, minutes).
// Run with `bun run test:live:anthropic:batch` from the repository root; never part of `bun test`.
// Requests: 1 batch create + polls every 15 s + 1 results listing (realistic 6 to 22, at most 42); 2 billed.

import { afterAll, beforeAll, describe, expect, it } from 'bun:test';
import { createLlm, type LlmError, type LlmJsonSchema, type LlmOutput, type Result } from '@smart-llm/core';
import { batchTiming } from '../../src/batch.js';
import { createAnthropicProvider } from '../../src/index.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Default model: cheapest current Claude. */
const MODEL = 'claude-haiku-5-5';

/** API key from env; suite skipped without it. */
const API_KEY = process.env.ANTHROPIC_API_KEY;

/** Word class of invented word. */
const KIND_SCHEMA: LlmJsonSchema = {
    type: 'object',
    properties: { kind: { type: 'string', enum: ['noun', 'verb', 'adjective'] } },
    required: ['kind'],
};

/** Batch timing restored after the suite. */
const ORIGINAL_TIMING = { ...batchTiming };

if (API_KEY === undefined || API_KEY === '') {
    console.warn('[live] ANTHROPIC_API_KEY not set: Anthropic batch live test skipped');
}

/** Test LLM instance: Haiku 5.5 with batch calls and one-second collection window. */
const llm = API_KEY
    ? createLlm({
          limiterOptions: { maxRetries: 1, maxWaitMs: 120_000 },
          providers: {
              anthropic: {
                  adapter: createAnthropicProvider({ timeoutMs: 60_000, batchWaitSec: 1 }),
                  models: {
                      haiku: { model: MODEL, capabilities: { json: true, batch: { supported: true } } },
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
    return `FAIL ${error.code} status=${error.status ?? '-'} retryable=${error.retryable} ${error.message}`;
}

/**
 * Logs one masked line and returns answer.
 * @throws {Error} If result failed (masked error description).
 */
function expectOk(label: string, result: Result<LlmOutput, LlmError>): LlmOutput {
    if (!result.ok) {
        console.log(mask(`[live] ${label}: ${describeError(result.error)}`));
        throw new Error(mask(`${label}: ${describeError(result.error)}`));
    }
    const { text, json, finishReason, usage, ms, attempts } = result.data;
    const line =
        `[live] ${label}: ${ms} ms ok text=${JSON.stringify(text.slice(0, 80))} json=${JSON.stringify(json)} ` +
        `finish=${finishReason} usage=${usage.inputTokens}/${usage.outputTokens}/${usage.reasoningTokens} ` +
        `attempts=${attempts}`;
    console.log(mask(line));
    return result.data;
}

describe.skipIf(llm === undefined)('Anthropic Message Batches (live, opt-in)', () => {
    const live = llm as NonNullable<typeof llm>;

    beforeAll(() => {
        batchTiming.pollMs = 15_000;
        batchTiming.maxPollMs = 600_000;
    });

    afterAll(() => {
        Object.assign(batchTiming, ORIGINAL_TIMING);
    });

    it('B batch: two calls share one Message Batch, results by custom_id', async () => {
        const startedAt = Date.now();
        const [pong, kind] = await Promise.all([
            live.haiku('Reply with the single word: pong', {
                batch: true,
                max_tokens: 100,
                signal: AbortSignal.timeout(620_000),
            }),
            live.haiku(
                [
                    { role: 'system', content: 'You classify invented English words by their role in a sentence.' },
                    { role: 'developer', content: 'Answer only with the JSON object.' },
                    { role: 'user', content: 'Sentence: "The children found a shiny blorf under the bridge."' },
                ],
                {
                    response_format: { type: 'json_schema', json_schema: { name: 'answer', schema: KIND_SCHEMA } },
                    batch: true,
                    max_tokens: 100,
                    signal: AbortSignal.timeout(620_000),
                },
            ),
        ]);
        console.log(`[live] B: batch settled after ${Math.round((Date.now() - startedAt) / 1000)} s`);
        const pongData = expectOk('B pong', pong);
        const kindData = expectOk('B kind', kind);
        expect(pongData.text.toLowerCase()).toContain('pong');
        expect(kindData.json).toEqual({ kind: 'noun' });
        expect(pongData.settings.batch).toBe(true);
        expect(kindData.settings.batch).toBe(true);
    }, 630_000);
});
