// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type {
    LlmProvider,
    LlmProviderError,
    LlmProviderOutput,
    LlmProviderRequest,
    LlmProviderSupport,
    Result,
} from '../src/types.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Call record captured by fake provider. */
export type FakeCall = { request: LlmProviderRequest; extra: Record<string, unknown> };

/** Reply handler for simulated provider call. */
export type FakeReply = (call: FakeCall) => Result<LlmProviderOutput, LlmProviderError>;

/** Fake provider recording calls and queueing replies (repeats last reply). */
export type FakeProvider = LlmProvider & { calls: FakeCall[]; reply(...replies: FakeReply[]): void };

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Full provider capabilities support with native JSON mode. */
export const FULL_SUPPORT: LlmProviderSupport = {
    images: true,
    tools: true,
    webSearch: true,
    reasoning: true,
    batch: true,
    json: 'native',
};

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/** Creates fake successful reply with text and simulated token usage. */
export function answer(text: string): FakeReply {
    return (call) => ({
        ok: true,
        data: {
            text,
            model: call.request.model,
            usage: { inputTokens: 10, outputTokens: 5, reasoningTokens: 0 },
            finishReason: 'stop',
            ms: 1,
        },
    });
}

/** Creates fake error reply with specified error. */
export function failure(error: LlmProviderError): FakeReply {
    return () => ({ ok: false, error });
}

/** Creates fake provider for tests with configurable capability support. */
export function createFakeProvider(name = 'fake', supports: LlmProviderSupport = FULL_SUPPORT): FakeProvider {
    let replies: FakeReply[] = [answer('ok')];
    const calls: FakeCall[] = [];
    return {
        name,
        calls,
        reply(...next) {
            replies = next;
        },
        supports,
        async generate(request, { extra }) {
            const call = { request, extra };
            calls.push(call);
            const reply = replies.length > 1 ? (replies.shift() as FakeReply) : (replies[0] as FakeReply);
            return reply(call);
        },
        async listModels() {
            return { ok: true, data: [{ id: `${name}-model` }] };
        },
    };
}
