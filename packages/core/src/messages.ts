// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { LlmMessage, LlmOutput, LlmToolCall } from './types.js';

// -------------------------------------------------------------------
// --- 1. Exported Functions
// -------------------------------------------------------------------

/**
 * **Formats LLM output as assistant message**: preserves text, tool calls, and reasoning.
 *
 * - sets `content` to `null` when output contains tool calls without text
 * - preserves reasoning for providers requiring reasoning history
 */
export function assistantMessage(output: Pick<LlmOutput, 'text' | 'toolCalls' | 'reasoning'>): LlmMessage {
    const toolCalls = output.toolCalls ?? [];
    const content = output.text === '' && toolCalls.length > 0 ? null : output.text;
    return {
        role: 'assistant',
        content,
        // biome-ignore lint/style/useNamingConvention: OpenAI field name
        ...(toolCalls.length === 0 ? {} : { tool_calls: toolCalls }),
        ...(output.reasoning === undefined ? {} : { reasoning: output.reasoning }),
    };
}

/**
 * **Formats tool call result as tool message**: never throws.
 *
 * - strings unchanged; other values as JSON (top-level `undefined`, function or symbol as `null`)
 * - BigInt as decimal string; circular reference as `"[Circular]"`
 * - fallback when JSON fails (e.g. throwing `toJSON`): `String(result)`, else `[unserializable tool result]`
 */
export function toolMessage(call: Pick<LlmToolCall, 'id'>, result: unknown): LlmMessage {
    const content = typeof result === 'string' ? result : serialize(result);
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    return { role: 'tool', tool_call_id: call.id, content };
}

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Serializes value to JSON; falls back to `String(value)`, then to fixed placeholder. */
function serialize(value: unknown): string {
    try {
        return JSON.stringify(value, safeReplacer()) ?? 'null';
    } catch {
        try {
            return String(value);
        } catch {
            return '[unserializable tool result]';
        }
    }
}

/**
 * **Creates JSON replacer for BigInt and circular references.**
 *
 * - BigInt: decimal string
 * - object on current ancestor path: `"[Circular]"`; same object in sibling branches serialized each time
 */
function safeReplacer(): (this: unknown, key: string, value: unknown) => unknown {
    // `this` is the holder of `value`; ancestors[i] is the holder at depth i
    const ancestors: unknown[] = [];
    return function replace(this: unknown, _key: string, value: unknown): unknown {
        if (typeof value === 'bigint') {
            return value.toString();
        }
        if (typeof value !== 'object' || value === null) {
            return value;
        }
        // drop finished branches: keep path up to and including current holder
        const depth = ancestors.lastIndexOf(this);
        ancestors.length = depth + 1;
        if (ancestors.includes(value)) {
            return '[Circular]';
        }
        ancestors.push(value);
        return value;
    };
}
