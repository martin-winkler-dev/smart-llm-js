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

/** Formats tool call result as tool message; passes strings unchanged and serializes other values to JSON. */
export function toolMessage(call: Pick<LlmToolCall, 'id'>, result: unknown): LlmMessage {
    const content = typeof result === 'string' ? result : (JSON.stringify(result) ?? 'null');
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    return { role: 'tool', tool_call_id: call.id, content };
}
