// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'bun:test';
import { assistantMessage, toolMessage } from '../src/messages.js';
import type { LlmToolCall } from '../src/types.js';

const call: LlmToolCall = { id: 'c1', type: 'function', function: { name: 'f', arguments: '{}' } };

describe('assistantMessage', () => {
    it('keeps the text, and adds tool calls and reasoning only when the answer has them', () => {
        expect(assistantMessage({ text: 'hi' })).toEqual({ role: 'assistant', content: 'hi' });
        expect(assistantMessage({ text: '', toolCalls: [] })).toEqual({ role: 'assistant', content: '' });
        expect(assistantMessage({ text: '', toolCalls: [call], reasoning: 'r' })).toEqual({
            role: 'assistant',
            content: null,
            tool_calls: [call],
            reasoning: 'r',
        });
        expect(assistantMessage({ text: 'let me check', toolCalls: [call] })).toEqual({
            role: 'assistant',
            content: 'let me check',
            tool_calls: [call],
        });
    });
});

describe('toolMessage', () => {
    it('sends a string as it is and anything else as JSON', () => {
        expect(toolMessage(call, 'sunny')).toEqual({ role: 'tool', tool_call_id: 'c1', content: 'sunny' });
        expect(toolMessage(call, { sky: 'sunny' })).toEqual({
            role: 'tool',
            tool_call_id: 'c1',
            content: '{"sky":"sunny"}',
        });
        expect(toolMessage(call, undefined)).toEqual({ role: 'tool', tool_call_id: 'c1', content: 'null' });
    });
});
