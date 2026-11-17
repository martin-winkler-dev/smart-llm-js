// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'bun:test';
import { assistantMessage, toolMessage } from '../src/messages.js';
import type { LlmToolCall } from '../src/types.js';

const call: LlmToolCall = { id: 'c1', type: 'function', function: { name: 'f', arguments: '{}' } };

describe('assistantMessage', () => {
    it('preserves text and includes toolCalls and reasoning when present', () => {
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
    it('keeps strings as is and serializes other values as JSON', () => {
        expect(toolMessage(call, 'sunny')).toEqual({ role: 'tool', tool_call_id: 'c1', content: 'sunny' });
        expect(toolMessage(call, { sky: 'sunny' })).toEqual({
            role: 'tool',
            tool_call_id: 'c1',
            content: '{"sky":"sunny"}',
        });
        expect(toolMessage(call, undefined)).toEqual({ role: 'tool', tool_call_id: 'c1', content: 'null' });
    });

    it('serializes BigInt as decimal string and circular references as [Circular]', () => {
        const circular: Record<string, unknown> = { name: 'x' };
        circular.self = circular;
        circular.list = [circular];
        expect(toolMessage(call, 1n).content).toBe('"1"');
        expect(toolMessage(call, { id: 1n, name: 'x' }).content).toBe('{"id":"1","name":"x"}');
        expect(toolMessage(call, circular).content).toBe('{"name":"x","self":"[Circular]","list":["[Circular]"]}');
        const shared = { v: 1 };
        expect(toolMessage(call, { a: shared, b: [shared] }).content).toBe('{"a":{"v":1},"b":[{"v":1}]}');
        expect(toolMessage(call, () => 1).content).toBe('null');
        expect(toolMessage(call, Symbol('s')).content).toBe('null');
    });

    it('never throws: falls back to String(result), then to a placeholder, when JSON fails', () => {
        const throwing = {
            toJSON() {
                throw new Error('no');
            },
            toString() {
                return 'custom';
            },
        };
        expect(toolMessage(call, throwing).content).toBe('custom');
        const bare = Object.create(null) as Record<string, unknown>;
        bare.toJSON = () => {
            throw new Error('no');
        };
        expect(toolMessage(call, bare)).toEqual({
            role: 'tool',
            tool_call_id: 'c1',
            content: '[unserializable tool result]',
        });
    });
});
