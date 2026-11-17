// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type Anthropic from '@anthropic-ai/sdk';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** HTTP request recorded by fake fetch; body parsed from JSON. */
export type Sent = {
    url: string;
    path: string;
    method: string;
    headers: Headers;
    body: unknown;
    signal: AbortSignal | null | undefined;
};

/** Answers one recorded request. */
export type Route = (sent: Sent) => Response | Promise<Response>;

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Usage of fake messages: 10 input and 20 output tokens, no cache or thinking tokens. */
export const USAGE: Anthropic.Usage = {
    cache_creation: null,
    cache_creation_input_tokens: null,
    cache_read_input_tokens: null,
    inference_geo: null,
    input_tokens: 10,
    output_tokens: 20,
    output_tokens_details: null,
    server_tool_use: null,
    service_tier: null,
    speed: null,
};

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Creates fake fetch recording requests**: answers each request through `route`.
 *
 * - `path`: URL path without origin and query
 * - `body`: parsed JSON of string bodies, else undefined
 */
export function fakeFetch(route: Route): { fetch: typeof fetch; sent: Sent[] } {
    const sent: Sent[] = [];
    async function fetcher(input: string | URL | Request, init: RequestInit = {}): Promise<Response> {
        const url = input instanceof Request ? input.url : String(input);
        const body = typeof init.body === 'string' ? JSON.parse(init.body) : undefined;
        const request: Sent = {
            url,
            path: new URL(url).pathname,
            method: (init.method ?? 'GET').toUpperCase(),
            headers: new Headers(init.headers),
            body,
            signal: init.signal,
        };
        sent.push(request);
        return route(request);
    }
    return { fetch: fetcher as typeof fetch, sent };
}

/** Returns route answering requests in order; last answer repeats. */
export function queued(...answers: Route[]): Route {
    return (sent) => {
        const next = answers.length > 1 ? answers.shift() : answers[0];
        return (next as Route)(sent);
    };
}

/** Returns route answering with JSON payload. */
export function json(body: unknown, status = 200, headers: Record<string, string> = {}): Route {
    return () =>
        new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', ...headers } });
}

/** Returns route answering with Anthropic error body `{type:'error', error:{type, message}}`. */
export function apiError(status: number, type: string, message: string, headers: Record<string, string> = {}): Route {
    return json({ type: 'error', error: { type, message }, request_id: 'req_1' }, status, headers);
}

/** Returns route waiting until request is aborted, then rejecting with abort reason. */
export function hanging(): Route {
    return (sent) =>
        new Promise<Response>((_resolve, reject) => {
            sent.signal?.addEventListener('abort', () => reject(sent.signal?.reason), { once: true });
        });
}

/** Creates Anthropic message with content blocks; fields overridable. */
export function message(content: unknown[], overrides: Partial<Anthropic.Message> = {}): Anthropic.Message {
    return {
        id: 'msg_1',
        container: null,
        content: content as Anthropic.ContentBlock[],
        diagnostics: null,
        model: 'claude-served',
        role: 'assistant',
        stop_details: null,
        stop_reason: 'end_turn',
        stop_sequence: null,
        type: 'message',
        usage: USAGE,
        ...overrides,
    };
}

/** Creates text content block with optional citations. */
export function text(value: string, citations: unknown[] | null = null): Anthropic.TextBlock {
    return { type: 'text', text: value, citations: citations as Anthropic.TextCitation[] | null };
}

/** Waits until condition holds, polling every millisecond for up to 2 s. */
export async function waitFor(condition: () => boolean): Promise<void> {
    const until = Date.now() + 2000;
    while (!condition()) {
        if (Date.now() > until) {
            throw new Error('Condition not reached within 2 s');
        }
        await new Promise((resolve) => setTimeout(resolve, 1));
    }
}
