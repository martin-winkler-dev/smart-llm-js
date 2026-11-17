// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it } from 'bun:test';
import Anthropic from '@anthropic-ai/sdk';
import type { LlmProviderRequest } from '@smart-llm/core';
import { batchTiming } from '../src/batch.js';
import { createAnthropicProvider } from '../src/index.js';
import { toMessageParams } from '../src/request.js';
import { apiError, fakeFetch, json, message, queued, type Route, type Sent, text, waitFor } from './fake.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

const call = { signal: new AbortController().signal, extra: {} };

const usage = { inputTokens: 10, outputTokens: 20, reasoningTokens: null };

const create = ['POST', '/v1/messages/batches'] as const;
const retrieve = ['GET', '/v1/messages/batches/msgbatch_1'] as const;
const cancel = ['POST', '/v1/messages/batches/msgbatch_1/cancel'] as const;
const resultsPath = ['GET', '/v1/messages/batches/msgbatch_1/results'] as const;

const timing = { ...batchTiming };

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Creates message batch object with processing status; results_url set once ended. */
function batch(status: 'in_progress' | 'canceling' | 'ended') {
    return {
        id: 'msgbatch_1',
        type: 'message_batch',
        processing_status: status,
        request_counts: { processing: status === 'ended' ? 0 : 2, succeeded: 0, errored: 0, canceled: 0, expired: 0 },
        created_at: '2026-11-08T00:00:00Z',
        expires_at: '2026-11-09T00:00:00Z',
        ended_at: status === 'ended' ? '2026-11-08T00:10:00Z' : null,
        archived_at: null,
        cancel_initiated_at: status === 'canceling' ? '2026-11-08T00:05:00Z' : null,
        results_url: status === 'ended' ? 'https://api.anthropic.com/v1/messages/batches/msgbatch_1/results' : null,
    };
}

/**
 * **Fake Message Batches API**: routes batch requests of batch `msgbatch_1`.
 *
 * - `create`: POST /v1/messages/batches (default: in_progress batch)
 * - `retrieve`: GET /v1/messages/batches/msgbatch_1 (default: ended batch)
 * - `results`: JSONL lines of GET results_url, or route answering it
 * - `cancel`: POST /v1/messages/batches/msgbatch_1/cancel (default: canceling batch)
 */
function server(routes: { create?: Route; retrieve?: Route; results?: unknown[] | Route; cancel?: Route }): Route {
    return (sent) => {
        if (sent.method === 'POST' && sent.path === '/v1/messages/batches') {
            return (routes.create ?? json(batch('in_progress')))(sent);
        }
        if (sent.method === 'POST' && sent.path === '/v1/messages/batches/msgbatch_1/cancel') {
            return (routes.cancel ?? json(batch('canceling')))(sent);
        }
        if (sent.method === 'GET' && sent.path === '/v1/messages/batches/msgbatch_1/results') {
            if (typeof routes.results === 'function') {
                return routes.results(sent);
            }
            return jsonl(routes.results ?? [])(sent);
        }
        if (sent.method === 'GET' && sent.path === '/v1/messages/batches/msgbatch_1') {
            return (routes.retrieve ?? json(batch('ended')))(sent);
        }
        return new Response('not found', { status: 404 });
    };
}

/** Returns route answering with JSONL result lines. */
function jsonl(lines: unknown[]): Route {
    const body = lines.map((line) => `${JSON.stringify(line)}\n`).join('');
    return () => new Response(body, { headers: { 'content-type': 'application/binary' } });
}

/** Returns route answering with JSONL body that fails after first chunk with error. */
function broken(first: string, error: Error): Route {
    return () =>
        new Response(
            new ReadableStream({
                start(controller) {
                    controller.enqueue(new TextEncoder().encode(first));
                    controller.error(error);
                },
            }),
            { headers: { 'content-type': 'application/binary' } },
        );
}

/** Creates succeeded batch result line with answer text. */
function succeeded(customId: string, answer: string, model = 'claude-a') {
    return { custom_id: customId, result: { type: 'succeeded', message: message([text(answer)], { model }) } };
}

/** Creates errored batch result line with Anthropic error type. */
function errored(customId: string, type: string, value: string) {
    return {
        custom_id: customId,
        result: { type: 'errored', error: { type: 'error', error: { type, message: value }, request_id: null } },
    };
}

/** Returns requests sent with method to path. */
function sentTo(sent: Sent[], method: string, path: string): Sent[] {
    return sent.filter((one) => one.method === method && one.path === path);
}

/** Creates batch request for model with user text. */
function batchRequest(model: string, content: string): LlmProviderRequest {
    return { model, messages: [{ role: 'user', content }], batch: true };
}

beforeEach(() => {
    batchTiming.msPerSecond = 1;
    batchTiming.pollMs = 1;
});

afterEach(() => {
    Object.assign(batchTiming, timing);
});

describe('Anthropic batch calls', () => {
    it('batches calls in collection window, polls until ended, maps results by custom_id', async () => {
        const fake = fakeFetch(
            server({
                retrieve: queued(json(batch('in_progress')), json(batch('in_progress')), json(batch('ended'))),
                results: [succeeded('c2', 'two', 'claude-b'), succeeded('c1', 'one', 'claude-a')],
            }),
        );
        const anthropic = createAnthropicProvider({ apiKey: 'key', fetch: fake.fetch, batchWaitSec: 20 });
        const first = batchRequest('claude-a', 'first');
        const second = batchRequest('claude-b', 'second');
        const results = await Promise.all([anthropic.generate(first, call), anthropic.generate(second, call)]);
        expect(results).toEqual([
            {
                ok: true,
                data: { text: 'one', model: 'claude-a', usage, finishReason: 'end_turn', ms: expect.any(Number) },
            },
            {
                ok: true,
                data: { text: 'two', model: 'claude-b', usage, finishReason: 'end_turn', ms: expect.any(Number) },
            },
        ]);
        const creates = sentTo(fake.sent, ...create);
        expect(creates).toHaveLength(1);
        const params = [toMessageParams(first, {}, 16_000), toMessageParams(second, {}, 16_000)];
        expect(creates[0]?.body).toEqual({
            requests: [
                { custom_id: 'c1', params: params[0]?.ok ? params[0].data : undefined },
                { custom_id: 'c2', params: params[1]?.ok ? params[1].data : undefined },
            ],
        });
        expect(creates[0]?.body).toEqual({
            requests: [
                {
                    custom_id: 'c1',
                    params: { model: 'claude-a', max_tokens: 16_000, messages: [{ role: 'user', content: 'first' }] },
                },
                {
                    custom_id: 'c2',
                    params: { model: 'claude-b', max_tokens: 16_000, messages: [{ role: 'user', content: 'second' }] },
                },
            ],
        });
        expect(creates[0]?.headers.get('authorization')).toBe('Bearer key');
        expect(creates[0]?.headers.get('x-stainless-retry-count')).toBe('0');
        expect(creates[0]?.headers.get('x-stainless-timeout')).toBe('600');
        expect(sentTo(fake.sent, ...retrieve)).toHaveLength(4);
        expect(sentTo(fake.sent, 'POST', '/v1/messages')).toHaveLength(0);
        expect(sentTo(fake.sent, ...cancel)).toHaveLength(0);
    });

    it('maps errored, canceled, expired and missing results', async () => {
        const fake = fakeFetch(
            server({
                results: [
                    errored('c1', 'invalid_request_error', 'bad param'),
                    errored('c2', 'rate_limit_error', 'slow down'),
                    errored('c3', 'overloaded_error', 'Overloaded'),
                    errored('c4', 'api_error', 'Internal error'),
                    errored('c5', 'authentication_error', 'invalid key'),
                    { custom_id: 'c6', result: { type: 'canceled' } },
                    { custom_id: 'c7', result: { type: 'expired' } },
                    errored('c8', 'timeout_error', 'Request timed out'),
                ],
            }),
        );
        const anthropic = createAnthropicProvider({ apiKey: 'key', fetch: fake.fetch, batchWaitSec: 20 });
        const results = await Promise.all(
            Array.from({ length: 9 }, (_, index) => anthropic.generate(batchRequest('claude-a', `q${index}`), call)),
        );
        expect(results).toMatchObject([
            {
                ok: false,
                error: {
                    code: 'http',
                    status: 400,
                    retryable: false,
                    message: 'Batch request failed: invalid_request_error: bad param',
                },
            },
            {
                ok: false,
                error: {
                    code: 'rate_limited',
                    status: 429,
                    message: 'Batch request failed: rate_limit_error: slow down',
                },
            },
            {
                ok: false,
                error: {
                    code: 'http',
                    status: 529,
                    retryable: true,
                    message: 'Batch request failed: overloaded_error: Overloaded',
                },
            },
            {
                ok: false,
                error: {
                    code: 'http',
                    status: 500,
                    retryable: true,
                    message: 'Batch request failed: api_error: Internal error',
                },
            },
            {
                ok: false,
                error: {
                    code: 'http',
                    status: 401,
                    retryable: false,
                    message: 'Batch request failed: authentication_error: invalid key',
                },
            },
            { ok: false, error: { code: 'aborted', retryable: false } },
            {
                ok: false,
                error: { code: 'timeout', retryable: true, message: 'Batch request expired before processing' },
            },
            {
                ok: false,
                error: {
                    code: 'http',
                    status: 504,
                    retryable: true,
                    message: 'Batch request failed: timeout_error: Request timed out',
                },
            },
            { ok: false, error: { code: 'unexpected' } },
        ]);
        expect(sentTo(fake.sent, ...create)).toHaveLength(1);
    });

    it('maps every errored type to its HTTP status, none for unknown types', async () => {
        const types = ['billing_error', 'permission_error', 'not_found_error', 'other_error'];
        const fake = fakeFetch(
            server({ results: types.map((type, index) => errored(`c${index + 1}`, type, 'failed')) }),
        );
        const anthropic = createAnthropicProvider({ apiKey: 'key', fetch: fake.fetch, batchWaitSec: 20 });
        const results = await Promise.all(
            types.map((_, index) => anthropic.generate(batchRequest('claude-a', `q${index}`), call)),
        );
        expect(results.map((result) => (result.ok ? undefined : result.error.status))).toEqual([
            402,
            403,
            404,
            undefined,
        ]);
        expect(results.every((result) => !result.ok && result.error.code === 'http' && !result.error.retryable)).toBe(
            true,
        );
    });

    it('fails ended batch missing results_url as unexpected without repolling', async () => {
        const fake = fakeFetch(server({ retrieve: json({ ...batch('ended'), results_url: null }) }));
        const anthropic = createAnthropicProvider({ apiKey: 'key', fetch: fake.fetch, batchWaitSec: 5 });
        expect(await anthropic.generate(batchRequest('claude-a', 'a'), call)).toEqual({
            ok: false,
            error: { code: 'unexpected', message: 'Batch msgbatch_1 ended without results_url', retryable: false },
        });
        expect(sentTo(fake.sent, ...retrieve)).toHaveLength(1);
        expect(sentTo(fake.sent, ...resultsPath)).toHaveLength(0);
    });

    it('polls again after a dropped or truncated results download', async () => {
        const line = `${JSON.stringify(succeeded('c1', 'one'))}\n`;
        const downloads: Route[] = [
            broken(line, new TypeError('terminated')),
            () => new Response(`${line}{"custom_id":"c2","res`, { headers: { 'content-type': 'application/binary' } }),
        ];
        for (const download of downloads) {
            const fake = fakeFetch(
                server({ results: queued(download, jsonl([succeeded('c1', 'one'), succeeded('c2', 'two')])) }),
            );
            const anthropic = createAnthropicProvider({ apiKey: 'key', fetch: fake.fetch, batchWaitSec: 5 });
            const results = await Promise.all([
                anthropic.generate(batchRequest('claude-a', 'a'), call),
                anthropic.generate(batchRequest('claude-a', 'b'), call),
            ]);
            expect(results).toMatchObject([
                { ok: true, data: { text: 'one' } },
                { ok: true, data: { text: 'two' } },
            ]);
            expect(sentTo(fake.sent, ...resultsPath)).toHaveLength(2);
        }
    });

    it('fails with non-retryable unexpected after 3 unparsable results downloads', async () => {
        const truncated: Route = () =>
            new Response('{"custom_id":"c1","res', { headers: { 'content-type': 'application/binary' } });
        const fake = fakeFetch(server({ results: truncated }));
        const anthropic = createAnthropicProvider({ apiKey: 'key', fetch: fake.fetch, batchWaitSec: 5 });
        const results = await Promise.all([
            anthropic.generate(batchRequest('claude-a', 'a'), call),
            anthropic.generate(batchRequest('claude-a', 'b'), call),
        ]);
        const error = {
            code: 'unexpected',
            retryable: false,
            message: expect.stringMatching(/^Batch msgbatch_1 results download failed: .* \(3 attempts\)$/),
        };
        expect(results).toMatchObject([
            { ok: false, error },
            { ok: false, error },
        ]);
        expect(sentTo(fake.sent, ...resultsPath)).toHaveLength(3);
    });

    it('sends the timeout of an existing Anthropic client with every batch request', async () => {
        const fake = fakeFetch(server({ results: [succeeded('c1', 'one')] }));
        const client = new Anthropic({ apiKey: 'key', fetch: fake.fetch, timeout: 5000 });
        const anthropic = createAnthropicProvider(client);
        expect(await anthropic.generate(batchRequest('claude-a', 'a'), call)).toMatchObject({ ok: true });
        expect(fake.sent.length).toBeGreaterThanOrEqual(3);
        expect(fake.sent.map((sent) => sent.headers.get('x-stainless-timeout'))).toEqual(fake.sent.map(() => '5'));
    });

    it('settles every call when a succeeded result line is malformed', async () => {
        const fake = fakeFetch(
            server({
                results: [
                    { custom_id: 'c1', result: { type: 'succeeded', message: null } },
                    { custom_id: 'c2', result: null },
                    succeeded('c3', 'three'),
                ],
            }),
        );
        const anthropic = createAnthropicProvider({ apiKey: 'key', fetch: fake.fetch, batchWaitSec: 5 });
        const results = await Promise.all(
            ['a', 'b', 'c'].map((content) => anthropic.generate(batchRequest('claude-a', content), call)),
        );
        expect(results).toMatchObject([
            { ok: false, error: { code: 'unexpected', message: 'The response is not an Anthropic message' } },
            {
                ok: false,
                error: {
                    code: 'unexpected',
                    retryable: false,
                    message: 'Batch msgbatch_1 result for this call is not an object',
                },
            },
            { ok: true, data: { text: 'three' } },
        ]);
    });

    it('propagates submit error to all batch calls without retries or polling', async () => {
        const fake = fakeFetch(server({ create: apiError(400, 'invalid_request_error', 'requests: too many') }));
        const anthropic = createAnthropicProvider({ apiKey: 'key', fetch: fake.fetch, batchWaitSec: 20 });
        const results = await Promise.all([
            anthropic.generate(batchRequest('claude-a', 'a'), call),
            anthropic.generate(batchRequest('claude-a', 'b'), call),
        ]);
        const error = { code: 'http', status: 400, retryable: false, message: 'HTTP 400: requests: too many' };
        expect(results).toEqual([
            { ok: false, error },
            { ok: false, error },
        ] as never);
        expect(fake.sent).toHaveLength(1);
    });

    it('retries polling on retryable retrieve error and fails on non-retryable one', async () => {
        const retried = fakeFetch(
            server({
                retrieve: queued(apiError(503, 'api_error', 'unavailable'), json(batch('ended'))),
                results: [succeeded('c1', 'one')],
            }),
        );
        const anthropic = createAnthropicProvider({ apiKey: 'key', fetch: retried.fetch, batchWaitSec: 5 });
        expect(await anthropic.generate(batchRequest('claude-a', 'a'), call)).toMatchObject({
            ok: true,
            data: { text: 'one' },
        });
        const failed = fakeFetch(server({ retrieve: apiError(404, 'not_found_error', 'no batch') }));
        const failing = createAnthropicProvider({ apiKey: 'key', fetch: failed.fetch, batchWaitSec: 5 });
        expect(await failing.generate(batchRequest('claude-a', 'a'), call)).toMatchObject({
            ok: false,
            error: { code: 'http', status: 404, retryable: false, message: 'HTTP 404: no batch' },
        });
        expect(sentTo(failed.sent, ...retrieve)).toHaveLength(1);
    });

    it('fails with timeout and cancels the batch after maxPollMs', async () => {
        batchTiming.maxPollMs = 30;
        const fake = fakeFetch(server({ retrieve: json(batch('in_progress')) }));
        const anthropic = createAnthropicProvider({ apiKey: 'key', fetch: fake.fetch, batchWaitSec: 5 });
        expect(await anthropic.generate(batchRequest('claude-a', 'a'), call)).toMatchObject({
            ok: false,
            error: { code: 'timeout', retryable: false },
        });
        await waitFor(() => sentTo(fake.sent, ...cancel).length === 1);
    });
});

describe('Anthropic batch aborts', () => {
    it('returns aborted without request when signal is already aborted', async () => {
        const fake = fakeFetch(server({}));
        const anthropic = createAnthropicProvider({ apiKey: 'key', fetch: fake.fetch, batchWaitSec: 5 });
        const controller = new AbortController();
        controller.abort();
        expect(
            await anthropic.generate(batchRequest('claude-a', 'a'), { signal: controller.signal, extra: {} }),
        ).toMatchObject({ ok: false, error: { code: 'aborted', retryable: false } });
        await new Promise((resolve) => setTimeout(resolve, 30));
        expect(fake.sent).toHaveLength(0);
    });

    it('removes a call aborted in the collection window from the batch', async () => {
        const fake = fakeFetch(server({ results: [succeeded('c2', 'two')] }));
        const anthropic = createAnthropicProvider({ apiKey: 'key', fetch: fake.fetch, batchWaitSec: 20 });
        const controller = new AbortController();
        const first = anthropic.generate(batchRequest('claude-a', 'a'), { signal: controller.signal, extra: {} });
        const second = anthropic.generate(batchRequest('claude-a', 'b'), call);
        controller.abort();
        expect(await first).toMatchObject({ ok: false, error: { code: 'aborted', retryable: false } });
        expect(await second).toMatchObject({ ok: true, data: { text: 'two' } });
        const body = sentTo(fake.sent, ...create)[0]?.body as { requests: { custom_id: string }[] } | undefined;
        const requests = body?.requests ?? [];
        expect(requests.map((request) => request.custom_id)).toEqual(['c2']);
    });

    it('submits no batch when every call left the collection window', async () => {
        const fake = fakeFetch(server({}));
        const anthropic = createAnthropicProvider({ apiKey: 'key', fetch: fake.fetch, batchWaitSec: 5 });
        const controller = new AbortController();
        const pending = anthropic.generate(batchRequest('claude-a', 'a'), { signal: controller.signal, extra: {} });
        controller.abort();
        expect(await pending).toMatchObject({ ok: false, error: { code: 'aborted' } });
        await new Promise((resolve) => setTimeout(resolve, 30));
        expect(fake.sent).toHaveLength(0);
    });

    it('aborts single submitted call without canceling batch while others wait', async () => {
        let ended = false;
        const fake = fakeFetch(
            server({
                retrieve: (sent) => json(batch(ended ? 'ended' : 'in_progress'))(sent),
                results: [succeeded('c1', 'one'), succeeded('c2', 'two')],
            }),
        );
        const anthropic = createAnthropicProvider({ apiKey: 'key', fetch: fake.fetch, batchWaitSec: 5 });
        const controller = new AbortController();
        const first = anthropic.generate(batchRequest('claude-a', 'a'), { signal: controller.signal, extra: {} });
        const second = anthropic.generate(batchRequest('claude-a', 'b'), call);
        await waitFor(() => sentTo(fake.sent, ...retrieve).length > 0);
        controller.abort();
        expect(await first).toMatchObject({ ok: false, error: { code: 'aborted', retryable: false } });
        ended = true;
        expect(await second).toMatchObject({ ok: true, data: { text: 'two' } });
        expect(sentTo(fake.sent, ...cancel)).toHaveLength(0);
    });

    it('cancels batch once all submitted calls abort, ignoring cancel errors', async () => {
        const fake = fakeFetch(
            server({ retrieve: json(batch('in_progress')), cancel: apiError(500, 'api_error', 'Internal error') }),
        );
        const anthropic = createAnthropicProvider({ apiKey: 'key', fetch: fake.fetch, batchWaitSec: 5 });
        const controllers = [new AbortController(), new AbortController()];
        const pending = controllers.map((controller, index) =>
            anthropic.generate(batchRequest('claude-a', `q${index}`), { signal: controller.signal, extra: {} }),
        );
        await waitFor(() => sentTo(fake.sent, ...create).length === 1);
        controllers[0]?.abort();
        expect(await pending[0]).toMatchObject({ ok: false, error: { code: 'aborted' } });
        await new Promise((resolve) => setTimeout(resolve, 10));
        expect(sentTo(fake.sent, ...cancel)).toHaveLength(0);
        controllers[1]?.abort();
        expect(await pending[1]).toMatchObject({ ok: false, error: { code: 'aborted' } });
        await waitFor(() => sentTo(fake.sent, ...cancel).length === 1);
        await new Promise((resolve) => setTimeout(resolve, 10));
        expect(sentTo(fake.sent, ...cancel)).toHaveLength(1);
    });
});
