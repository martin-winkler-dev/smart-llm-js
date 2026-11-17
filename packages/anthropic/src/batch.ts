// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { Anthropic } from '@anthropic-ai/sdk';
import type { LlmProviderError, LlmProviderOutput, Result } from '@smart-llm/core';
import { isPlainObject } from './request.js';
import { readMessage, toProviderError } from './response.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Result outcome of batch call. */
type CallResult = Result<LlmProviderOutput, LlmProviderError>;

/** In-flight batch call awaiting settlement. */
type Call = {
    key: string;
    params: Anthropic.MessageCreateParamsNonStreaming;
    bytes: number;
    addedAt: number;
    settled: boolean;
    settle(result: CallResult): void;
    leave?: () => void;
};

/** Batch calls collected within collection window. */
type Group = { calls: Call[]; timer: ReturnType<typeof setTimeout> };

/** Submitted batch tracking open calls and polling state. */
type Job = {
    calls: Map<string, Call>;
    id?: string;
    done: boolean;
    submittedAt?: number;
    timer?: ReturnType<typeof setTimeout>;
    /** Consecutive results downloads failed by stream drop or unparsable content. */
    resultsFailures: number;
};

/** Batch call manager. */
export type BatchCalls = {
    /** Adds Messages params to batch queue; resolves to individual result. */
    add(params: Anthropic.MessageCreateParamsNonStreaming, signal?: AbortSignal): Promise<CallResult>;
};

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/**
 * Timing of batch polling and collection window.
 * @internal - exported for testing only
 */
export const batchTiming = { pollMs: 60_000, maxPollMs: 25 * 3_600_000, msPerSecond: 1000 };

/** Maximum requests per Anthropic message batch. */
const MAX_BATCH_REQUESTS = 100_000;

/** Maximum JSON bytes per message batch (below 256 MB API limit). */
const MAX_BATCH_BYTES = 250_000_000;

/** Consecutive stream/parse failures of results download before giving up. */
const MAX_RESULTS_FAILURES = 3;

/** Milliseconds per hour. */
const MS_PER_HOUR = 3_600_000;

/** Errored result types worth retrying. */
const RETRYABLE_ERROR_TYPES: ReadonlySet<string> = new Set(['overloaded_error', 'api_error', 'timeout_error']);

/** HTTP status per errored result type. */
const ERROR_TYPE_STATUSES: ReadonlyMap<string, number> = new Map([
    ['invalid_request_error', 400],
    ['authentication_error', 401],
    ['billing_error', 402],
    ['permission_error', 403],
    ['not_found_error', 404],
    ['rate_limit_error', 429],
    ['timeout_error', 504],
    ['api_error', 500],
    ['overloaded_error', 529],
]);

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Manages Message Batches collection and polling lifecycle**.
 *
 * - one collection window for all models (batches may mix models); first call starts `waitSec` timer
 * - flushes into batches of at most 100,000 requests and ~250 MB JSON
 * - polls batch until `ended`, then reads results by `custom_id`; cancels after `maxPollMs`
 * - aborted call leaves its group or batch; batch without open calls gets canceled
 */
export function createBatchCalls(client: Anthropic, waitSec: number, timeoutMs: number): BatchCalls {
    const requestOptions = { timeout: timeoutMs, maxRetries: 0 };
    let group: Group | undefined;
    let nextKey = 1;

    /** Submits collected calls (several batches when over request or byte limit). */
    function flush(flushed: Group): void {
        if (group === flushed) {
            group = undefined;
        }
        let chunk: Call[] = [];
        let bytes = 0;
        for (const call of flushed.calls) {
            if (chunk.length > 0 && (chunk.length >= MAX_BATCH_REQUESTS || bytes + call.bytes > MAX_BATCH_BYTES)) {
                void submitJob(chunk);
                chunk = [];
                bytes = 0;
            }
            chunk.push(call);
            bytes += call.bytes;
        }
        if (chunk.length > 0) {
            void submitJob(chunk);
        }
    }

    /** Submits one batch and polls it; every outcome settles its calls. */
    async function submitJob(calls: Call[]): Promise<void> {
        const job: Job = { calls: new Map(calls.map((call) => [call.key, call])), done: false, resultsFailures: 0 };
        for (const call of calls) {
            call.leave = () => onCallLeft(job);
        }
        try {
            // biome-ignore lint/style/useNamingConvention: Anthropic field name
            const requests = calls.map((call) => ({ custom_id: call.key, params: call.params }));
            const batch = await client.messages.batches.create({ requests }, requestOptions);
            job.id = batch.id;
            job.submittedAt = Date.now();
            if (job.done) {
                cancelBatch(batch.id);
                return;
            }
            schedulePoll(job);
        } catch (error) {
            const failed = toProviderError(error);
            finish(job, () => ({ ok: false, error: failed }));
        }
    }

    /** Polls batch after batchTiming.pollMs (or retryAfterMs); cancels with timeout after maxPollMs. */
    function schedulePoll(job: Job, retryAfterMs = 0): void {
        const leftMs = batchTiming.maxPollMs - (Date.now() - (job.submittedAt ?? 0));
        if (leftMs <= 0) {
            const hours = Math.round(batchTiming.maxPollMs / MS_PER_HOUR);
            const message = `Batch ${job.id} not ended after ${hours} h; canceled`;
            finish(job, () => ({ ok: false, error: { code: 'timeout', message, retryable: false } }));
            cancelBatch(job.id as string);
            return;
        }
        job.timer = setTimeout(
            () => {
                void poll(job);
            },
            Math.min(Math.max(batchTiming.pollMs, retryAfterMs), leftMs),
        );
    }

    /** One poll; reads results of ended batch (none without `results_url`: `unexpected`), else polls again. */
    async function poll(job: Job): Promise<void> {
        if (job.done) {
            return;
        }
        const id = job.id as string;
        let read: Result<Map<string, Anthropic.Messages.MessageBatchResult>, LlmProviderError>;
        try {
            const batch = await client.messages.batches.retrieve(id, null, requestOptions);
            if (job.done) {
                return;
            }
            if (batch.processing_status !== 'ended') {
                schedulePoll(job);
                return;
            }
            if (typeof batch.results_url !== 'string' || batch.results_url === '') {
                const message = `Batch ${id} ended without results_url`;
                finish(job, () => ({ ok: false, error: { code: 'unexpected', message, retryable: false } }));
                return;
            }
            read = await readResults(job, id);
        } catch (error) {
            read = { ok: false, error: toProviderError(error) };
        }
        if (job.done) {
            return;
        }
        if (read.ok) {
            const results = read.data;
            finish(job, (call) => toCallResult(results.get(call.key), call, id));
        } else if (read.error.retryable) {
            schedulePoll(job, read.error.retryAfterMs);
        } else {
            const failed = read.error;
            finish(job, () => ({ ok: false, error: failed }));
        }
    }

    /**
     * **Downloads results of ended batch by `custom_id`**.
     *
     * - stream drop or unparsable JSONL line → retryable `network` (polling continues)
     * - `MAX_RESULTS_FAILURES` such failures in a row → non-retryable `unexpected`
     * - HTTP and connection errors mapped as usual
     */
    async function readResults(
        job: Job,
        id: string,
    ): Promise<Result<Map<string, Anthropic.Messages.MessageBatchResult>, LlmProviderError>> {
        const results = new Map<string, Anthropic.Messages.MessageBatchResult>();
        try {
            for await (const entry of await client.messages.batches.results(id, undefined, requestOptions)) {
                const customId: unknown = (entry as Partial<typeof entry> | null)?.custom_id;
                if (typeof customId === 'string') {
                    results.set(customId, entry.result);
                }
            }
            return { ok: true, data: results };
        } catch (error) {
            const failed = toProviderError(error);
            if (failed.code !== 'invalid_request' && failed.code !== 'unexpected') {
                job.resultsFailures = 0;
                return { ok: false, error: failed };
            }
            job.resultsFailures++;
            const message = `Batch ${id} results download failed: ${failed.message}`;
            if (job.resultsFailures >= MAX_RESULTS_FAILURES) {
                const final = `${message} (${job.resultsFailures} attempts)`;
                return { ok: false, error: { code: 'unexpected', message: final, retryable: false } };
            }
            return { ok: false, error: { code: 'network', message, retryable: true } };
        }
    }

    /** Settles every open call of job and ends its polling; unreadable result → `unexpected` for that call only. */
    function finish(job: Job, resultOf: (call: Call) => CallResult): void {
        job.done = true;
        clearTimeout(job.timer);
        for (const call of job.calls.values()) {
            let result: CallResult;
            try {
                result = resultOf(call);
            } catch (error) {
                const message = `Unreadable batch result: ${error instanceof Error ? error.message : String(error)}`;
                result = { ok: false, error: { code: 'unexpected', message, retryable: false } };
            }
            call.settle(result);
        }
    }

    /** After abort: job whose calls are all settled stops polling and gets canceled. */
    function onCallLeft(job: Job): void {
        if (job.done || [...job.calls.values()].some((call) => !call.settled)) {
            return;
        }
        job.done = true;
        clearTimeout(job.timer);
        if (job.id !== undefined) {
            cancelBatch(job.id);
        }
    }

    /** Cancels batch, ignoring errors. */
    function cancelBatch(id: string): void {
        try {
            void Promise.resolve(client.messages.batches.cancel(id, null, requestOptions)).catch(() => undefined);
        } catch {
            // cancel is best effort
        }
    }

    return {
        add(params, signal) {
            if (signal?.aborted) {
                return Promise.resolve(aborted());
            }
            const key = `c${nextKey++}`;
            // biome-ignore lint/style/useNamingConvention: Anthropic field name
            const bytes = Buffer.byteLength(JSON.stringify({ custom_id: key, params }));
            if (bytes > MAX_BATCH_BYTES) {
                const message = `Request of ${bytes} bytes exceeds the batch limit of ${MAX_BATCH_BYTES} bytes`;
                return Promise.resolve({ ok: false, error: { code: 'invalid_request', message, retryable: false } });
            }
            return new Promise<CallResult>((resolve) => {
                const call: Call = {
                    key,
                    params,
                    bytes,
                    addedAt: Date.now(),
                    settled: false,
                    settle(result) {
                        if (call.settled) {
                            return;
                        }
                        call.settled = true;
                        signal?.removeEventListener('abort', onAbort);
                        resolve(result);
                    },
                };
                if (group === undefined) {
                    const created: Group = {
                        calls: [],
                        timer: setTimeout(() => flush(created), waitSec * batchTiming.msPerSecond),
                    };
                    group = created;
                }
                const joined = group;
                joined.calls.push(call);
                call.leave = () => {
                    joined.calls.splice(joined.calls.indexOf(call), 1);
                    if (joined.calls.length === 0 && group === joined) {
                        clearTimeout(joined.timer);
                        group = undefined;
                    }
                };
                function onAbort(): void {
                    call.settle(aborted());
                    call.leave?.();
                }
                signal?.addEventListener('abort', onAbort, { once: true });
            });
        },
    };
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/** Returns aborted batch call result. */
function aborted(): CallResult {
    return { ok: false, error: { code: 'aborted', message: 'Batch call aborted', retryable: false } };
}

/** Maps individual batch result to call result. */
function toCallResult(result: Anthropic.Messages.MessageBatchResult | undefined, call: Call, id: string): CallResult {
    if (result === undefined) {
        const message = `No result for this call in batch ${id}`;
        return { ok: false, error: { code: 'unexpected', message, retryable: false } };
    }
    if (!isPlainObject(result)) {
        const message = `Batch ${id} result for this call is not an object`;
        return { ok: false, error: { code: 'unexpected', message, retryable: false } };
    }
    switch (result.type) {
        case 'succeeded':
            return readMessage(result.message, call.params.model, Date.now() - call.addedAt);
        case 'errored':
            return { ok: false, error: toErroredResultError(result.error) };
        case 'canceled':
            return { ok: false, error: { code: 'aborted', message: 'Batch request canceled', retryable: false } };
        case 'expired':
            return {
                ok: false,
                error: { code: 'timeout', message: 'Batch request expired before processing', retryable: true },
            };
        default:
            return {
                ok: false,
                error: { code: 'unexpected', message: 'Unknown batch result type', retryable: false },
            };
    }
}

/**
 * **Maps errored batch result to provider error**.
 *
 * - `status` per error type (e.g. 400, 401, 529); none for unknown types
 * - `rate_limit_error` → `rate_limited`; others → `http`
 * - `overloaded_error`, `api_error`, `timeout_error`, `rate_limit_error` retryable
 */
function toErroredResultError(
    response: Anthropic.Messages.MessageBatchErroredResult['error'] | undefined,
): LlmProviderError {
    const type = response?.error?.type ?? 'unknown_error';
    const message = `Batch request failed: ${type}: ${response?.error?.message ?? ''}`;
    const status = ERROR_TYPE_STATUSES.get(type);
    const withStatus = status === undefined ? {} : { status };
    if (type === 'rate_limit_error') {
        return { code: 'rate_limited', message, retryable: true, ...withStatus };
    }
    return { code: 'http', message, retryable: RETRYABLE_ERROR_TYPES.has(type), ...withStatus };
}
