// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import {
    type Bucket,
    createConcurrencyBucket,
    createRPD,
    createRPM,
    createSlidingWindow,
    createTPD,
    createTPM,
} from 'smart-limit';
import { checkKeys } from './request.js';
import type { LlmLimiterOptions, LlmLimits, LlmProviderRequest } from './types.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Members of `LlmLimits`. */
const LIMIT_KEYS: ReadonlySet<string> = new Set([
    'concurrency',
    'rps',
    'rpm',
    'rpd',
    'tpm',
    'tpd',
    'intervalMs',
    'buckets',
]);

/** Integer limits (lowest value 1). */
const INTEGER_LIMITS = ['concurrency', 'rps', 'rpm', 'rpd', 'tpm', 'tpd', 'intervalMs'] as const;

/** Members of `LlmLimiterOptions` (integers from 0). */
const LIMITER_KEYS: ReadonlySet<string> = new Set(['maxRetries', 'retryDelayMs', 'maxRetryDelayMs', 'maxWaitMs']);

/** Milliseconds per second for rps window. */
const MS_PER_SECOND = 1000;

/** Estimated input tokens booked per image; settled to reported usage afterwards. */
const IMAGE_TOKENS = 1000;

/** Estimated output tokens booked for calls without max_tokens. */
const DEFAULT_OUTPUT_TOKENS = 1000;

// -------------------------------------------------------------------
// --- 2. Exported Functions
// -------------------------------------------------------------------

/**
 * Validates integer rate and token limits.
 * @throws {TypeError} If any limit is not an integer >= 1 or if buckets is not an array.
 */
export function validateLimits(limits: unknown, option: string): asserts limits is LlmLimits {
    checkKeys(limits, LIMIT_KEYS, `${option}.`);
    const values = limits as LlmLimits;
    for (const key of INTEGER_LIMITS) {
        const value = values[key];
        if (value !== undefined && (!Number.isInteger(value) || value < 1)) {
            throw new TypeError(`Invalid value for option "${option}.${key}"`);
        }
    }
    if (values.buckets !== undefined && !Array.isArray(values.buckets)) {
        throw new TypeError(`Invalid value for option "${option}.buckets"`);
    }
}

/**
 * Validates limiter retry and wait options.
 * @throws {TypeError} If any limiter option is not an integer >= 0.
 */
export function validateLimiterOptions(
    limiterOptions: unknown,
): asserts limiterOptions is LlmLimiterOptions | undefined {
    if (limiterOptions === undefined) {
        return;
    }
    checkKeys(limiterOptions, LIMITER_KEYS, 'limiterOptions.');
    for (const [key, value] of Object.entries(limiterOptions as LlmLimiterOptions)) {
        if (value !== undefined && (!Number.isInteger(value) || value < 0)) {
            throw new TypeError(`Invalid value for option "limiterOptions.${key}"`);
        }
    }
}

/**
 * Validates `charsPerToken` configuration value.
 * @throws {TypeError} If value is not a positive finite number.
 */
export function validateCharsPerToken(charsPerToken: unknown, option: string): number | undefined {
    if (
        charsPerToken !== undefined &&
        (typeof charsPerToken !== 'number' || !Number.isFinite(charsPerToken) || charsPerToken <= 0)
    ) {
        throw new TypeError(`Invalid value for option "${option}"`);
    }
    return charsPerToken;
}

/**
 * **Creates smart-limit buckets**: instantiates rate and concurrency limits.
 *
 * - instantiates concurrency, sliding window, and quota buckets
 * - excludes concurrency when handled by model queue
 * @throws {TypeError} If bucket configuration is invalid.
 */
export function createLimitBuckets(given: LlmLimits | undefined, option: string, withConcurrency: boolean): Bucket[] {
    const limits = given ?? {};
    const buckets: Bucket[] = [
        ...(limits.concurrency === undefined || !withConcurrency ? [] : [createConcurrencyBucket(limits.concurrency)]),
        ...(limits.rps === undefined ? [] : [createSlidingWindow({ requests: limits.rps, perMs: MS_PER_SECOND })]),
        ...(limits.rpm === undefined ? [] : [createRPM(limits.rpm)]),
        ...(limits.rpd === undefined ? [] : [createRPD(limits.rpd)]),
        ...(limits.tpm === undefined ? [] : [createTPM(limits.tpm)]),
        ...(limits.tpd === undefined ? [] : [createTPD(limits.tpd)]),
        ...(limits.intervalMs === undefined ? [] : [createSlidingWindow({ requests: 1, perMs: limits.intervalMs })]),
        ...(limits.buckets ?? []),
    ];
    const invalid = buckets.find((bucket) => bucket.configError !== undefined);
    if (invalid !== undefined) {
        throw new TypeError(`Invalid value for option "${option}": ${invalid.configError?.message ?? ''}`);
    }
    return buckets;
}

/**
 * **Estimates request token usage**: calculates initial token booking before call.
 *
 * - counts characters across messages, tool calls, and tool definitions
 * - converts characters to tokens using `charsPerToken` ratio
 * - adds fixed allowance per image and requested `max_tokens` (or default)
 */
export function estimateTokens(request: LlmProviderRequest, charsPerToken: number): number {
    let chars = request.tools === undefined ? 0 : JSON.stringify(request.tools).length;
    let images = 0;
    for (const message of request.messages) {
        if (message.role === 'assistant') {
            chars += message.reasoning?.length ?? 0;
            for (const call of message.tool_calls ?? []) {
                chars += call.function.name.length + call.function.arguments.length;
            }
        }
        if (message.content === null) {
            continue;
        }
        if (typeof message.content === 'string') {
            chars += message.content.length;
            continue;
        }
        for (const part of message.content) {
            if (part.type === 'text') {
                chars += part.text.length;
            } else {
                images += 1;
            }
        }
    }
    return Math.ceil(chars / charsPerToken) + images * IMAGE_TOKENS + (request.max_tokens ?? DEFAULT_OUTPUT_TOKENS);
}
