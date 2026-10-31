// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { checkKeys, REASONING_EFFORTS } from './request.js';
import type {
    LlmCapabilities,
    LlmJsonMode,
    LlmModelCapabilities,
    LlmProviderSupport,
    LlmReasoningEffort,
} from './types.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Members of `LlmProviderSupport` that are `true` when set. */
const SUPPORT_FLAGS = ['images', 'tools', 'webSearch', 'reasoning', 'batch', 'embeddings'] as const;

/** Members of `LlmProviderSupport`. */
const SUPPORT_KEYS: ReadonlySet<string> = new Set([...SUPPORT_FLAGS, 'json']);

/** Members of `LlmModelCapabilities`, of its `webSearch` / `batch` and of its `reasoning`. */
const CAPABILITY_KEYS: ReadonlySet<string> = new Set([
    'images',
    'tools',
    'json',
    'webSearch',
    'batch',
    'reasoning',
    'embeddings',
]);
const SWITCH_KEYS: ReadonlySet<string> = new Set(['supported', 'default']);
const REASONING_KEYS: ReadonlySet<string> = new Set(['supported', 'default', 'levels']);
const EMBEDDING_KEYS: ReadonlySet<string> = new Set(['supported', 'maxInputs']);

/** Default texts per embedding provider call when unspecified. */
const DEFAULT_MAX_EMBED_INPUTS = 64;

/** Values of `LlmJsonMode`. */
const JSON_MODES: ReadonlySet<unknown> = new Set<LlmJsonMode>(['native', 'strict', 'grammar', 'prompt']);

// -------------------------------------------------------------------
// --- 2. Exported Functions
// -------------------------------------------------------------------

/**
 * Validates adapter `supports` flags and modes.
 * @throws {TypeError} If option contains unsupported keys, non-boolean flags, or invalid JSON mode.
 */
export function validateSupport(supports: unknown, option: string): asserts supports is LlmProviderSupport {
    checkKeys(supports, SUPPORT_KEYS, `${option}.`);
    const given = supports as Record<string, unknown>;
    for (const key of SUPPORT_FLAGS) {
        if (given[key] !== undefined && given[key] !== true) {
            throw new TypeError(`Invalid value for option "${option}.${key}": true or left out`);
        }
    }
    if (given.json !== undefined && !JSON_MODES.has(given.json)) {
        throw new TypeError(`Invalid value for option "${option}.json": native, strict, grammar or prompt`);
    }
}

/**
 * **Resolves model capabilities**: validates declared options against adapter support.
 *
 * - verifies declared capabilities are supported by adapter
 * - checks capability defaults and reasoning effort levels
 * - sets embeddings maxInputs (declared up to adapter limit, else adapter max, else 64)
 * @throws {TypeError} If capability declaration or values are invalid or unsupported.
 */
export function resolveCapabilities(
    declared: LlmModelCapabilities | undefined,
    supports: LlmProviderSupport,
    adapterName: string,
    option: string,
    maxEmbedInputs?: number,
): LlmCapabilities {
    const given = declared ?? {};
    checkKeys(given, CAPABILITY_KEYS, `${option}.`);
    function unsupported(key: string): TypeError {
        return new TypeError(
            `Invalid value for option "${option}.${key}": the ${adapterName} adapter does not support it`,
        );
    }
    for (const key of ['images', 'tools'] as const) {
        if (given[key] !== undefined && typeof given[key] !== 'boolean') {
            throw new TypeError(`Invalid value for option "${option}.${key}": true or false`);
        }
        if (given[key] === true && supports[key] !== true) {
            throw unsupported(key);
        }
    }
    const { json } = given;
    if (json !== undefined && typeof json !== 'boolean' && !JSON_MODES.has(json)) {
        throw new TypeError(
            `Invalid value for option "${option}.json": true, false, native, strict, grammar or prompt`,
        );
    }
    if (json !== undefined && json !== false && supports.json === undefined) {
        throw unsupported('json');
    }
    return {
        images: given.images === true,
        tools: given.tools === true,
        json: json === true ? (supports.json as LlmJsonMode) : (json ?? false),
        webSearch: resolveSwitch(given.webSearch, supports.webSearch === true, `${option}.webSearch`, unsupported),
        batch: resolveSwitch(given.batch, supports.batch === true, `${option}.batch`, unsupported),
        reasoning: resolveReasoning(given.reasoning, supports.reasoning === true, `${option}.reasoning`, unsupported),
        embeddings: resolveEmbeddings(
            given.embeddings,
            supports.embeddings === true,
            maxEmbedInputs,
            option,
            unsupported,
        ),
    };
}

// -------------------------------------------------------------------
// --- 3. Private Functions
// -------------------------------------------------------------------

/**
 * Resolves `webSearch` or `batch` capability settings.
 * @throws {TypeError} If capability declaration or values are invalid.
 */
function resolveSwitch(
    given: unknown,
    supported: boolean,
    option: string,
    unsupported: (key: string) => TypeError,
): { supported: boolean; default: boolean } {
    if (given === undefined) {
        return { supported: false, default: false };
    }
    checkKeys(given, SWITCH_KEYS, `${option}.`);
    const value = given as { supported?: unknown; default?: unknown };
    if (typeof value.supported !== 'boolean') {
        throw new TypeError(`Invalid value for option "${option}.supported": true or false`);
    }
    if (value.default !== undefined && typeof value.default !== 'boolean') {
        throw new TypeError(`Invalid value for option "${option}.default": true or false`);
    }
    if (value.supported && !supported) {
        throw unsupported(option.slice(option.lastIndexOf('.') + 1));
    }
    if (value.default === true && !value.supported) {
        throw new TypeError(`Invalid value for option "${option}.default": true needs supported: true`);
    }
    return { supported: value.supported, default: value.default === true };
}

/**
 * Resolves model reasoning capabilities and allowed effort levels.
 * @throws {TypeError} If reasoning configuration or effort levels are invalid.
 */
function resolveReasoning(
    given: unknown,
    supported: boolean,
    option: string,
    unsupported: (key: string) => TypeError,
): LlmCapabilities['reasoning'] {
    if (given === undefined) {
        return { supported: false };
    }
    checkKeys(given, REASONING_KEYS, `${option}.`);
    const value = given as { supported?: unknown; default?: unknown; levels?: unknown };
    if (typeof value.supported !== 'boolean') {
        throw new TypeError(`Invalid value for option "${option}.supported": true or false`);
    }
    const { levels } = value;
    if (
        levels !== undefined &&
        (!Array.isArray(levels) ||
            !levels.every((level) => REASONING_EFFORTS.includes(level)) ||
            new Set(levels).size !== levels.length ||
            !levels.some((level) => level !== 'none'))
    ) {
        throw new TypeError(
            `Invalid value for option "${option}.levels": distinct reasoning efforts, at least one other than none`,
        );
    }
    const effort = value.default;
    if (effort !== undefined && !REASONING_EFFORTS.includes(effort as LlmReasoningEffort)) {
        throw new TypeError(`Invalid value for option "${option}.default": a reasoning effort`);
    }
    if (value.supported && !supported) {
        throw unsupported('reasoning');
    }
    if (!value.supported) {
        if (effort !== undefined || levels !== undefined) {
            throw new TypeError(`Invalid value for option "${option}": default and levels need supported: true`);
        }
        return { supported: false };
    }
    const declaredLevels = levels as LlmReasoningEffort[] | undefined;
    if (
        effort !== undefined &&
        declaredLevels !== undefined &&
        !declaredLevels.includes(effort as LlmReasoningEffort)
    ) {
        throw new TypeError(`Invalid value for option "${option}.default": one of levels`);
    }
    return {
        supported: true,
        ...(effort === undefined ? {} : { default: effort as LlmReasoningEffort }),
        ...(declaredLevels === undefined ? {} : { levels: [...declaredLevels] }),
    };
}

/**
 * Resolves embeddings capability and texts per provider call.
 * @throws {TypeError} If declaration is invalid, unsupported, or maxInputs exceeds adapter maximum.
 */
function resolveEmbeddings(
    given: unknown,
    supported: boolean,
    maxEmbedInputs: number | undefined,
    option: string,
    unsupported: (key: string) => TypeError,
): LlmCapabilities['embeddings'] {
    const fallback = maxEmbedInputs ?? DEFAULT_MAX_EMBED_INPUTS;
    if (given === undefined || typeof given === 'boolean') {
        if (given === true && !supported) {
            throw unsupported('embeddings');
        }
        return { supported: given === true, maxInputs: fallback };
    }
    checkKeys(given, EMBEDDING_KEYS, `${option}.embeddings.`);
    const value = given as { supported?: unknown; maxInputs?: unknown };
    if (typeof value.supported !== 'boolean') {
        throw new TypeError(`Invalid value for option "${option}.embeddings.supported": true or false`);
    }
    const { maxInputs } = value;
    const limit = maxEmbedInputs ?? Number.POSITIVE_INFINITY;
    if (
        maxInputs !== undefined &&
        !(Number.isInteger(maxInputs) && (maxInputs as number) > 0 && (maxInputs as number) <= limit)
    ) {
        const range = maxEmbedInputs === undefined ? 'a positive integer' : `an integer from 1 to ${maxEmbedInputs}`;
        throw new TypeError(`Invalid value for option "${option}.embeddings.maxInputs": ${range}`);
    }
    if (value.supported && !supported) {
        throw unsupported('embeddings');
    }
    return { supported: value.supported, maxInputs: (maxInputs as number | undefined) ?? fallback };
}
