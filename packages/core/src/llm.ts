// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import {
    createQueue,
    type JobError,
    type JobErrorType,
    type JobOutcome,
    type JobResult,
    type LimiterError,
    type PushOptions,
    type Queue,
} from 'smart-limit';
import { resolveCapabilities, validateSupport } from './capabilities.js';
import {
    createLimitBuckets,
    estimateTokens,
    validateCharsPerToken,
    validateLimiterOptions,
    validateLimits,
} from './limits.js';
import {
    applyDefaults,
    checkKeys,
    copyPlain,
    isPlainObject,
    type PreparedRequest,
    prepareRequest,
    validateDefaults,
    validateRequest,
} from './request.js';
import { findSchemaMismatch, parseJsonText, removeOptionalNulls } from './schema.js';
import type {
    Llm,
    LlmCallOptions,
    LlmConfig,
    LlmContentPart,
    LlmDefaults,
    LlmEmbedInput,
    LlmEmbedOutput,
    LlmEmbedRequest,
    LlmError,
    LlmLimiterOptions,
    LlmMessage,
    LlmModel,
    LlmModelCall,
    LlmModelInfo,
    LlmModelInput,
    LlmModelNames,
    LlmModelOptions,
    LlmModelSpec,
    LlmOutput,
    LlmProvider,
    LlmProviderEmbedOutput,
    LlmProviderEmbedRequest,
    LlmProviderError,
    LlmProviderOutput,
    LlmProviderSpec,
    LlmRequest,
    LlmSettings,
    Result,
} from './types.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Configured model entry with queues for normal, batch, and embedding calls. */
type ModelEntry = LlmModel & {
    adapter: LlmProvider;
    defaults: (LlmDefaults | undefined)[];
    extras: Record<string, unknown>[];
    queue: Queue<Job, Answer>;
    batchQueue: Queue<Job, Answer>;
    embedQueue: Queue<EmbedJob, LlmProviderEmbedOutput>;
    charsPerToken: number;
};

/** One provider call within an embedding request. */
type EmbedJob = { request: LlmProviderEmbedRequest; attempts: number; lastError?: AttemptError };

/** Shared configuration settings applied across providers. */
type SharedSettings = { defaults: LlmDefaults | undefined; limiterOptions: LlmLimiterOptions; charsPerToken: number };

/** One queued generation request attempt. */
type Job = {
    prepared: PreparedRequest;
    extra: Record<string, unknown>;
    attempts: number;
    lastError?: AttemptError;
};

/** Provider attempt output with checked JSON value. */
type Answer = LlmProviderOutput & { json?: unknown };

/** Provider attempt error, including JSON schema validation errors. */
type AttemptError = Omit<LlmProviderError, 'output'> & { output?: Answer };

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Members of `LlmConfig`, `LlmProviderSpec`, `LlmModelSpec` and the call options. */
const CONFIG_KEYS: ReadonlySet<string> = new Set(['defaults', 'limiterOptions', 'charsPerToken', 'providers']);
const PROVIDER_KEYS: ReadonlySet<string> = new Set([
    'adapter',
    'defaults',
    'sharedLimits',
    'charsPerToken',
    'extra',
    'models',
]);
const MODEL_KEYS: ReadonlySet<string> = new Set([
    'model',
    'defaults',
    'modelLimits',
    'charsPerToken',
    'capabilities',
    'extra',
]);
const CALL_OPTION_KEYS: ReadonlySet<string> = new Set(['signal', 'extra']);

/** Members of `LlmEmbedRequest`, of an input object and of the `embed` options. */
const EMBED_KEYS: ReadonlySet<string> = new Set(['model', 'input', 'task', 'dimensions']);
const EMBED_INPUT_KEYS: ReadonlySet<string> = new Set(['text', 'title']);
const EMBED_OPTION_KEYS: ReadonlySet<string> = new Set(['signal']);

/** Values of `LlmEmbedTask`. */
const EMBED_TASKS: ReadonlySet<unknown> = new Set(['query', 'document', 'similarity', 'classification', 'clustering']);

/** Request fields copied unchanged into settings. */
const SETTING_FIELDS = [
    'temperature',
    'top_p',
    'max_tokens',
    'seed',
    'stop',
    'reasoning_effort',
    'tool_choice',
    'web_search',
    'batch',
] as const;

/** Reserved property names disallowed as model names. */
const RESERVED_NAMES: ReadonlySet<string> = new Set([
    'generate',
    'embed',
    'models',
    'listModels',
    'then',
    'toJSON',
    ...Object.getOwnPropertyNames(Object.prototype),
]);

/** Default concurrency when model concurrency limit is omitted. */
const UNLIMITED_CONCURRENCY = Number.POSITIVE_INFINITY;

/** Unbounded queue buffer allowing sleeping retries without blocking active slots. */
const UNLIMITED_BUFFER = Number.POSITIVE_INFINITY;

/** Default character-to-token divisor. */
const DEFAULT_CHARS_PER_TOKEN = 4;

/** Lowest HTTP status indicating server error (5xx). */
const MIN_SERVER_ERROR_STATUS = 500;

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Creates multi-provider LLM instance**: routes named models through shared queues and limits.
 *
 * - routes calls by configured model name across adapters with layered defaults
 * - attaches direct callable model methods: `llm.flash(input, options)`
 * - enforces per-model and shared rate/concurrency limits via smart-limit queues
 * - manages separate batch and embedding queues with retry backoff and rate limit locks
 * @throws {TypeError} If configuration options are invalid or contain reserved names.
 */
export function createLlm<const C extends LlmConfig>(config: C): Llm<LlmModelNames<C>> {
    checkKeys(config, CONFIG_KEYS, '');
    if (config.defaults !== undefined) {
        validateDefaults(config.defaults, 'defaults');
    }
    validateLimiterOptions(config.limiterOptions);
    const charsPerToken = validateCharsPerToken(config.charsPerToken, 'charsPerToken') ?? DEFAULT_CHARS_PER_TOKEN;
    // deep clone configuration to prevent external mutations from affecting calls
    const shared: SharedSettings = {
        defaults: copyPlain(config.defaults),
        limiterOptions: config.limiterOptions ?? {},
        charsPerToken,
    };
    if (!isPlainObject(config.providers) || Object.keys(config.providers).length === 0) {
        throw new TypeError('Invalid value for option "providers"');
    }
    const entries = new Map<string, ModelEntry>();
    for (const [providerName, spec] of Object.entries<LlmProviderSpec>(config.providers)) {
        for (const entry of createProviderEntries(providerName, spec, shared)) {
            if (entries.has(entry.name) || RESERVED_NAMES.has(entry.name)) {
                const reason = entries.has(entry.name) ? 'the name is taken' : `${entry.name} is reserved`;
                throw new TypeError(
                    `Invalid value for option "providers.${providerName}.models.${entry.name}": ${reason}`,
                );
            }
            entries.set(entry.name, entry);
        }
    }
    const adapters = new Map(
        Object.entries<LlmProviderSpec>(config.providers).map(([name, spec]) => [name, spec.adapter]),
    );
    const llm: Record<string, unknown> = {
        models: [...entries.values()].map(({ name, provider, model, capabilities }) => ({
            name,
            provider,
            model,
            capabilities: copyPlain(capabilities),
        })),
        generate(request: LlmRequest, options: LlmCallOptions = {}) {
            return generate(entries, request, options);
        },
        embed(request: LlmEmbedRequest, options: { signal?: AbortSignal } = {}) {
            return embed(entries, request, options);
        },
        listModels(provider: string, options: { signal?: AbortSignal } = {}) {
            return listModels(adapters, provider, options);
        },
    };
    for (const entry of entries.values()) {
        // define own property so model names cannot collide with object prototype setters
        Object.defineProperty(llm, entry.name, { value: createModelCall(entries, entry), enumerable: true });
    }
    return llm as Llm<LlmModelNames<C>>;
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/**
 * Creates model entries and queues for provider.
 * @throws {TypeError} If provider configuration or adapter is invalid.
 */
function createProviderEntries(providerName: string, spec: LlmProviderSpec, shared: SharedSettings): ModelEntry[] {
    const option = `providers.${providerName}`;
    checkKeys(spec, PROVIDER_KEYS, `${option}.`);
    const { adapter } = spec;
    if (
        !isPlainObject(adapter) ||
        typeof adapter.generate !== 'function' ||
        typeof adapter.name !== 'string' ||
        !isOptionalFunction(adapter.listModels) ||
        !isOptionalFunction(adapter.validateExtra) ||
        !isOptionalFunction(adapter.mergeExtra) ||
        !isOptionalFunction(adapter.embed) ||
        !(
            adapter.maxEmbedInputs === undefined ||
            (Number.isInteger(adapter.maxEmbedInputs) && adapter.maxEmbedInputs > 0)
        )
    ) {
        throw new TypeError(`Invalid value for option "${option}.adapter"`);
    }
    validateSupport(adapter.supports, `${option}.adapter.supports`);
    if (adapter.supports.embeddings === true && adapter.embed === undefined) {
        throw new TypeError(`Invalid value for option "${option}.adapter": supports embeddings without embed`);
    }
    if (spec.defaults !== undefined) {
        validateDefaults(spec.defaults, `${option}.defaults`);
    }
    if (spec.sharedLimits !== undefined) {
        validateLimits(spec.sharedLimits, `${option}.sharedLimits`);
    }
    const providerChars = validateCharsPerToken(spec.charsPerToken, `${option}.charsPerToken`);
    validateExtra(adapter, spec.extra, `${option}.extra`);
    if (!isPlainObject(spec.models) || Object.keys(spec.models).length === 0) {
        throw new TypeError(`Invalid value for option "${option}.models"`);
    }
    const providerBuckets = createLimitBuckets(spec.sharedLimits, `${option}.sharedLimits`, true);
    return Object.entries<string | LlmModelSpec>(spec.models).map(([name, entry]) => {
        const modelOption = `${option}.models.${name}`;
        const model = toModelSpec(entry, modelOption);
        validateExtra(adapter, model.extra, `${modelOption}.extra`);
        const buckets = createLimitBuckets(model.modelLimits, `${modelOption}.modelLimits`, false);
        const concurrency = model.modelLimits?.concurrency ?? UNLIMITED_CONCURRENCY;
        return {
            name,
            provider: providerName,
            model: model.model,
            capabilities: copyPlain(
                resolveCapabilities(
                    model.capabilities,
                    adapter.supports,
                    adapter.name,
                    `${modelOption}.capabilities`,
                    adapter.maxEmbedInputs,
                ),
            ),
            adapter,
            defaults: [shared.defaults, copyPlain(spec.defaults), copyPlain(model.defaults)],
            extras: [copyPlain(spec.extra ?? {}), copyPlain(model.extra ?? {})],
            queue: createQueue<Job, Answer>({
                ...shared.limiterOptions,
                concurrency,
                maxBufferSize: UNLIMITED_BUFFER,
                buckets: [...providerBuckets, ...buckets],
                processJob: (job, context) => attempt(adapter, job, context.signal, false),
            }),
            batchQueue: createQueue<Job, Answer>({
                ...shared.limiterOptions,
                concurrency,
                maxBufferSize: UNLIMITED_BUFFER,
                processJob: (job, context) => attempt(adapter, job, context.signal, true),
            }),
            embedQueue: createQueue<EmbedJob, LlmProviderEmbedOutput>({
                ...shared.limiterOptions,
                concurrency,
                maxBufferSize: UNLIMITED_BUFFER,
                buckets: [...providerBuckets, ...buckets],
                processJob: (job, context) => attemptEmbed(adapter, job, context.signal),
            }),
            charsPerToken: model.charsPerToken ?? providerChars ?? shared.charsPerToken,
        };
    });
}

/**
 * Normalizes model specification object.
 * @throws {TypeError} If model entry configuration is invalid.
 */
function toModelSpec(entry: string | LlmModelSpec, option: string): LlmModelSpec {
    if (typeof entry === 'string') {
        if (entry === '') {
            throw new TypeError(`Invalid value for option "${option}"`);
        }
        return { model: entry };
    }
    checkKeys(entry, MODEL_KEYS, `${option}.`);
    if (typeof entry.model !== 'string' || entry.model === '') {
        throw new TypeError(`Invalid value for option "${option}.model"`);
    }
    if (entry.defaults !== undefined) {
        validateDefaults(entry.defaults, `${option}.defaults`);
    }
    if (entry.modelLimits !== undefined) {
        validateLimits(entry.modelLimits, `${option}.modelLimits`);
    }
    validateCharsPerToken(entry.charsPerToken, `${option}.charsPerToken`);
    return entry;
}

/** Executes request through model queue; never throws or rejects. */
async function generate(
    entries: ReadonlyMap<string, ModelEntry>,
    request: LlmRequest,
    options: LlmCallOptions,
): Promise<Result<LlmOutput, LlmError>> {
    const checked = checkCall(entries, request, options);
    if (!checked.ok) {
        return checked;
    }
    const { entry, filled } = checked.data;
    const names = { name: entry.name, provider: entry.provider };
    let job: Job | undefined;
    let settings: LlmSettings | undefined;
    try {
        const prepared = prepareRequest(filled, entry.model, entry.capabilities);
        if (!prepared.ok) {
            return { ok: false, error: { ...prepared.error, ...names } };
        }
        const layers = [...entry.extras, options.extra ?? {}];
        const extra = entry.adapter.mergeExtra?.(layers) ?? Object.assign({}, ...layers);
        job = { prepared: prepared.data, extra, attempts: 0 };
        settings = toSettings(entry, prepared.data, extra);
        const isBatch = prepared.data.request.batch === true;
        const push: PushOptions = {};
        if (!isBatch) {
            push.cost = { tokens: estimateTokens(prepared.data.request, entry.charsPerToken) };
        }
        if (options.signal !== undefined) {
            push.signal = options.signal;
        }
        const outcome = await (isBatch ? entry.batchQueue : entry.queue).push(job, push);
        return toResult(entry, outcome, job, settings);
    } catch (error) {
        const unexpected = unexpectedError(error);
        return {
            ok: false,
            error:
                settings === undefined
                    ? { ...unexpected, ...names }
                    : toCallError(entry, unexpected, settings, job?.attempts ?? 0),
        };
    }
}

/** Extracts applied settings from prepared request and merged extra options. */
function toSettings(entry: ModelEntry, prepared: PreparedRequest, extra: Record<string, unknown>): LlmSettings {
    const { request } = prepared;
    const settings: Record<string, unknown> = { name: entry.name, provider: entry.provider, model: request.model };
    // clone settings so caller mutations do not affect subsequent calls
    for (const field of SETTING_FIELDS) {
        if (request[field] !== undefined) {
            settings[field] = copyPlain(request[field]);
        }
    }
    if (prepared.json && entry.capabilities.json !== false) {
        settings.jsonMode = entry.capabilities.json;
    }
    if (request.tools !== undefined) {
        settings.tools = request.tools.map((tool) => tool.function.name);
    }
    if (Object.keys(extra).length > 0) {
        settings.extra = copyPlain(extra);
    }
    return settings as LlmSettings;
}

/** Creates model shortcut function bound to model entry; never throws or rejects. */
function createModelCall(entries: ReadonlyMap<string, ModelEntry>, entry: ModelEntry): LlmModelCall {
    return (input, options = {}) => {
        const call = toModelCall(input, options, entry.name);
        if (!call.ok) {
            const error = { ...call.error, name: entry.name, provider: entry.provider };
            return Promise.resolve({ ok: false, error });
        }
        return generate(entries, call.data.request, call.data.options);
    };
}

/** Converts model shortcut arguments into request payload and call options. */
function toModelCall(
    input: LlmModelInput,
    options: LlmModelOptions,
    name: string,
): Result<{ request: LlmRequest; options: LlmCallOptions }, LlmError> {
    try {
        if (!isPlainObject(options)) {
            throw new TypeError('Invalid value for the options: an object');
        }
        const { signal, extra, ...fields } = options;
        for (const key of ['model', 'messages']) {
            if (key in fields) {
                throw new TypeError(`Option "${key}" is not supported: llm.${name}() sets it from its name and input`);
            }
        }
        const callOptions: LlmCallOptions = {};
        if (signal !== undefined) {
            callOptions.signal = signal;
        }
        if (extra !== undefined) {
            callOptions.extra = extra;
        }
        const request = { ...fields, model: name, messages: toMessages(input) } as LlmRequest;
        return { ok: true, data: { request, options: callOptions } };
    } catch (error) {
        return { ok: false, error: { code: 'invalid_request', message: errorMessage(error), retryable: false } };
    }
}

/**
 * Normalizes model input into message array.
 * @throws {TypeError} If input is empty or invalid.
 */
function toMessages(input: LlmModelInput): LlmMessage[] {
    if (typeof input === 'string') {
        return [{ role: 'user', content: input }];
    }
    if (Array.isArray(input) && input.length > 0) {
        if (input.every((item) => isPlainObject(item) && 'role' in item)) {
            return [...(input as readonly LlmMessage[])];
        }
        if (input.every((item) => isPlainObject(item) && 'type' in item)) {
            return [{ role: 'user', content: [...(input as readonly LlmContentPart[])] }];
        }
    }
    throw new TypeError('Invalid value for the input: a string, content parts or messages');
}

/** Validates call options, resolves model entry, and applies defaults. */
function checkCall(
    entries: ReadonlyMap<string, ModelEntry>,
    request: LlmRequest,
    options: LlmCallOptions,
): Result<{ entry: ModelEntry; filled: LlmRequest }, LlmError> {
    let entry: ModelEntry | undefined;
    try {
        // lookup early so errors carry model and provider names
        const model: unknown = isPlainObject(request) ? request.model : undefined;
        entry = typeof model === 'string' ? entries.get(model) : undefined;
        checkKeys(options, CALL_OPTION_KEYS, '');
        if (
            options.extra !== undefined &&
            (!isPlainObject(options.extra) || Object.hasOwn(options.extra, '__proto__'))
        ) {
            throw new TypeError('Invalid value for option "extra"');
        }
        if (options.signal !== undefined && !(options.signal instanceof AbortSignal)) {
            throw new TypeError('Invalid value for option "signal"');
        }
        validateRequest(request);
        if (entry === undefined) {
            throw new TypeError(`Invalid value for option "model": ${request.model} is not configured`);
        }
        if (options.extra !== undefined) {
            entry.adapter.validateExtra?.(options.extra, 'extra');
        }
        const filled = applyDefaults(request, ...entry.defaults);
        return { ok: true, data: { entry, filled } };
    } catch (error) {
        const invalid: LlmError = { code: 'invalid_request', message: errorMessage(error), retryable: false };
        return {
            ok: false,
            error: entry === undefined ? invalid : { ...invalid, name: entry.name, provider: entry.provider },
        };
    }
}

/**
 * **Executes embedding request**: splits input into batched provider calls via embedding queue.
 *
 * - books token estimates per batch, settled on reported input tokens
 * - fails fast on first failing call and aborts remaining batches
 */
async function embed(
    entries: ReadonlyMap<string, ModelEntry>,
    request: LlmEmbedRequest,
    options: { signal?: AbortSignal },
): Promise<Result<LlmEmbedOutput, LlmError>> {
    const checked = checkEmbed(entries, request, options);
    if (!checked.ok) {
        return checked;
    }
    const { entry, inputs } = checked.data;
    const names = namesOf(entry);
    const startedAt = performance.now();
    const controller = new AbortController();
    const signal =
        options.signal === undefined ? controller.signal : AbortSignal.any([options.signal, controller.signal]);
    const { maxInputs } = entry.capabilities.embeddings;
    const jobs: EmbedJob[] = [];
    for (let start = 0; start < inputs.length; start += maxInputs) {
        const fields = {
            ...(request.task === undefined ? {} : { task: request.task }),
            ...(request.dimensions === undefined ? {} : { dimensions: request.dimensions }),
        };
        jobs.push({
            request: { model: entry.model, input: inputs.slice(start, start + maxInputs), ...fields },
            attempts: 0,
        });
    }
    // capture first failure in time to abort remaining batches immediately
    let failed: AttemptError | undefined;
    try {
        const outcomes = await Promise.all(
            jobs.map(async (job) => {
                const characters = job.request.input.reduce(
                    (sum, input) => sum + input.text.length + (input.title?.length ?? 0),
                    0,
                );
                const cost = { tokens: Math.ceil(characters / entry.charsPerToken) };
                const outcome = await entry.embedQueue.push(job, { cost, signal });
                if (!outcome.ok && failed === undefined) {
                    failed = fromLimiterError(outcome.error, job);
                    controller.abort();
                }
                return outcome;
            }),
        );
        const attempts = jobs.reduce((sum, job) => sum + job.attempts, 0);
        if (failed !== undefined) {
            const { output: _output, ...error } = failed;
            return { ok: false, error: { ...error, ...names, attempts } };
        }
        const embeddings: number[][] = [];
        let inputTokens: number | null = null;
        let model = entry.model;
        for (const outcome of outcomes) {
            if (outcome.ok) {
                embeddings.push(...outcome.data.embeddings);
                model = outcome.data.model;
                inputTokens =
                    outcome.data.usage.inputTokens === null
                        ? inputTokens
                        : (inputTokens ?? 0) + outcome.data.usage.inputTokens;
            }
        }
        const ms = Math.round(performance.now() - startedAt);
        return {
            ok: true,
            data: { embeddings, ...names, model, usage: { inputTokens }, ms, calls: jobs.length, attempts },
        };
    } catch (error) {
        controller.abort();
        return { ok: false, error: { ...unexpectedError(error), ...names } };
    }
}

/** Validates embedding request parameters and checks model embedding capability. */
function checkEmbed(
    entries: ReadonlyMap<string, ModelEntry>,
    request: LlmEmbedRequest,
    options: { signal?: AbortSignal },
): Result<{ entry: ModelEntry; inputs: { text: string; title?: string }[] }, LlmError> {
    let entry: ModelEntry | undefined;
    try {
        const model: unknown = isPlainObject(request) ? request.model : undefined;
        entry = typeof model === 'string' ? entries.get(model) : undefined;
        checkKeys(options, EMBED_OPTION_KEYS, '');
        if (options.signal !== undefined && !(options.signal instanceof AbortSignal)) {
            throw new TypeError('Invalid value for option "signal"');
        }
        checkKeys(request, EMBED_KEYS, '');
        if (entry === undefined) {
            throw new TypeError(`Invalid value for option "model": ${String(model)} is not configured`);
        }
        const given: unknown = typeof request.input === 'string' ? [request.input] : request.input;
        if (!Array.isArray(given) || given.length === 0) {
            throw new TypeError('Invalid value for option "input": one or more texts');
        }
        const inputs = given.map((input: LlmEmbedInput, index) => toEmbedText(input, `input.${index}`));
        if (request.task !== undefined && !EMBED_TASKS.has(request.task)) {
            throw new TypeError(
                'Invalid value for option "task": query, document, similarity, classification or clustering',
            );
        }
        if (request.task !== 'document' && inputs.some((input) => input.title !== undefined)) {
            throw new TypeError('Option "input.title" is not supported: only with task document');
        }
        const { dimensions } = request;
        if (dimensions !== undefined && !(Number.isInteger(dimensions) && dimensions > 0)) {
            throw new TypeError('Invalid value for option "dimensions": a positive integer');
        }
        if (!entry.capabilities.embeddings.supported) {
            const message = `${entry.name} does not support embeddings`;
            return { ok: false, error: { code: 'unsupported', message, retryable: false, ...namesOf(entry) } };
        }
        return { ok: true, data: { entry, inputs } };
    } catch (error) {
        const invalid: LlmError = { code: 'invalid_request', message: errorMessage(error), retryable: false };
        return { ok: false, error: entry === undefined ? invalid : { ...invalid, ...namesOf(entry) } };
    }
}

/**
 * Normalizes embedding input text and optional title.
 * @throws {TypeError} If input text or title is invalid.
 */
function toEmbedText(input: LlmEmbedInput, option: string): { text: string; title?: string } {
    if (typeof input === 'string') {
        if (input === '') {
            throw new TypeError(`Invalid value for option "${option}": a non-empty text`);
        }
        return { text: input };
    }
    checkKeys(input, EMBED_INPUT_KEYS, `${option}.`);
    if (typeof input.text !== 'string' || input.text === '') {
        throw new TypeError(`Invalid value for option "${option}.text": a non-empty text`);
    }
    if (input.title !== undefined && (typeof input.title !== 'string' || input.title === '')) {
        throw new TypeError(`Invalid value for option "${option}.title": a non-empty text`);
    }
    return input.title === undefined ? { text: input.text } : { text: input.text, title: input.title };
}

/** Returns model and provider name object from model entry. */
function namesOf(entry: ModelEntry): { name: string; provider: string } {
    return { name: entry.name, provider: entry.provider };
}

/** Executes single embedding attempt and settles token bucket usage. */
async function attemptEmbed(
    adapter: LlmProvider,
    job: EmbedJob,
    signal: AbortSignal,
): Promise<JobResult<LlmProviderEmbedOutput>> {
    job.attempts += 1;
    try {
        const result = await (adapter.embed as NonNullable<LlmProvider['embed']>)(job.request, { signal });
        if (!result.ok) {
            job.lastError = result.error;
            return toJobError(result.error);
        }
        const count = result.data.embeddings.length;
        if (count !== job.request.input.length) {
            const message = `${adapter.name} returned ${count} vectors for ${job.request.input.length} texts`;
            job.lastError = { code: 'unexpected', message, retryable: false };
            return toJobError(job.lastError);
        }
        const tokens = result.data.usage.inputTokens;
        return tokens === null ? { ok: true, data: result.data } : { ok: true, data: result.data, usage: { tokens } };
    } catch (error) {
        return toJobError(unexpectedError(error));
    }
}

/** Lists models supported by provider adapter; never throws or rejects. */
async function listModels(
    adapters: ReadonlyMap<string, LlmProvider>,
    provider: string,
    options: { signal?: AbortSignal },
): Promise<Result<LlmModelInfo[], LlmError>> {
    const adapter = adapters.get(provider);
    if (adapter === undefined) {
        const message = `Invalid value for option "provider": ${provider} is not configured`;
        return { ok: false, error: { code: 'invalid_request', message, retryable: false } };
    }
    if (adapter.listModels === undefined) {
        return {
            ok: false,
            error: { code: 'unsupported', message: `${adapter.name} cannot list models`, retryable: false },
        };
    }
    try {
        return await adapter.listModels(options);
    } catch (error) {
        return { ok: false, error: unexpectedError(error) };
    }
}

/**
 * Validates extra provider options object.
 * @throws {TypeError} If extra is not plain object, contains __proto__, or fails adapter check.
 */
function validateExtra(adapter: LlmProvider, extra: unknown, option: string): void {
    if (extra === undefined) {
        return;
    }
    if (!isPlainObject(extra) || Object.hasOwn(extra, '__proto__')) {
        throw new TypeError(`Invalid value for option "${option}"`);
    }
    adapter.validateExtra?.(extra, option);
}

/**
 * **Executes single provider attempt**: calls provider adapter and validates answer JSON.
 *
 * - normal call: 429 or outage locks model queue; reported usage settles token buckets
 * - batch call: retries retryable errors with backoff without locking queue
 */
async function attempt(
    adapter: LlmProvider,
    job: Job,
    signal: AbortSignal,
    isBatch: boolean,
): Promise<JobResult<Answer>> {
    job.attempts += 1;
    // catch adapter throws or malformed results and map to unexpected error
    try {
        const result = await adapter.generate(job.prepared.request, { signal, extra: job.extra });
        if (isBatch) {
            const checked = result.ok ? checkAnswer(job.prepared, result.data) : result;
            if (checked.ok) {
                return { ok: true, data: checked.data };
            }
            const { error } = checked;
            job.lastError = error;
            const type: JobErrorType = error.retryable ? 'retry_backoff' : 'fatal';
            const jobError: JobError = { type, message: error.message, code: error.code, cause: error };
            if (error.retryAfterMs !== undefined) {
                jobError.retryAfterMs = error.retryAfterMs;
            }
            return { ok: false, error: jobError };
        }
        if (!result.ok) {
            job.lastError = result.error;
            return toJobError(result.error);
        }
        const checked = checkAnswer(job.prepared, result.data);
        if (!checked.ok) {
            job.lastError = checked.error;
            return toJobError(checked.error);
        }
        const tokens = usedTokens(result.data.usage);
        return tokens === undefined
            ? { ok: true, data: checked.data }
            : { ok: true, data: checked.data, usage: { tokens } };
    } catch (error) {
        return toJobError(unexpectedError(error));
    }
}

/** Parses and validates answer text against expected JSON schema. */
function checkAnswer(prepared: PreparedRequest, output: LlmProviderOutput): Result<Answer, AttemptError> {
    if (!prepared.json || (output.toolCalls?.length ?? 0) > 0) {
        return { ok: true, data: output };
    }
    const parsed = parseJsonText(output.text);
    const answered = { text: output.text, model: output.model, usage: output.usage, ms: output.ms };
    if (parsed === undefined) {
        return {
            ok: false,
            error: {
                code: 'invalid_json',
                message: 'The answer is not JSON',
                retryable: prepared.prompted,
                ...answered,
                output,
            },
        };
    }
    const json =
        prepared.strict && prepared.schema !== undefined
            ? removeOptionalNulls(parsed.value, prepared.schema)
            : parsed.value;
    const mismatch = prepared.schema === undefined ? undefined : findSchemaMismatch(json, prepared.schema);
    if (mismatch !== undefined) {
        return {
            ok: false,
            error: {
                code: 'invalid_json',
                message: `The answer does not match the schema: ${mismatch}`,
                retryable: prepared.prompted,
                ...answered,
                output: { ...output, json },
            },
        };
    }
    return { ok: true, data: { ...output, json } };
}

/** Maps attempt error to smart-limit job error with retry and usage policy. */
function toJobError(error: AttemptError): { ok: false; error: JobError } {
    const type: JobErrorType =
        error.code === 'invalid_json' && error.retryable ? 'retry_immediate' : jobErrorType(error);
    const jobError: JobError = { type, message: error.message, code: error.code, cause: error };
    if (error.retryAfterMs !== undefined) {
        jobError.retryAfterMs = error.retryAfterMs;
    }
    const tokens = error.usage === undefined ? undefined : usedTokens(error.usage);
    if (tokens !== undefined) {
        jobError.usage = { tokens };
    } else if (error.code === 'rate_limited') {
        // refund token estimate on rate limit rejection
        jobError.usage = { tokens: 0 };
    }
    return { ok: false, error: jobError };
}

/**
 * **Maps error to queue retry type**: determines backoff strategy for smart-limit queue.
 *
 * - `backoff_global`: 429, network errors, and 5xx server outages that pause queue
 * - `retry_backoff`: transient request errors (timeouts, 408/409/425)
 * - `fatal`: non-retryable errors
 */
function jobErrorType(error: AttemptError): JobErrorType {
    if (!error.retryable) {
        return 'fatal';
    }
    const isServerError = error.code === 'http' && (error.status ?? 0) >= MIN_SERVER_ERROR_STATUS;
    return error.code === 'rate_limited' || error.code === 'network' || isServerError
        ? 'backoff_global'
        : 'retry_backoff';
}

/** Creates result object from job outcome with settings and attempt count. */
function toResult(
    entry: ModelEntry,
    outcome: JobOutcome<Answer>,
    job: Job,
    settings: LlmSettings,
): Result<LlmOutput, LlmError> {
    if (outcome.ok) {
        const names = { name: entry.name, provider: entry.provider };
        return { ok: true, data: { ...outcome.data, ...names, settings, attempts: job.attempts } };
    }
    return { ok: false, error: toCallError(entry, fromLimiterError(outcome.error, job), settings, job.attempts) };
}

/** Enriches error with model metadata, applied settings, and attempt count. */
function toCallError(entry: ModelEntry, error: AttemptError, settings: LlmSettings, attempts: number): LlmError {
    const names = { name: entry.name, provider: entry.provider };
    const { output, ...rest } = error;
    return {
        ...rest,
        ...names,
        ...(output === undefined ? {} : { output: { ...output, ...names } }),
        settings,
        attempts,
    };
}

/** Maps smart-limit limiter error or underlying cause to attempt error. */
function fromLimiterError(error: LimiterError, job: { lastError?: AttemptError }): AttemptError {
    if (error.code === 'ABORTED') {
        return { code: 'aborted', message: 'Aborted', retryable: false };
    }
    const cause = error.cause as { cause?: unknown } | undefined;
    const last = cause?.cause as AttemptError | undefined;
    if (last !== undefined && typeof last === 'object' && 'code' in last && 'retryable' in last) {
        return last;
    }
    if (job.lastError !== undefined && (error.code === 'MAX_WAIT_EXCEEDED' || error.code === 'STOPPED')) {
        return job.lastError;
    }
    if (error.code === 'TIMEOUT') {
        return { code: 'timeout', message: error.message, retryable: true };
    }
    return { code: 'limiter', message: error.message, retryable: false, limiterCode: error.code };
}

/** Creates unexpected error from thrown exception. */
function unexpectedError(error: unknown): LlmError {
    return { code: 'unexpected', message: errorMessage(error), retryable: false };
}

/** Extracts error message string from unknown error. */
function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

/** Checks whether value is undefined or a function. */
function isOptionalFunction(value: unknown): boolean {
    return value === undefined || typeof value === 'function';
}

/** Sums input, output, and reasoning tokens from usage report. */
function usedTokens(usage: LlmOutput['usage']): number | undefined {
    if (usage.inputTokens === null && usage.outputTokens === null && usage.reasoningTokens === null) {
        return undefined;
    }
    return (usage.inputTokens ?? 0) + (usage.outputTokens ?? 0) + (usage.reasoningTokens ?? 0);
}
