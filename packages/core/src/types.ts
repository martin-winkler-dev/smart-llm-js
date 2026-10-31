// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { Bucket } from 'smart-limit';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Outcome of every call; only factories throw TypeError for invalid options. All other calls return Result. */
export type Result<T, E> = { ok: true; data: T } | { ok: false; error: E };

/** Role of chat message: system, developer, user, assistant, or tool. */
export type LlmRole = 'system' | 'developer' | 'user' | 'assistant' | 'tool';

/** Text part of message in OpenAI format. */
export type LlmTextPart = { type: 'text'; text: string };

/**
 * **Image part of message**: OpenAI image part, valid only in user messages.
 *
 * - `url`: base64 data URL (e.g. data:image/png;base64,...); other URLs rejected for cross-provider compatibility
 * - `detail`: optional detail level (auto, low, high) forwarded where supported
 */
export type LlmImagePart = {
    type: 'image_url';
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    image_url: { url: string; detail?: 'auto' | 'low' | 'high' };
};

/** Part of message: text, or image in user message. */
export type LlmContentPart = LlmTextPart | LlmImagePart;

/**
 * **Chat message**: message in OpenAI format within conversation history.
 *
 * - `assistant`: prior completion; content may be null with tool_calls; reasoning preserved where required
 * - `tool`: result of tool call tool_call_id, as text
 */
export type LlmMessage =
    | { role: 'system' | 'developer' | 'user'; content: string | LlmContentPart[] }
    | {
          role: 'assistant';
          content: string | LlmTextPart[] | null;
          // biome-ignore lint/style/useNamingConvention: OpenAI field name
          tool_calls?: LlmToolCall[];
          reasoning?: string;
      }
    | {
          role: 'tool';
          // biome-ignore lint/style/useNamingConvention: OpenAI field name
          tool_call_id: string;
          content: string | LlmTextPart[];
      };

/**
 * **Tool declaration**: function model may call (OpenAI tools entry).
 *
 * - `name`: up to 64 characters (letters, digits, _, -); validated to prevent provider HTTP 400
 * - `parameters`: JSON Schema of arguments; strict forwarded where supported
 */
export type LlmTool = {
    type: 'function';
    function: { name: string; description?: string; parameters?: LlmJsonSchema; strict?: boolean | null };
};

/** Which tools model may call: auto, none, required, or named function. */
export type LlmToolChoice = 'auto' | 'none' | 'required' | { type: 'function'; function: { name: string } };

/**
 * **Tool call**: function invocation in assistant message.
 *
 * - `id`: provider call ID or generated call_ ID
 * - `function.arguments`: arguments as JSON text
 * - `extra_content`: provider metadata passed back in history (e.g. Gemini thought signature)
 */
export type LlmToolCall = {
    id: string;
    type: 'function';
    function: { name: string; arguments: string };
    // biome-ignore lint/style/useNamingConvention: OpenAI-compatible field name
    extra_content?: Record<string, unknown>;
};

/**
 * **JSON Schema of answer**: plain object schema for structured outputs.
 *
 * - supports common subset: type, properties, required, items, enum, description, additionalProperties, anyOf
 * - answers checked against these keywords; unverified keywords passed through
 */
export type LlmJsonSchema = { [key: string]: unknown };

/**
 * **Answer format**: OpenAI response_format specification.
 *
 * - `json_schema`: JSON matching json_schema.schema; delivered natively or via prompt injection
 * - `json_object`: arbitrary JSON object; text: plain text (default)
 */
export type LlmResponseFormat =
    | { type: 'text' }
    | { type: 'json_object' }
    | {
          type: 'json_schema';
          // biome-ignore lint/style/useNamingConvention: OpenAI field name
          json_schema: { name: string; schema?: LlmJsonSchema; description?: string; strict?: boolean | null };
      };

/**
 * **Reasoning effort**: OpenAI reasoning_effort, lowest first.
 *
 * - `none`: turns reasoning off where supported
 * - mapped to nearest declared level in model capabilities; sent as is without levels
 */
export type LlmReasoningEffort = 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max';

/**
 * **Request payload**: subset of OpenAI chat completion body plus web_search and batch.
 *
 * - `model`: configured model name routing call to designated provider
 * - `max_tokens` and `max_completion_tokens`: mutually exclusive output limit
 * - `web_search`, `batch`, `reasoning_effort`: override model defaults; unsupported flags fail with unsupported
 * - `tools` and `tool_choice`: function calling definitions and selection policy
 */
export type LlmRequest<M extends string = string> = {
    model: M;
    messages: LlmMessage[];
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    response_format?: LlmResponseFormat;
    temperature?: number;
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    top_p?: number;
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    max_tokens?: number;
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    max_completion_tokens?: number;
    seed?: number;
    stop?: string | string[];
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    reasoning_effort?: LlmReasoningEffort;
    // biome-ignore lint/style/useNamingConvention: same style as the OpenAI fields
    web_search?: boolean;
    tools?: LlmTool[];
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    tool_choice?: LlmToolChoice;
    batch?: boolean;
};

/**
 * **Request defaults**: sampling parameters and output limit applied across calls.
 *
 * - includes temperature, top_p, max_tokens, max_completion_tokens, seed, and stop
 * - capabilities and per-call options disallowed in defaults
 */
export type LlmDefaults = Pick<
    LlmRequest,
    'temperature' | 'top_p' | 'max_tokens' | 'max_completion_tokens' | 'seed' | 'stop'
>;

/**
 * **Per-call options**: abort signal and provider-specific extra parameters.
 *
 * - `signal`: abort signal cancelling wait or ongoing network request
 * - `extra`: provider-specific parameters merged over provider and model extras
 */
export type LlmCallOptions = { signal?: AbortSignal; extra?: Record<string, unknown> };

// -------------------------------------------------------------------
// --- 1.1 Answers and Errors
// -------------------------------------------------------------------

/** Token counts as reported; null when unreported; outputTokens excludes reasoning tokens. */
export type LlmUsage = { inputTokens: number | null; outputTokens: number | null; reasoningTokens: number | null };

/** Web citation grounding model completion. */
export type LlmSource = { url: string; title?: string };

/**
 * **Applied call settings**: parameters sent to provider after merging defaults and capabilities.
 *
 * - `name`, `provider`, `model`: configured names and provider model identifier
 * - `max_tokens`: resolved output limit (folds max_completion_tokens)
 * - `reasoning_effort`: mapped to supported model levels
 * - `jsonMode`: applied JSON schema delivery mode
 * - `tools`: declared tool names; extra: merged provider options
 */
export type LlmSettings = {
    name: string;
    provider: string;
    model: string;
    temperature?: number;
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    top_p?: number;
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    max_tokens?: number;
    seed?: number;
    stop?: string | string[];
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    reasoning_effort?: LlmReasoningEffort;
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    tool_choice?: LlmToolChoice;
    // biome-ignore lint/style/useNamingConvention: same style as the OpenAI fields
    web_search?: true;
    batch?: true;
    jsonMode?: LlmJsonMode;
    tools?: string[];
    extra?: Record<string, unknown>;
};

/**
 * **Answer of request**: completion text, parsed JSON, reasoning, tool calls, and metadata.
 *
 * - `text`: answer text; json: parsed and checked answer for JSON response_format
 * - `reasoning`: reasoning text returned by provider
 * - `toolCalls`: tool calls from model (omits json)
 * - `settings`: applied call settings; attempts: total provider calls including retries
 */
export type LlmOutput = {
    text: string;
    json?: unknown;
    reasoning?: string;
    sources?: LlmSource[];
    toolCalls?: LlmToolCall[];
    name: string;
    provider: string;
    model: string;
    usage: LlmUsage;
    finishReason: string | null;
    ms: number;
    settings: LlmSettings;
    attempts: number;
};

/**
 * **Unusable answer of failed call**: LlmOutput of last attempt carried on LlmError.
 *
 * - `json`: parsed answer when answer was JSON but failed schema check
 * - `finishReason`: finish reason reported with error, or null
 */
export type LlmAnswer = Omit<LlmOutput, 'settings' | 'attempts'>;

/**
 * **Embedding task**: intended use of embedding vectors.
 *
 * - `document`: search corpus texts; query: search query (asymmetric pair)
 * - `similarity`, `classification`, `clustering`: symmetric comparison tasks
 */
export type LlmEmbedTask = 'query' | 'document' | 'similarity' | 'classification' | 'clustering';

/** Text to embed: string, or text with title for document task. */
export type LlmEmbedInput = string | { text: string; title?: string };

/**
 * **Embedding request**: vector generation request for one or more inputs.
 *
 * - `input`: texts split across provider calls bounded by maxInputs
 * - `task`: embedding task type; dimensions: output vector size
 */
export type LlmEmbedRequest<M extends string = string> = {
    model: M;
    input: string | readonly LlmEmbedInput[];
    task?: LlmEmbedTask;
    dimensions?: number;
};

/**
 * **Vectors of embedding request**: embedding vectors and execution statistics in input order.
 *
 * - `embeddings`: vector array matching input order
 * - `usage.inputTokens`: summed across batched provider calls
 * - `calls`: batched provider calls made; attempts: total calls including retries
 */
export type LlmEmbedOutput = {
    embeddings: number[][];
    name: string;
    provider: string;
    model: string;
    usage: { inputTokens: number | null };
    ms: number;
    calls: number;
    attempts: number;
};

/**
 * **Error codes**: categories of LLM call failures.
 *
 * - `rate_limited`, `timeout`, `network`, `http`, `aborted`: transport and rate limit errors
 * - `blocked`, `truncated`, `invalid_json`: unusable completion responses
 * - `unsupported`: model lacks capability requested
 * - `invalid_request`: malformed request or options
 * - `limiter`: queue wait or quota refusal; unexpected: uncaught exception
 */
export type LlmErrorCode =
    | 'rate_limited'
    | 'timeout'
    | 'network'
    | 'http'
    | 'aborted'
    | 'blocked'
    | 'truncated'
    | 'invalid_json'
    | 'unsupported'
    | 'invalid_request'
    | 'limiter'
    | 'unexpected';

/**
 * **Failed call**: structured error details from LLM operation.
 *
 * - `code`, `message`, `retryable`: failure classification and retry guidance
 * - `output`: unusable answer details when completion arrived
 * - `settings`, `attempts`: applied settings and attempt count
 */
export type LlmError = {
    code: LlmErrorCode;
    message: string;
    retryable: boolean;
    status?: number;
    retryAfterMs?: number;
    limiterCode?: string;
    text?: string;
    model?: string;
    usage?: LlmUsage;
    ms?: number;
    name?: string;
    provider?: string;
    output?: LlmAnswer;
    settings?: LlmSettings;
    attempts?: number;
};

// -------------------------------------------------------------------
// --- 1.2 Providers
// -------------------------------------------------------------------

/**
 * **JSON schema delivery mode**: how JSON response_format reaches model.
 *
 * - `native`: schema in provider setting read by model (Gemini, OpenAI)
 * - `strict`: strict schema with required properties and nullable optionals (Groq)
 * - `grammar`: decoding constraint alongside system prompt schema instruction (Ollama)
 * - `prompt`: prompt-injected schema instruction with answer verification retries (DeepSeek)
 */
export type LlmJsonMode = 'native' | 'strict' | 'grammar' | 'prompt';

/**
 * **Provider API capabilities**: features supported by provider adapter.
 *
 * - omitted members treated as unsupported
 * - models may only declare capabilities supported by adapter
 */
export type LlmProviderSupport = {
    images?: true;
    tools?: true;
    webSearch?: true;
    reasoning?: true;
    batch?: true;
    json?: LlmJsonMode;
    embeddings?: true;
};

/**
 * **Declared model capabilities**: capabilities and default behaviors declared in configuration.
 *
 * - `images`, `tools`: enables vision and function calling features
 * - `json`: JSON mode configuration (boolean or specific LlmJsonMode)
 * - `webSearch`, `batch`: search and batch execution support with optional defaults
 * - `reasoning`: supported effort levels and default reasoning effort
 * - `embeddings`: embedding generation with batch input limits
 */
export type LlmModelCapabilities = {
    images?: boolean;
    tools?: boolean;
    json?: boolean | LlmJsonMode;
    webSearch?: { supported: boolean; default?: boolean };
    batch?: { supported: boolean; default?: boolean };
    reasoning?: { supported: boolean; default?: LlmReasoningEffort; levels?: readonly LlmReasoningEffort[] };
    embeddings?: boolean | { supported: boolean; maxInputs?: number };
};

/** Resolved model capabilities with populated defaults and adapter validation applied. */
export type LlmCapabilities = {
    images: boolean;
    tools: boolean;
    json: LlmJsonMode | false;
    webSearch: { supported: boolean; default: boolean };
    batch: { supported: boolean; default: boolean };
    reasoning: { supported: boolean; default?: LlmReasoningEffort; levels?: readonly LlmReasoningEffort[] };
    embeddings: { supported: boolean; maxInputs: number };
};

/**
 * **Adapter request payload**: normalized request prepared for provider adapter.
 *
 * - `model`: provider model ID with defaults merged and capabilities validated
 * - `max_tokens`: unified output limit (folds max_completion_tokens)
 * - `response_format` and `reasoning_effort`: mapped to adapter format
 */
export type LlmProviderRequest = Omit<LlmRequest, 'max_completion_tokens'>;

/** Raw completion returned by provider adapter before core metadata enrichment. */
export type LlmProviderOutput = Omit<LlmOutput, 'json' | 'name' | 'provider' | 'settings' | 'attempts'>;

/** Failure returned by provider adapter with optional raw output before core metadata enrichment. */
export type LlmProviderError = Omit<LlmError, 'output' | 'settings' | 'attempts'> & { output?: LlmProviderOutput };

/** Embedding request payload prepared for provider adapter. */
export type LlmProviderEmbedRequest = {
    model: string;
    input: { text: string; title?: string }[];
    task?: LlmEmbedTask;
    dimensions?: number;
};

/** Embedding vectors and usage returned by provider adapter. */
export type LlmProviderEmbedOutput = {
    embeddings: number[][];
    model: string;
    usage: { inputTokens: number | null };
    ms: number;
};

/** Model metadata returned by provider model listing API. */
export type LlmModelInfo = { id: string; displayName?: string; inputTokenLimit?: number; outputTokenLimit?: number };

/**
 * **Provider adapter interface**: connects core orchestration to underlying LLM API.
 *
 * - `generate`: performs single API call, returning result without retries
 * - `supports`: declares features and JSON modes supported by provider
 * - `validateExtra`/`mergeExtra`: validates and merges layered extra options
 * - `embed`: performs single embedding API call for up to maxEmbedInputs texts
 */
export type LlmProvider = {
    readonly name: string;
    readonly supports: LlmProviderSupport;
    generate(
        request: LlmProviderRequest,
        options: { signal: AbortSignal; extra: Record<string, unknown> },
    ): Promise<Result<LlmProviderOutput, LlmProviderError>>;
    listModels?(options?: { signal?: AbortSignal }): Promise<Result<LlmModelInfo[], LlmError>>;
    validateExtra?(extra: Record<string, unknown>, option: string): void;
    mergeExtra?(layers: Record<string, unknown>[]): Record<string, unknown>;
    embed?(
        request: LlmProviderEmbedRequest,
        options: { signal: AbortSignal },
    ): Promise<Result<LlmProviderEmbedOutput, LlmProviderError>>;
    readonly maxEmbedInputs?: number;
};

// -------------------------------------------------------------------
// --- 1.3 Configuration
// -------------------------------------------------------------------

/**
 * **Rate and concurrency limits**: smart-limit bucket configurations.
 *
 * - `sharedLimits`: shared across all models of provider (e.g. account-wide quotas)
 * - `modelLimits`: independent bucket limits per model
 * - `concurrency`, `rps`, `rpm`, `rpd`, `tpm`, `tpd`, `intervalMs`: built-in rate bucket shortcuts
 * - `buckets`: custom smart-limit Bucket instances
 */
export type LlmLimits = {
    concurrency?: number;
    rps?: number;
    rpm?: number;
    rpd?: number;
    tpm?: number;
    tpd?: number;
    intervalMs?: number;
    buckets?: readonly Bucket[];
};

/**
 * **Queue retry and wait options**: retry and timeout configuration for smart-limit queues.
 *
 * - `maxRetries`: maximum retry attempts on transient failures (default 4)
 * - `retryDelayMs` and `maxRetryDelayMs`: exponential backoff range
 * - `maxWaitMs`: maximum wait time for bucket space, lock release, or retry
 */
export type LlmLimiterOptions = {
    maxRetries?: number;
    retryDelayMs?: number;
    maxRetryDelayMs?: number;
    maxWaitMs?: number;
};

/**
 * **Model configuration**: model-specific parameters, limits, capabilities, and defaults.
 *
 * - `model`: provider model ID
 * - `modelLimits`: model-level concurrency and rate limits
 * - `capabilities`: declared features supported by model
 * - `charsPerToken`: model token estimation divisor override
 */
export type LlmModelSpec = {
    model: string;
    defaults?: LlmDefaults;
    modelLimits?: LlmLimits;
    charsPerToken?: number;
    capabilities?: LlmModelCapabilities;
    extra?: Record<string, unknown>;
};

/** Provider configuration: adapter instance, shared limits, defaults, and model mappings. */
export type LlmProviderSpec = {
    adapter: LlmProvider;
    defaults?: LlmDefaults;
    sharedLimits?: LlmLimits;
    charsPerToken?: number;
    extra?: Record<string, unknown>;
    models: Readonly<Record<string, string | LlmModelSpec>>;
};

/**
 * **LLM configuration**: root configuration for createLlm.
 *
 * - `providers`: provider adapters and their mapped model configurations
 * - `defaults`: root default request parameters inherited by all calls
 * - `limiterOptions`: global retry and queue timeout settings
 * - `charsPerToken`: default character-to-token ratio for limit booking (default 4)
 */
export type LlmConfig = {
    defaults?: LlmDefaults;
    limiterOptions?: LlmLimiterOptions;
    charsPerToken?: number;
    providers: Readonly<Record<string, LlmProviderSpec>>;
};

/** Extracts configured model names from configuration type. */
export type LlmModelNames<C extends LlmConfig> = {
    [P in keyof C['providers']]: keyof C['providers'][P]['models'] & string;
}[keyof C['providers']];

/** Resolved model metadata and capabilities. */
export type LlmModel = { name: string; provider: string; model: string; capabilities: LlmCapabilities };

/**
 * **Model input formats**: valid input types for model shortcut functions.
 *
 * - string: single user message text
 * - content parts: single user message containing text and image parts
 * - messages: full multi-turn conversation history
 */
export type LlmModelInput = string | readonly LlmContentPart[] | readonly LlmMessage[];

/**
 * **Model shortcut options**: request fields and call options for model shortcut calls.
 *
 * - excludes model and messages (provided by shortcut method and input)
 * - undefined values preserve configured defaults
 */
export type LlmModelOptions = {
    [K in keyof (Omit<LlmRequest, 'model' | 'messages'> & LlmCallOptions)]?:
        | (Omit<LlmRequest, 'model' | 'messages'> & LlmCallOptions)[K]
        | undefined;
};

/**
 * **Model shortcut function**: direct callable interface for configured model (e.g. llm.flash(input, options)).
 *
 * - routes input and options to generate with fixed model binding
 * - unbound function independent of this context
 * - never throws or rejects; invalid input fails with invalid_request
 */
export type LlmModelCall = (input: LlmModelInput, options?: LlmModelOptions) => Promise<Result<LlmOutput, LlmError>>;

/** Disallowed model names reserved by Llm instance methods and Object prototype. */
export type LlmReservedName =
    | 'generate'
    | 'embed'
    | 'models'
    | 'listModels'
    | 'then'
    | 'toJSON'
    | '__proto__'
    // biome-ignore lint/complexity/noBannedTypes: Object represents Object.prototype member names
    | keyof Object;

/**
 * **LLM instance**: multi-provider coordinator with rate limits and model shortcut functions.
 *
 * - `generate`: sends chat completion through model and provider limit queues
 * - `embed`: generates embeddings split across batched provider calls
 * - `models`: list of configured models and resolved capabilities
 * - `listModels`: queries models available from provider API
 * - dynamic model methods: `llm[modelName](input, options)`
 */
export type Llm<M extends string> = {
    generate(request: LlmRequest<M>, options?: LlmCallOptions): Promise<Result<LlmOutput, LlmError>>;
    embed(request: LlmEmbedRequest<M>, options?: { signal?: AbortSignal }): Promise<Result<LlmEmbedOutput, LlmError>>;
    readonly models: readonly LlmModel[];
    listModels(provider: string, options?: { signal?: AbortSignal }): Promise<Result<LlmModelInfo[], LlmError>>;
} & { readonly [K in Exclude<M, LlmReservedName>]: LlmModelCall };
