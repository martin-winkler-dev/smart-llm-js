// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { IMAGE_DETAILS } from './images.js';
import { cleanSchema, schemaInstruction, toStrictSchema } from './schema.js';
import type {
    LlmCapabilities,
    LlmDefaults,
    LlmError,
    LlmJsonSchema,
    LlmMessage,
    LlmProviderRequest,
    LlmReasoningEffort,
    LlmRequest,
    LlmToolCall,
    Result,
} from './types.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/**
 * **Prepared provider request**: provider payload and answer validation configuration.
 *
 * - `schema`: schema used to validate completion (`undefined` without `json_schema`)
 * - `json`: whether output must be valid JSON
 * - `strict`: whether optional null fields are stripped from completion
 */
export type PreparedRequest = {
    request: LlmProviderRequest;
    json: boolean;
    schema: LlmJsonSchema | undefined;
    strict: boolean;
};

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Fields of `LlmRequest`; any other field is rejected. */
const REQUEST_KEYS: ReadonlySet<string> = new Set([
    'model',
    'messages',
    'response_format',
    'temperature',
    'top_p',
    'max_tokens',
    'max_completion_tokens',
    'seed',
    'stop',
    'reasoning_effort',
    'web_search',
    'tools',
    'tool_choice',
    'batch',
]);

/** Fields of `LlmDefaults`. */
const DEFAULT_KEYS: ReadonlySet<string> = new Set([
    'temperature',
    'top_p',
    'max_tokens',
    'max_completion_tokens',
    'seed',
    'stop',
]);

/** Request fields that are no defaults, with where they go instead. */
const MOVED_DEFAULTS: Readonly<Record<string, string>> = {
    // biome-ignore lint/style/useNamingConvention: request field name
    web_search: "the model's capabilities.webSearch.default, or web_search per call",
    batch: "the model's capabilities.batch.default, or batch per call",
    // biome-ignore lint/style/useNamingConvention: request field name
    reasoning_effort: "the model's capabilities.reasoning.default, or reasoning_effort per call",
    // biome-ignore lint/style/useNamingConvention: request field name
    response_format: 'each call',
    tools: 'each call',
    // biome-ignore lint/style/useNamingConvention: request field name
    tool_choice: 'each call',
};

/** Values of `reasoning_effort`, lowest first. */
export const REASONING_EFFORTS: readonly LlmReasoningEffort[] = [
    'none',
    'minimal',
    'low',
    'medium',
    'high',
    'xhigh',
    'max',
];

/** Values of `reasoning_effort`. */
const EFFORTS: ReadonlySet<string> = new Set<string>(REASONING_EFFORTS);

/** Members of `LlmMessage` by role; a role not listed is rejected. */
const MESSAGE_KEYS: ReadonlyMap<string, ReadonlySet<string>> = new Map([
    ['system', new Set(['role', 'content'])],
    ['developer', new Set(['role', 'content'])],
    ['user', new Set(['role', 'content'])],
    ['assistant', new Set(['role', 'content', 'tool_calls', 'reasoning'])],
    ['tool', new Set(['role', 'tool_call_id', 'content'])],
]);

/** Members of a text part. */
const TEXT_PART_KEYS: ReadonlySet<string> = new Set(['type', 'text']);

/** Members of an image part. */
const IMAGE_PART_KEYS: ReadonlySet<string> = new Set(['type', 'image_url']);

/** Members of an image part's `image_url`. */
const IMAGE_URL_KEYS: ReadonlySet<string> = new Set(['url', 'detail']);

/** Members of an `LlmTool`, its `function`, an `LlmToolCall` and its `function`. */
const TOOL_KEYS: ReadonlySet<string> = new Set(['type', 'function']);
const TOOL_FUNCTION_KEYS: ReadonlySet<string> = new Set(['name', 'description', 'parameters', 'strict']);
const TOOL_CALL_KEYS: ReadonlySet<string> = new Set(['id', 'type', 'function', 'extra_content']);
const TOOL_CALL_FUNCTION_KEYS: ReadonlySet<string> = new Set(['name', 'arguments']);
const TOOL_CHOICE_FUNCTION_KEYS: ReadonlySet<string> = new Set(['name']);

/** Values of a string `tool_choice`. */
const TOOL_CHOICES: ReadonlySet<string> = new Set(['auto', 'none', 'required']);

/** Tool function name pattern: up to 64 letters, digits, _ and -. */
const MATCH_FUNCTION_NAME = /^[A-Za-z0-9_-]{1,64}$/;

/** Members of `response_format` by type; a type not listed is rejected. */
const FORMAT_KEYS: ReadonlyMap<string, ReadonlySet<string>> = new Map([
    ['text', new Set(['type'])],
    ['json_object', new Set(['type'])],
    ['json_schema', new Set(['type', 'json_schema'])],
]);

/** Members of `response_format.json_schema`. */
const JSON_SCHEMA_KEYS: ReadonlySet<string> = new Set(['name', 'schema', 'description', 'strict']);

/** Schema keywords holding one sub-schema; `items` may also hold an array (draft-07 tuple). */
const SUB_SCHEMA_KEYS: ReadonlySet<string> = new Set([
    'items',
    'additionalProperties',
    'not',
    'contains',
    'if',
    'then',
    'else',
    'propertyNames',
]);

/** Schema keywords holding an array of sub-schemas. */
const SUB_SCHEMA_LIST_KEYS: ReadonlySet<string> = new Set(['anyOf', 'oneOf', 'allOf', 'prefixItems']);

/** Schema keywords holding a name-to-sub-schema map. */
const SUB_SCHEMA_MAP_KEYS: ReadonlySet<string> = new Set(['properties', 'patternProperties', '$defs', 'definitions']);

/** Base64 data URL pattern of an image; at least one base64 character. */
const MATCH_IMAGE_DATA_URL = /^data:image\/[\w.+-]+;base64,[A-Za-z0-9+/]+={0,2}$/;

/** Message fields that only one role takes. */
const MESSAGE_FIELD_ROLES: readonly [field: string, role: string][] = [
    ['tool_calls', 'assistant'],
    ['reasoning', 'assistant'],
    ['tool_call_id', 'tool'],
];

/** Numeric request fields. */
const NUMBER_FIELDS = ['temperature', 'top_p', 'max_tokens', 'max_completion_tokens', 'seed'] as const;

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * Validates request payload structure, messages, parameters, and tools.
 * @throws {TypeError} If request structure or any field value is invalid.
 */
export function validateRequest(request: unknown): asserts request is LlmRequest {
    if (!isPlainObject(request)) {
        throw new TypeError('Invalid value for option "request"');
    }
    checkKeys(request, REQUEST_KEYS, '');
    if (typeof request.model !== 'string' || request.model === '') {
        throw new TypeError('Invalid value for option "model"');
    }
    const { messages } = request;
    if (!Array.isArray(messages) || !messages.some((message) => message?.role === 'user')) {
        throw new TypeError('Invalid value for option "messages": at least one user message');
    }
    const callIds = new Set<string>();
    for (const message of messages) {
        validateMessage(message, callIds);
    }
    validateFields(request, 'request');
}

/**
 * Validates default configuration values.
 * @throws {TypeError} If option contains unsupported keys or invalid option values.
 */
export function validateDefaults(defaults: unknown, option: string): asserts defaults is LlmDefaults {
    if (!isPlainObject(defaults)) {
        throw new TypeError(`Invalid value for option "${option}"`);
    }
    for (const key of Object.keys(defaults)) {
        const target = MOVED_DEFAULTS[key];
        if (target !== undefined) {
            throw new TypeError(`Option "${option}.${key}" is not supported: set it in ${target}`);
        }
    }
    checkKeys(defaults, DEFAULT_KEYS, `${option}.`);
    validateFields(defaults, option);
}

/**
 * **Merges layered defaults**: applies shared, provider, model, and request settings in priority order.
 *
 * - later layers override earlier ones
 * - folds `max_tokens` and `max_completion_tokens` into single limit
 * - deep-copies plain values, so result shares no objects or arrays with caller request or stored defaults
 * @throws {TypeError} If a field contains a circular reference.
 */
export function applyDefaults(request: LlmRequest, ...defaults: (LlmDefaults | undefined)[]): LlmRequest {
    const filled: Record<string, unknown> = {};
    for (const layer of [...defaults, request]) {
        if (layer === undefined) {
            continue;
        }
        if (layer.max_tokens !== undefined || layer.max_completion_tokens !== undefined) {
            delete filled.max_tokens;
            delete filled.max_completion_tokens;
        }
        for (const [key, value] of Object.entries(layer)) {
            if (value !== undefined) {
                filled[key] = copyPlain(value, key);
            }
        }
    }
    return filled as LlmRequest;
}

/**
 * **Prepares request for target model**: validates and applies model capabilities.
 *
 * - validates required capabilities (images, tools, web search, batch, reasoning)
 * - maps reasoning effort to supported model levels
 * - configures JSON mode, schema injection, or native format
 */
export function prepareRequest(
    request: LlmRequest,
    model: string,
    capabilities: LlmCapabilities,
): Result<PreparedRequest, LlmError> {
    const { max_completion_tokens: maxCompletionTokens, model: _name, ...fields } = request;
    const prepared: LlmProviderRequest = { ...fields, model };
    if (maxCompletionTokens !== undefined) {
        prepared.max_tokens = maxCompletionTokens;
    }
    if (!capabilities.images && request.messages.some(hasImage)) {
        return unsupported('images');
    }
    if (!capabilities.tools && (request.tools !== undefined || request.messages.some(isToolTurn))) {
        return unsupported('tools');
    }
    if (request.tools === undefined) {
        delete prepared.tool_choice;
    }
    const webSearch = request.web_search ?? capabilities.webSearch.default;
    if (webSearch && !capabilities.webSearch.supported) {
        return unsupported('web search');
    }
    delete prepared.web_search;
    if (webSearch) {
        prepared.web_search = true;
    }
    const batch = request.batch ?? capabilities.batch.default;
    if (batch && !capabilities.batch.supported) {
        return unsupported('batch calls');
    }
    delete prepared.batch;
    if (batch) {
        prepared.batch = true;
    }
    const { reasoning } = capabilities;
    const effort = request.reasoning_effort ?? (reasoning.supported ? reasoning.default : undefined);
    delete prepared.reasoning_effort;
    if (effort !== undefined && effort !== 'none' && !reasoning.supported) {
        return unsupported('reasoning');
    }
    if (effort !== undefined && reasoning.supported) {
        prepared.reasoning_effort = mapEffort(effort, reasoning.levels);
    }
    const format = request.response_format;
    if (format === undefined || format.type === 'text') {
        // `text` is the default of every provider: not forwarded
        delete prepared.response_format;
        return {
            ok: true,
            data: { request: prepared, json: false, schema: undefined, strict: false },
        };
    }
    const mode = capabilities.json;
    if (mode === false) {
        return unsupported('JSON output');
    }
    const schema = format.type === 'json_schema' ? format.json_schema.schema : undefined;
    if (mode === 'prompt' || mode === 'grammar') {
        prepared.messages = withInstruction(request.messages, schemaInstruction(schema && cleanSchema(schema)));
    }
    if (format.type === 'json_schema' && schema !== undefined && mode !== 'prompt') {
        const strict = mode === 'strict';
        prepared.response_format = {
            type: 'json_schema',
            // biome-ignore lint/style/useNamingConvention: OpenAI field name
            json_schema: {
                ...format.json_schema,
                schema: strict ? toStrictSchema(schema) : cleanSchema(schema),
                ...(strict ? { strict: true } : {}),
            },
        };
        return { ok: true, data: { request: prepared, json: true, schema, strict } };
    }
    prepared.response_format = { type: 'json_object' };
    return { ok: true, data: { request: prepared, json: true, schema, strict: false } };
}

/**
 * **Maps reasoning effort**: finds closest supported level declared by model.
 *
 * - preserves requested level if supported
 * - falls back to nearest available non-none level (higher on tie)
 */
export function mapEffort(
    effort: LlmReasoningEffort,
    levels: readonly LlmReasoningEffort[] | undefined,
): LlmReasoningEffort {
    if (levels === undefined || levels.includes(effort)) {
        return effort;
    }
    const wanted = REASONING_EFFORTS.indexOf(effort);
    let best: LlmReasoningEffort | undefined;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const level of levels) {
        if (level === 'none') {
            continue;
        }
        const distance = Math.abs(REASONING_EFFORTS.indexOf(level) - wanted);
        if (distance < bestDistance || (distance === bestDistance && best !== undefined && isHigher(level, best))) {
            best = level;
            bestDistance = distance;
        }
    }
    return best ?? effort;
}

/**
 * Validates object keys against allowed set.
 * @throws {TypeError} If value is not an object or contains unsupported keys.
 */
export function checkKeys(value: unknown, allowed: ReadonlySet<string>, prefix: string): void {
    if (!isPlainObject(value)) {
        throw new TypeError(`Invalid value for option "${prefix.slice(0, -1) || 'options'}"`);
    }
    for (const key of Object.keys(value)) {
        if (!allowed.has(key)) {
            throw new TypeError(`Option "${prefix}${key}" is not supported`);
        }
    }
}

/** Checks whether value is a non-null, non-array object. */
export function isPlainObject(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Creates deep clone of plain objects and arrays, preserving other values.
 * @throws {TypeError} If value contains a circular reference, naming `option`.
 */
export function copyPlain<T>(value: T, option = 'options'): T {
    return copyNode(value, option, new WeakSet()) as T;
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/**
 * **Validates message structure, role, and content parts.**
 *
 * - keys per role; `tool_calls` and array `content` need at least one entry
 * - parts: text, or base64 data URL image (with optional `detail`) in user messages
 * @throws {TypeError} If message structure or content is invalid.
 */
function validateMessage(message: unknown, callIds: Set<string>): void {
    const keys = isPlainObject(message) ? MESSAGE_KEYS.get(message.role as string) : undefined;
    if (!isPlainObject(message) || keys === undefined) {
        throw new TypeError('Invalid value for option "messages"');
    }
    for (const [field, role] of MESSAGE_FIELD_ROLES) {
        if (field in message && message.role !== role) {
            throw new TypeError(`Invalid value for option "messages": ${field} only in ${role} messages`);
        }
    }
    checkKeys(message, keys, 'messages.');
    const { content } = message;
    if (message.role === 'assistant') {
        const calls = message.tool_calls;
        if (calls !== undefined && (!Array.isArray(calls) || calls.length === 0)) {
            throw new TypeError('Invalid value for option "messages.tool_calls": one or more tool calls');
        }
        for (const call of (calls ?? []) as unknown[]) {
            validateToolCall(call);
            callIds.add((call as LlmToolCall).id);
        }
        if (message.reasoning !== undefined && typeof message.reasoning !== 'string') {
            throw new TypeError('Invalid value for option "messages.reasoning"');
        }
        if (content === null) {
            if (calls === undefined) {
                throw new TypeError('Invalid value for option "messages": content null only with tool_calls');
            }
            return;
        }
    }
    if (message.role === 'tool' && !callIds.has(message.tool_call_id as string)) {
        throw new TypeError(
            'Invalid value for option "messages.tool_call_id": the id of a tool call in an earlier assistant message',
        );
    }
    if (typeof content === 'string') {
        return;
    }
    if (!Array.isArray(content)) {
        throw new TypeError('Invalid value for option "messages"');
    }
    if (content.length === 0) {
        throw new TypeError('Invalid value for option "messages.content": a string or one or more parts');
    }
    for (const part of content) {
        validatePart(part, message.role as string);
    }
}

/**
 * Validates content part: text, or image with base64 data URL and optional `detail` in user message.
 * @throws {TypeError} If part type, keys, role, URL or detail is invalid.
 */
function validatePart(part: unknown, role: string): void {
    if (!isPlainObject(part) || (part.type !== 'text' && part.type !== 'image_url')) {
        throw new TypeError('Invalid value for option "messages"');
    }
    if (part.type === 'text') {
        checkKeys(part, TEXT_PART_KEYS, 'messages.content.');
        if (typeof part.text !== 'string') {
            throw new TypeError('Invalid value for option "messages"');
        }
        return;
    }
    if (role !== 'user') {
        throw new TypeError('Invalid value for option "messages": images only in user messages');
    }
    checkKeys(part, IMAGE_PART_KEYS, 'messages.content.');
    const image = part.image_url;
    if (!isPlainObject(image) || typeof image.url !== 'string' || !MATCH_IMAGE_DATA_URL.test(image.url)) {
        throw new TypeError('Invalid value for option "messages": an image must be a base64 data URL');
    }
    checkKeys(image, IMAGE_URL_KEYS, 'messages.content.image_url.');
    if (image.detail !== undefined && !IMAGE_DETAILS.has(image.detail)) {
        throw new TypeError('Invalid value for option "messages.content.image_url.detail": auto, low or high');
    }
}

/**
 * Validates assistant tool call structure and arguments.
 * @throws {TypeError} If tool call format or arguments JSON is invalid.
 */
function validateToolCall(call: unknown): void {
    checkKeys(call, TOOL_CALL_KEYS, 'messages.tool_calls.');
    const { id, type, function: fn, extra_content: extra } = call as Record<string, unknown>;
    if (typeof id !== 'string' || id === '') {
        throw new TypeError('Invalid value for option "messages.tool_calls.id"');
    }
    if (type !== 'function') {
        throw new TypeError('Invalid value for option "messages.tool_calls.type": function');
    }
    checkKeys(fn, TOOL_CALL_FUNCTION_KEYS, 'messages.tool_calls.function.');
    const { name, arguments: args } = fn as Record<string, unknown>;
    if (typeof name !== 'string' || name === '') {
        throw new TypeError('Invalid value for option "messages.tool_calls.function.name"');
    }
    if (typeof args !== 'string' || (args !== '' && !isPlainObject(parseJson(args)))) {
        throw new TypeError(
            'Invalid value for option "messages.tool_calls.function.arguments": JSON text of an object',
        );
    }
    if (extra !== undefined && !isPlainObject(extra)) {
        throw new TypeError('Invalid value for option "messages.tool_calls.extra_content"');
    }
}

/**
 * **Validates tools and tool_choice**: checks declarations and function names.
 *
 * - `tools`: non-empty list of functions with unique names (up to 64 letters, digits, `_`, `-`)
 * - `tool_choice`: `auto`, `none`, `required`, or named function present in `tools`
 * @throws {TypeError} If declarations or choices are invalid.
 */
function validateTools(fields: Record<string, unknown>): void {
    const { tools, tool_choice: choice } = fields;
    const names = new Set<string>();
    if (tools !== undefined) {
        if (!Array.isArray(tools) || tools.length === 0) {
            throw new TypeError('Invalid value for option "tools"');
        }
        for (const tool of tools as unknown[]) {
            checkKeys(tool, TOOL_KEYS, 'tools.');
            const { type, function: fn } = tool as Record<string, unknown>;
            if (type !== 'function') {
                throw new TypeError('Invalid value for option "tools.type": function');
            }
            checkKeys(fn, TOOL_FUNCTION_KEYS, 'tools.function.');
            const { name, description, parameters, strict } = fn as Record<string, unknown>;
            if (typeof name !== 'string' || !MATCH_FUNCTION_NAME.test(name)) {
                throw new TypeError(
                    'Invalid value for option "tools.function.name": up to 64 letters, digits, _ and -',
                );
            }
            if (names.has(name)) {
                throw new TypeError(`Invalid value for option "tools.function.name": ${name} is declared twice`);
            }
            names.add(name);
            if (description !== undefined && typeof description !== 'string') {
                throw new TypeError('Invalid value for option "tools.function.description"');
            }
            if (parameters !== undefined) {
                validateSchemaShape(parameters, 'tools.function.parameters');
            }
            if (strict !== undefined && strict !== null && typeof strict !== 'boolean') {
                throw new TypeError('Invalid value for option "tools.function.strict"');
            }
        }
    }
    if (choice === undefined || (typeof choice === 'string' && TOOL_CHOICES.has(choice))) {
        return;
    }
    if (typeof choice === 'string') {
        throw new TypeError('Invalid value for option "tool_choice": auto, none, required or a function');
    }
    checkKeys(choice, TOOL_KEYS, 'tool_choice.');
    const { type, function: fn } = choice as Record<string, unknown>;
    if (type !== 'function') {
        throw new TypeError('Invalid value for option "tool_choice.type": function');
    }
    checkKeys(fn, TOOL_CHOICE_FUNCTION_KEYS, 'tool_choice.function.');
    const { name } = fn as Record<string, unknown>;
    if (typeof name !== 'string' || (tools !== undefined && !names.has(name))) {
        throw new TypeError('Invalid value for option "tool_choice.function.name": the name of one of "tools"');
    }
}

/** Parses JSON text; returns undefined on parse failure. */
function parseJson(text: string): unknown {
    try {
        return JSON.parse(text);
    } catch {
        return undefined;
    }
}

/**
 * Validates request and default fields (sampling parameters, stop sequences, format).
 * @throws {TypeError} If any parameter value or combination is invalid.
 */
function validateFields(fields: Record<string, unknown>, option: string): void {
    for (const field of NUMBER_FIELDS) {
        const value = fields[field];
        if (value !== undefined && (typeof value !== 'number' || !Number.isFinite(value))) {
            throw new TypeError(`Invalid value for option "${field}"`);
        }
    }
    for (const field of ['max_tokens', 'max_completion_tokens'] as const) {
        const value = fields[field];
        if (value !== undefined && (!Number.isInteger(value) || (value as number) <= 0)) {
            throw new TypeError(`Invalid value for option "${field}"`);
        }
    }
    if (fields.max_tokens !== undefined && fields.max_completion_tokens !== undefined) {
        throw new TypeError(`Option "max_tokens" is not supported together with "max_completion_tokens" in ${option}`);
    }
    const { stop } = fields;
    if (
        stop !== undefined &&
        typeof stop !== 'string' &&
        !(Array.isArray(stop) && stop.every((entry) => typeof entry === 'string'))
    ) {
        throw new TypeError('Invalid value for option "stop"');
    }
    if (fields.reasoning_effort !== undefined && !EFFORTS.has(fields.reasoning_effort as string)) {
        throw new TypeError('Invalid value for option "reasoning_effort"');
    }
    if (fields.web_search !== undefined && typeof fields.web_search !== 'boolean') {
        throw new TypeError('Invalid value for option "web_search": true or false');
    }
    if (fields.batch !== undefined && typeof fields.batch !== 'boolean') {
        throw new TypeError('Invalid value for option "batch": true or false');
    }
    validateTools(fields);
    const format = fields.response_format as { type?: unknown; [key: string]: unknown } | undefined;
    if (format === undefined) {
        return;
    }
    const keys = isPlainObject(format) ? FORMAT_KEYS.get(format.type as string) : undefined;
    if (keys === undefined) {
        throw new TypeError('Invalid value for option "response_format"');
    }
    checkKeys(format, keys, 'response_format.');
    if (format.type !== 'json_schema') {
        return;
    }
    const spec = format.json_schema;
    checkKeys(spec, JSON_SCHEMA_KEYS, 'response_format.json_schema.');
    const { name, schema, description, strict } = spec as Record<string, unknown>;
    if (typeof name !== 'string' || !MATCH_FUNCTION_NAME.test(name)) {
        throw new TypeError(
            'Invalid value for option "response_format.json_schema.name": up to 64 letters, digits, _ and -',
        );
    }
    if (description !== undefined && typeof description !== 'string') {
        throw new TypeError('Invalid value for option "response_format.json_schema.description"');
    }
    if (strict !== undefined && strict !== null && typeof strict !== 'boolean') {
        throw new TypeError('Invalid value for option "response_format.json_schema.strict"');
    }
    if (schema !== undefined) {
        validateSchemaShape(schema, 'response_format.json_schema.schema');
    }
}

/**
 * **Validates schema shape**: catches malformed sub-schemas before any API call; no full JSON Schema validation.
 *
 * - root: plain object; sub-schemas: plain object or boolean
 * - single sub-schema keys: `items` (or array of them, draft-07 tuple), `additionalProperties`, `not`, `contains`, `if`/`then`/`else`, `propertyNames`
 * - array keys: `anyOf`, `oneOf`, `allOf`, `prefixItems`; map keys: `properties`, `patternProperties`, `$defs`, `definitions`
 * - no circular references; same sub-schema reused in separate branches allowed
 * @throws {TypeError} If schema or one of these nodes is invalid, with the path of that node.
 */
function validateSchemaShape(schema: unknown, option: string): void {
    if (!isPlainObject(schema)) {
        throw new TypeError(`Invalid value for option "${option}": a JSON Schema object`);
    }
    checkSubSchema(schema, option, new WeakSet());
}

/**
 * Validates sub-schema node and its nested sub-schemas recursively; `seen` holds nodes on current path.
 * @throws {TypeError} If node or container is malformed, or node is cyclic.
 */
function checkSubSchema(schema: unknown, option: string, seen: WeakSet<object>): void {
    if (typeof schema === 'boolean') {
        return;
    }
    if (!isPlainObject(schema)) {
        throw new TypeError(`Invalid value for option "${option}": a JSON Schema object or boolean`);
    }
    if (seen.has(schema)) {
        throw new TypeError(`Invalid value for option "${option}": a JSON Schema without circular references`);
    }
    seen.add(schema);
    for (const [key, value] of Object.entries(schema)) {
        const path = `${option}.${key}`;
        if (value === undefined) {
            continue;
        }
        if (key === 'items' && Array.isArray(value)) {
            checkSchemaList(value, path, seen);
        } else if (SUB_SCHEMA_KEYS.has(key)) {
            checkSubSchema(value, path, seen);
        } else if (SUB_SCHEMA_LIST_KEYS.has(key)) {
            if (!Array.isArray(value)) {
                throw new TypeError(`Invalid value for option "${path}": an array of JSON Schemas`);
            }
            checkSchemaList(value, path, seen);
        } else if (SUB_SCHEMA_MAP_KEYS.has(key)) {
            if (!isPlainObject(value)) {
                throw new TypeError(`Invalid value for option "${path}": an object of JSON Schemas`);
            }
            for (const [name, entry] of Object.entries(value)) {
                checkSubSchema(entry, `${path}.${name}`, seen);
            }
        }
    }
    seen.delete(schema);
}

/**
 * Validates each sub-schema of array container.
 * @throws {TypeError} If entry is malformed or cyclic.
 */
function checkSchemaList(list: unknown[], option: string, seen: WeakSet<object>): void {
    for (const [index, entry] of list.entries()) {
        checkSubSchema(entry, `${option}.${index}`, seen);
    }
}

/**
 * Copies plain node recursively; `ancestors` holds the objects on the current path.
 * @throws {TypeError} If node is its own ancestor.
 */
function copyNode(value: unknown, option: string, ancestors: WeakSet<object>): unknown {
    const isArray = Array.isArray(value);
    const prototype: unknown = isPlainObject(value) ? Object.getPrototypeOf(value) : undefined;
    if (!isArray && prototype !== Object.prototype && prototype !== null) {
        return value;
    }
    const node = value as object;
    if (ancestors.has(node)) {
        throw new TypeError(`Invalid value for option "${option}": a value without circular references`);
    }
    ancestors.add(node);
    const copy = isArray
        ? (value as unknown[]).map((entry) => copyNode(entry, option, ancestors))
        : Object.fromEntries(Object.entries(node).map(([key, entry]) => [key, copyNode(entry, option, ancestors)]));
    ancestors.delete(node);
    return copy;
}

/** Injects system instruction immediately after leading system/developer messages. */
function withInstruction(messages: LlmMessage[], instruction: string): LlmMessage[] {
    const index = messages.findIndex((message) => message.role !== 'system' && message.role !== 'developer');
    const at = index === -1 ? messages.length : index;
    return [...messages.slice(0, at), { role: 'system', content: instruction }, ...messages.slice(at)];
}

/** Checks whether message is part of tool round trip (assistant tool calls or tool result). */
function isToolTurn(message: LlmMessage): boolean {
    return message.role === 'tool' || (message.role === 'assistant' && (message.tool_calls?.length ?? 0) > 0);
}

/** Checks whether message contains image content part. */
function hasImage(message: LlmMessage): boolean {
    return Array.isArray(message.content) && message.content.some((part) => part.type === 'image_url');
}

/** Checks whether reasoning effort a is higher than b. */
function isHigher(a: LlmReasoningEffort, b: LlmReasoningEffort): boolean {
    return REASONING_EFFORTS.indexOf(a) > REASONING_EFFORTS.indexOf(b);
}

/** Creates unsupported error result for missing capability. */
function unsupported(capability: string): { ok: false; error: LlmError } {
    return {
        ok: false,
        error: { code: 'unsupported', message: `The model does not support ${capability}`, retryable: false },
    };
}
