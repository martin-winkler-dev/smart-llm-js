// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { Anthropic } from '@anthropic-ai/sdk';
import {
    type LlmContentPart,
    type LlmJsonSchema,
    type LlmMessage,
    type LlmProviderError,
    type LlmProviderRequest,
    type LlmReasoningEffort,
    type LlmTextPart,
    type LlmToolCall,
    type LlmToolChoice,
    type Result,
    schemaInstruction,
} from '@smart-llm/core';
import type { AnthropicExtra } from './index.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** User or assistant turn of Messages API. */
type Turn = Anthropic.UserMessageParam | Anthropic.AssistantMessageParam;

/** Reasoning settings of Messages params. */
type Reasoning = { thinking?: Anthropic.ThinkingConfigParam; effort?: NonNullable<Anthropic.OutputConfig['effort']> };

/** Image media type accepted by Anthropic. */
type ImageMediaType = Anthropic.Base64ImageSource['media_type'];

/** JSON schema with root `type: 'object'`. */
type ObjectSchema = LlmJsonSchema & { type: 'object' };

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Image media types accepted by Anthropic. */
const IMAGE_MEDIA_TYPES: ReadonlySet<string> = new Set(['image/jpeg', 'image/png', 'image/gif', 'image/webp']);

/** Base64 data URL: media type (1) and data (2). */
const MATCH_DATA_URL = /^data:([^;,]+);base64,(.*)$/s;

/** Block types of thinking, replayed only within current tool loop. */
const THINKING_TYPES: ReadonlySet<string> = new Set(['thinking', 'redacted_thinking']);

/** JSON schema keywords holding maps of named subschemas. */
const SCHEMA_MAP_KEYS: ReadonlySet<string> = new Set(['properties', 'patternProperties', '$defs', 'definitions']);

/** JSON schema keywords holding definitions, transformed per entry. */
const DEFINITION_KEYS = ['$defs', 'definitions'] as const;

/** String formats kept in strict schemas (same as SDK `transformJSONSchema`). */
const SUPPORTED_STRING_FORMATS: ReadonlySet<string> = new Set([
    'date-time',
    'time',
    'date',
    'duration',
    'email',
    'hostname',
    'uri',
    'ipv4',
    'ipv6',
    'uuid',
]);

/** Error of strict schema node without type (same text as SDK `transformJSONSchema`). */
const MISSING_TYPE = 'JSON schema must have a type defined if anyOf/oneOf/allOf are not used';

/** Error of boolean sub-schema (`true`/`false`), not expressible in strict subset. */
const UNSUPPORTED_BOOLEAN = 'boolean schemas (true/false) are not supported in Anthropic strict mode';

/** Error of draft-07 tuple `items` (array of schemas). */
const UNSUPPORTED_TUPLE = 'tuple items (array of schemas) are not supported in Anthropic strict mode';

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Builds Messages API params from provider request**.
 *
 * - hoists all system/developer messages into `system`; merges consecutive same-role turns
 * - replays assistant tool turns verbatim from raw content; thinking only within current tool loop (after last user turn without `tool_result`)
 * - maps tools, web search, tool choice, stop, reasoning effort and JSON schema (strict schemas via own transform)
 * - `invalid_request`: user turn without content, unconvertible or non-object root schema, non-object tool call arguments
 * - drops temperature, top_p and seed (rejected by API version 2026-10-21)
 * - deep-merges `extra.params` last (`null` deletes key)
 */
export function toMessageParams(
    request: LlmProviderRequest,
    extra: AnthropicExtra,
    defaultMaxTokens: number,
): Result<Anthropic.MessageCreateParamsNonStreaming, LlmProviderError> {
    const maxTokens = request.max_tokens ?? defaultMaxTokens;
    const turns = toTurns(request.messages);
    if (!turns.ok) {
        return turns;
    }
    const params: Anthropic.MessageCreateParamsNonStreaming = {
        model: request.model,
        // biome-ignore lint/style/useNamingConvention: Anthropic field name
        max_tokens: maxTokens,
        messages: turns.data,
    };
    const system = toSystem(request);
    if (system.length > 0) {
        params.system = system;
    }
    const tools = toTools(request, extra);
    if (!tools.ok) {
        return tools;
    }
    if (tools.data.length > 0) {
        params.tools = tools.data;
    }
    if (request.tool_choice !== undefined) {
        params.tool_choice = toToolChoice(request.tool_choice);
    }
    const stop = typeof request.stop === 'string' ? [request.stop] : (request.stop ?? []);
    if (stop.length > 0) {
        params.stop_sequences = [...stop];
    }
    const reasoning = toReasoning(request.reasoning_effort, extra, maxTokens);
    if (!reasoning.ok) {
        return reasoning;
    }
    if (reasoning.data.thinking !== undefined) {
        params.thinking = reasoning.data.thinking;
    }
    if (reasoning.data.effort !== undefined) {
        params.output_config = { effort: reasoning.data.effort };
    }
    const format = request.response_format;
    if (format?.type === 'json_schema' && format.json_schema.schema !== undefined) {
        const label = `response format ${format.json_schema.name}`;
        const root = toObjectSchema(format.json_schema.schema, `JSON schema of ${label}`);
        if (!root.ok) {
            return root;
        }
        const schema = toStrictSchema(root.data, label);
        if (!schema.ok) {
            return schema;
        }
        params.output_config = { ...params.output_config, format: { type: 'json_schema', schema: schema.data } };
    }
    if (extra.params === undefined) {
        return { ok: true, data: params };
    }
    const merged = mergeDeep(params as unknown as Record<string, unknown>, extra.params, true);
    return { ok: true, data: merged as unknown as Anthropic.MessageCreateParamsNonStreaming };
}

/** Checks whether value is plain object (object literal or null prototype). */
export function isPlainObject(value: unknown): value is Record<string, unknown> {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        return false;
    }
    const prototype: unknown = Object.getPrototypeOf(value);
    return prototype === Object.prototype || prototype === null;
}

/**
 * **Deep-merges `patch` over `base`** without mutating either.
 *
 * - plain objects merged recursively; arrays and other values replace
 * - `undefined` values skipped; `__proto__` keys ignored
 * - `deleteNull`: `null` value deletes key (else replaces like other values)
 */
export function mergeDeep(
    base: Record<string, unknown>,
    patch: Record<string, unknown>,
    deleteNull: boolean,
): Record<string, unknown> {
    const merged: Record<string, unknown> = { ...base };
    for (const [key, value] of Object.entries(patch)) {
        if (value === undefined || key === '__proto__') {
            continue;
        }
        if (value === null && deleteNull) {
            delete merged[key];
            continue;
        }
        const current = merged[key];
        merged[key] = isPlainObject(value)
            ? mergeDeep(isPlainObject(current) ? current : {}, value, deleteNull)
            : value;
    }
    return merged;
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/** Collects system/developer texts (any position) plus JSON instruction when needed. */
function toSystem(request: LlmProviderRequest): Anthropic.TextBlockParam[] {
    const blocks: Anthropic.TextBlockParam[] = [];
    for (const message of request.messages) {
        if (message.role === 'system' || message.role === 'developer') {
            blocks.push(...toTextBlocks(message.content));
        }
    }
    const format = request.response_format;
    const unschematized =
        format?.type === 'json_object' || (format?.type === 'json_schema' && format.json_schema.schema === undefined);
    const instruction = schemaInstruction(undefined);
    // core prompt mode may already have injected the instruction
    if (unschematized && !blocks.some((block) => block.text.startsWith(instruction))) {
        blocks.push({ type: 'text', text: instruction });
    }
    return blocks;
}

/** Maps string or parts to text blocks, skipping empty texts and non-text parts. */
function toTextBlocks(content: string | readonly LlmContentPart[] | null): Anthropic.TextBlockParam[] {
    if (content === null) {
        return [];
    }
    if (typeof content === 'string') {
        return content === '' ? [] : [{ type: 'text', text: content }];
    }
    return content.flatMap((part) =>
        part.type === 'text' && part.text !== '' ? [{ type: 'text' as const, text: part.text }] : [],
    );
}

/** Maps non-system messages to turns, merging consecutive same-role turns; user turn without content fails. */
function toTurns(messages: readonly LlmMessage[]): Result<Turn[], LlmProviderError> {
    const turns: Turn[] = [];
    for (const message of messages) {
        if (message.role === 'tool') {
            pushTurn(turns, { role: 'user', content: [toToolResult(message.tool_call_id, message.content)] });
        } else if (message.role === 'assistant') {
            const content = toAssistantContent(message.content, message.tool_calls ?? []);
            if (!content.ok) {
                return content;
            }
            if (content.data !== undefined) {
                pushTurn(turns, { role: 'assistant', content: content.data });
            }
        } else if (message.role === 'user') {
            const content = toUserContent(message.content);
            if (!content.ok) {
                return content;
            }
            pushTurn(turns, { role: 'user', content: content.data });
        }
    }
    stripEarlierThinking(turns);
    if (turns.some((turn) => toBlocks(turn.content).length === 0)) {
        return invalidRequest('A user message has no content (Anthropic rejects empty text)');
    }
    return { ok: true, data: turns };
}

/** Appends turn, or merges it into previous turn of same role; merged user turn starts with its `tool_result` blocks. */
function pushTurn(turns: Turn[], turn: Turn): void {
    const last = turns.at(-1);
    if (last === undefined || last.role !== turn.role) {
        turns.push(turn);
        return;
    }
    const blocks = [...toBlocks(last.content), ...toBlocks(turn.content)];
    last.content =
        turn.role === 'user'
            ? [...blocks.filter(isToolResult), ...blocks.filter((block) => !isToolResult(block))]
            : blocks;
}

/** Checks whether block is `tool_result`. */
function isToolResult(block: Anthropic.ContentBlockParam): boolean {
    return block.type === 'tool_result';
}

/** Strips thinking blocks of assistant turns before current tool loop (last user turn without `tool_result`). */
function stripEarlierThinking(turns: Turn[]): void {
    const loopStart = turns.findLastIndex((turn) => turn.role === 'user' && !toBlocks(turn.content).some(isToolResult));
    for (const turn of turns.slice(0, Math.max(loopStart, 0))) {
        if (turn.role === 'assistant' && typeof turn.content !== 'string') {
            turn.content = turn.content.filter((block) => !THINKING_TYPES.has(block.type));
        }
    }
}

/** Converts turn content to blocks (string becomes text block; empty string none). */
function toBlocks(content: Anthropic.MessageContentParam): Anthropic.ContentBlockParam[] {
    if (typeof content !== 'string') {
        return content;
    }
    return content === '' ? [] : [{ type: 'text', text: content }];
}

/** Maps user content: non-empty texts, base64 data URL images (JPEG, PNG, GIF, WebP). */
function toUserContent(
    content: string | readonly LlmContentPart[],
): Result<Anthropic.MessageContentParam, LlmProviderError> {
    if (typeof content === 'string') {
        return { ok: true, data: content === '' ? [] : content };
    }
    const blocks: Anthropic.ContentBlockParam[] = [];
    for (const part of content) {
        if (part.type === 'text') {
            if (part.text !== '') {
                blocks.push({ type: 'text', text: part.text });
            }
            continue;
        }
        const match = MATCH_DATA_URL.exec(part.image_url.url);
        if (match === null) {
            return invalidRequest('Anthropic accepts only base64 data URL images');
        }
        const [, mediaType = '', data = ''] = match;
        if (!IMAGE_MEDIA_TYPES.has(mediaType)) {
            return invalidRequest(`Anthropic accepts JPEG, PNG, GIF and WebP images, not ${mediaType}`);
        }
        blocks.push({
            type: 'image',
            // biome-ignore lint/style/useNamingConvention: Anthropic field name
            source: { type: 'base64', media_type: mediaType as ImageMediaType, data },
        });
    }
    return { ok: true, data: blocks };
}

/**
 * **Maps assistant message to content**: undefined when nothing to send.
 *
 * - raw content of first tool call (`tool_use` ids equal call ids): sent verbatim in original order, minus empty texts
 * - thinking blocks of raw content stripped later outside current tool loop (see `stripEarlierThinking`)
 * - otherwise rebuilt: non-empty texts + `tool_use` per call (no thinking)
 * - only non-empty string and nothing else → string content
 * - `reasoning` never sent (has no signature)
 */
function toAssistantContent(
    content: string | readonly LlmTextPart[] | null,
    calls: readonly LlmToolCall[],
): Result<Anthropic.MessageContentParam | undefined, LlmProviderError> {
    const raw = readRawContent(calls);
    if (raw !== undefined) {
        return { ok: true, data: raw.filter((block) => !(block.type === 'text' && block.text === '')) };
    }
    if (calls.length === 0 && typeof content === 'string' && content !== '') {
        return { ok: true, data: content };
    }
    const blocks: Anthropic.ContentBlockParam[] = toTextBlocks(content);
    for (const call of calls) {
        const input = parseArguments(call);
        if (!input.ok) {
            return input;
        }
        blocks.push({ type: 'tool_use', id: call.id, name: call.function.name, input: input.data });
    }
    return { ok: true, data: blocks.length === 0 ? undefined : blocks };
}

/**
 * **Reads raw content blocks** from first tool call's `extra_content.anthropic.content`.
 *
 * - every entry must be plain object with string `type`
 * - `tool_use` ids must equal tool call ids (same set and count); else undefined (rebuild fallback)
 */
function readRawContent(calls: readonly LlmToolCall[]): Anthropic.ContentBlockParam[] | undefined {
    const anthropic = calls[0]?.extra_content?.anthropic;
    const content: unknown = isPlainObject(anthropic) ? anthropic.content : undefined;
    if (!Array.isArray(content) || !content.every((block) => isPlainObject(block) && typeof block.type === 'string')) {
        return undefined;
    }
    const blocks = content as Anthropic.ContentBlockParam[];
    const ids = blocks.flatMap((block) => (block.type === 'tool_use' ? [block.id] : []));
    const callIds = new Set(calls.map((call) => call.id));
    const same =
        ids.length === calls.length && new Set(ids).size === callIds.size && ids.every((id) => callIds.has(id));
    return same ? blocks : undefined;
}

/** Parses tool call arguments (`''` → `{}`); JSON other than plain object → `invalid_request`. */
function parseArguments(call: LlmToolCall): Result<Record<string, unknown>, LlmProviderError> {
    if (call.function.arguments === '') {
        return { ok: true, data: {} };
    }
    let parsed: unknown;
    try {
        parsed = JSON.parse(call.function.arguments);
    } catch {
        return invalidRequest(`The arguments of tool call ${call.function.name} are not valid JSON`);
    }
    if (!isPlainObject(parsed)) {
        return invalidRequest(`The arguments of tool call ${call.function.name} are not a JSON object`);
    }
    return { ok: true, data: parsed };
}

/** Maps tool message to `tool_result` block; empty texts skipped, no `content` when nothing left. */
function toToolResult(toolUseId: string, content: string | readonly LlmTextPart[]): Anthropic.ToolResultBlockParam {
    const blocks = toTextBlocks(content);
    return {
        type: 'tool_result',
        // biome-ignore lint/style/useNamingConvention: Anthropic field name
        tool_use_id: toolUseId,
        ...(blocks.length === 0 ? {} : { content: typeof content === 'string' ? content : blocks }),
    };
}

/** Maps function tools and appends web search server tool when requested. */
function toTools(request: LlmProviderRequest, extra: AnthropicExtra): Result<Anthropic.ToolUnion[], LlmProviderError> {
    const tools: Anthropic.ToolUnion[] = [];
    for (const { function: tool } of request.tools ?? []) {
        const strict = tool.strict === true;
        const schema = toInputSchema(tool.name, tool.parameters, strict);
        if (!schema.ok) {
            return schema;
        }
        tools.push({
            name: tool.name,
            ...(tool.description === undefined ? {} : { description: tool.description }),
            // biome-ignore lint/style/useNamingConvention: Anthropic field name
            input_schema: schema.data,
            ...(strict ? { strict: true } : {}),
        });
    }
    if (request.web_search === true) {
        tools.push({ type: 'web_search_20260209', name: 'web_search', ...extra.webSearchTool } as Anthropic.ToolUnion);
    }
    return { ok: true, data: tools };
}

/**
 * **Maps tool parameters to input schema** without mutating them.
 *
 * - missing → `{ type: 'object', properties: {} }`; root checked by `toObjectSchema`
 * - `strict`: converted by own transform (see `toStrictSchema`)
 */
function toInputSchema(
    name: string,
    parameters: LlmJsonSchema | undefined,
    strict: boolean,
): Result<Anthropic.Tool.InputSchema, LlmProviderError> {
    const root = toObjectSchema(parameters ?? { properties: {} }, `parameters of tool ${name}`);
    return root.ok && strict ? toStrictSchema(root.data, `tool ${name}`) : root;
}

/** Checks root schema is object schema: missing `type` → `object`, other `type` → `invalid_request`. */
function toObjectSchema(schema: LlmJsonSchema, subject: string): Result<ObjectSchema, LlmProviderError> {
    const type = schema.type ?? 'object';
    if (type !== 'object') {
        return invalidRequest(`The ${subject} must be an object schema, not type ${JSON.stringify(type)}`);
    }
    return { ok: true, data: { ...schema, type: 'object' } };
}

/** Maps OpenAI tool choice (`required` → `any`, function → `tool`). */
function toToolChoice(choice: LlmToolChoice): Anthropic.ToolChoice {
    if (choice === 'auto' || choice === 'none') {
        return { type: choice };
    }
    if (choice === 'required') {
        return { type: 'any' };
    }
    return { type: 'tool', name: choice.function.name };
}

/**
 * **Maps reasoning effort to thinking config and output effort**.
 *
 * - `none`: by `extra.thinkingOff` (`disabled` default, `between_tools`, `omit`)
 * - `extra.thinkingBudgets`: manual `enabled` thinking with budget below max_tokens; no output effort
 * - otherwise: `adaptive` thinking + output effort (`minimal` → `low`)
 */
function toReasoning(
    effort: LlmReasoningEffort | undefined,
    extra: AnthropicExtra,
    maxTokens: number,
): Result<Reasoning, LlmProviderError> {
    if (effort === undefined) {
        return { ok: true, data: {} };
    }
    if (effort === 'none') {
        const off = extra.thinkingOff ?? 'disabled';
        return { ok: true, data: off === 'omit' ? {} : { thinking: { type: off } } };
    }
    const display = extra.thinkingDisplay ?? 'summarized';
    if (extra.thinkingBudgets === undefined) {
        return {
            ok: true,
            data: { thinking: { type: 'adaptive', display }, effort: effort === 'minimal' ? 'low' : effort },
        };
    }
    const budget = extra.thinkingBudgets[effort];
    if (budget === undefined) {
        return invalidRequest(
            `extra.thinkingBudgets has no budget for reasoning effort ${effort}; declare capabilities.reasoning.levels matching the budgets`,
        );
    }
    if (budget >= maxTokens) {
        return invalidRequest(`The thinking budget ${budget} must be below max_tokens ${maxTokens}`);
    }
    // biome-ignore lint/style/useNamingConvention: Anthropic field name
    return { ok: true, data: { thinking: { type: 'enabled', budget_tokens: budget, display } } };
}

/**
 * **Converts object root schema to Anthropic strict subset** via `toStrictNode` (works on deep copy).
 *
 * - root `type: 'object'` set after transform (dropped next to root `anyOf`/`allOf`/`$ref`); no object rules beside them
 * - node `toStrictNode` rejects or non-JSON value → `invalid_request`
 * - open or map objects (`additionalProperties` other than `false`) → `invalid_request` (transform would close them silently)
 */
function toStrictSchema(schema: ObjectSchema, label: string): Result<ObjectSchema, LlmProviderError> {
    if (hasOpenObject(schema)) {
        return invalidRequest(
            `The JSON schema of ${label} has additionalProperties other than false, which Anthropic structured outputs do not support`,
        );
    }
    try {
        return { ok: true, data: { ...toStrictNode(JSON.parse(JSON.stringify(schema))), type: 'object' } };
    } catch (error) {
        const reason = error instanceof Error ? error.message : String(error);
        return invalidRequest(`The JSON schema of ${label} cannot be converted for Anthropic: ${reason}`);
    }
}

/**
 * **Converts schema node to strict subset**: mirrors SDK `transformJSONSchema`, consumes `node`.
 *
 * - `$defs` and `definitions` transformed per entry; `$ref` node keeps only `$ref` (and definitions)
 * - first of `anyOf`/`oneOf` (→ `anyOf`)/`allOf` kept without `type`; `enum`/`const` kept
 * - object/string/array rules when `type` is or includes that type (see `applyTypeRules`)
 * - object/string/array rules only without kept `anyOf`/`allOf` (unlike SDK: rules next to them over-constrain)
 * - other keywords (e.g. `minLength`, `minimum`) moved into `description` as `{key: JSON, ...}`
 * @throws {Error} If node is boolean schema, tuple `items`, not plain object or without `type`/`anyOf`/`oneOf`/`allOf`/`$ref`/`enum`/`const`.
 */
function toStrictNode(node: unknown): Record<string, unknown> {
    if (typeof node === 'boolean') {
        throw new Error(UNSUPPORTED_BOOLEAN);
    }
    if (!isPlainObject(node)) {
        throw new Error(MISSING_TYPE);
    }
    const strict: Record<string, unknown> = {};
    for (const key of DEFINITION_KEYS) {
        const definitions = pop(node, key);
        if (definitions !== undefined) {
            strict[key] = toStrictMap(definitions);
        }
    }
    const ref = pop(node, '$ref');
    if (ref !== undefined) {
        strict.$ref = ref;
        return strict;
    }
    const type = pop(node, 'type');
    const anyOf = pop(node, 'anyOf');
    const oneOf = pop(node, 'oneOf');
    const allOf = pop(node, 'allOf');
    const values = pop(node, 'enum');
    const constant = pop(node, 'const');
    if (Array.isArray(anyOf) || Array.isArray(oneOf)) {
        strict.anyOf = (Array.isArray(anyOf) ? anyOf : (oneOf as unknown[])).map(toStrictNode);
    } else if (Array.isArray(allOf)) {
        strict.allOf = allOf.map(toStrictNode);
    } else if (type !== undefined) {
        strict.type = type;
    } else if (!Array.isArray(values) && constant === undefined) {
        throw new Error(MISSING_TYPE);
    }
    const kept = { enum: values, const: constant, description: pop(node, 'description'), title: pop(node, 'title') };
    for (const [key, value] of Object.entries(kept)) {
        if (value !== undefined) {
            strict[key] = value;
        }
    }
    if (strict.anyOf === undefined && strict.allOf === undefined) {
        applyTypeRules(node, strict, Array.isArray(type) ? type : [type]);
    }
    if (Object.keys(node).length > 0) {
        const rest = Object.entries(node).map(([key, value]) => `${key}: ${JSON.stringify(value)}`);
        strict.description = `${strict.description ? `${strict.description}\n\n` : ''}{${rest.join(', ')}}`;
    }
    return strict;
}

/**
 * **Moves type keywords of `node` into `strict`**: SDK rules, also for `type` arrays.
 *
 * - `object`: `properties` transformed (default `{}`), `additionalProperties: false`, `required` kept
 * - `string`: supported `format` kept
 * - `array`: `items` transformed, `minItems` kept when `0` or `1`
 * - keywords not kept stay in `node` (moved into `description` later)
 */
function applyTypeRules(node: Record<string, unknown>, strict: Record<string, unknown>, types: unknown[]): void {
    if (types.includes('object')) {
        strict.properties = toStrictMap(pop(node, 'properties') || {});
        pop(node, 'additionalProperties');
        strict.additionalProperties = false;
        const required = pop(node, 'required');
        if (required !== undefined) {
            strict.required = required;
        }
    }
    if (types.includes('string') && typeof node.format === 'string' && SUPPORTED_STRING_FORMATS.has(node.format)) {
        strict.format = pop(node, 'format');
    }
    if (types.includes('array')) {
        const items = pop(node, 'items');
        if (Array.isArray(items)) {
            throw new Error(UNSUPPORTED_TUPLE);
        }
        if (items !== undefined) {
            strict.items = toStrictNode(items);
        }
        if (node.minItems === 0 || node.minItems === 1) {
            strict.minItems = pop(node, 'minItems');
        }
    }
}

/** Transforms each schema of named schema map. */
function toStrictMap(map: unknown): Record<string, unknown> {
    return Object.fromEntries(
        Object.entries(map as Record<string, unknown>).map(([name, node]) => [name, toStrictNode(node)]),
    );
}

/** Deletes key from object and returns its value. */
function pop(node: Record<string, unknown>, key: string): unknown {
    const value = node[key];
    delete node[key];
    return value;
}

/** Checks schema tree for `additionalProperties` set to anything but `false`. */
function hasOpenObject(node: unknown): boolean {
    if (Array.isArray(node)) {
        return node.some(hasOpenObject);
    }
    if (!isPlainObject(node)) {
        return false;
    }
    const open = node.additionalProperties;
    if (open !== undefined && open !== false) {
        return true;
    }
    return Object.entries(node).some(([key, value]) =>
        // named schema maps: check each schema, not the map itself (a property may be called additionalProperties)
        SCHEMA_MAP_KEYS.has(key) && isPlainObject(value)
            ? Object.values(value).some(hasOpenObject)
            : hasOpenObject(value),
    );
}

/** Returns `invalid_request` error result. */
function invalidRequest(message: string): { ok: false; error: LlmProviderError } {
    return { ok: false, error: { code: 'invalid_request', message, retryable: false } };
}
