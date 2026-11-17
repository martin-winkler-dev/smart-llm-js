// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { LlmJsonSchema } from './types.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Markdown code fence pattern enclosing JSON output. */
const MATCH_CODE_FENCE = /^```(?:json)?\s*\n?([\s\S]*?)\n?```$/i;

/** Schema metadata keys stripped to prevent provider schema rejection. */
const META_KEYS: ReadonlySet<string> = new Set(['$schema', '$id']);

/** Keys holding one sub-schema. */
const SCHEMA_KEYS: ReadonlySet<string> = new Set(['items', 'additionalProperties', 'not']);

/** Keys holding an array of sub-schemas. */
const SCHEMA_LIST_KEYS: ReadonlySet<string> = new Set(['anyOf', 'allOf', 'oneOf', 'prefixItems']);

/** Keys holding a name-to-sub-schema map. */
const SCHEMA_MAP_KEYS: ReadonlySet<string> = new Set(['properties', '$defs', 'definitions']);

/** Keys whose subtrees `toStrictSchema` copies without strict rules: strict rules would change their meaning. */
const NON_STRICT_KEYS: ReadonlySet<string> = new Set(['allOf', 'oneOf', 'not', 'prefixItems', 'additionalProperties']);

/** Local definition ref like `#/$defs/<name>` or `#/definitions/<name>`. */
const MATCH_LOCAL_REF = /^#\/(\$defs|definitions)\/([^/]+)$/;

// -------------------------------------------------------------------
// --- 2. Exported Functions
// -------------------------------------------------------------------

/** Builds system instruction with optional JSON Schema for prompt-mode models. */
export function schemaInstruction(schema: LlmJsonSchema | undefined): string {
    if (schema === undefined) {
        return 'Reply with only one JSON object and no other text.';
    }
    return [
        'Reply with only one JSON object and no other text.',
        'It must match this JSON Schema; each "description" says what a field must contain:',
        JSON.stringify(schema, null, 2),
    ].join('\n');
}

/** Returns deep copy of schema with `$schema` and `$id` metadata removed from every sub-schema. */
export function cleanSchema(schema: LlmJsonSchema): LlmJsonSchema {
    return mapSchema(schema, (node) => Object.fromEntries(Object.entries(node).filter(([key]) => !META_KEYS.has(key))));
}

/**
 * **Transforms schema for strict structured output**: required by OpenAI and Groq `strict: true`.
 *
 * - applies to objects with `properties` reached via `properties`, `items`, `anyOf`, `$defs`, `definitions`
 * - marks all properties required, sets `additionalProperties: false`, makes optional properties nullable
 * - only cleans free-form `{ type: 'object' }` nodes and `allOf`/`oneOf`/`not`/`prefixItems`/tuple `items`/object `additionalProperties` subtrees
 */
export function toStrictSchema(schema: LlmJsonSchema): LlmJsonSchema {
    return mapSchema(
        cleanSchema(schema),
        (node) => {
            const properties = node.properties;
            if (!isObjectNode(node) || !isPlainObject(properties)) {
                return node;
            }
            const required = new Set(Array.isArray(node.required) ? (node.required as string[]) : []);
            const strictProperties = Object.fromEntries(
                Object.entries(properties).map(([key, value]) => [
                    key,
                    required.has(key) || !isPlainObject(value) ? value : nullable(value),
                ]),
            );
            return {
                ...node,
                properties: strictProperties,
                required: Object.keys(properties),
                additionalProperties: false,
            };
        },
        NON_STRICT_KEYS,
    );
}

/**
 * **Strips optional nulls**: recursively removes null values from fields schema does not require.
 *
 * - follows `properties`, `items`, matching `anyOf` branch and local `$ref`s (`#/$defs/<name>`, `#/definitions/<name>`)
 * - preserves null values on required properties
 */
export function removeOptionalNulls(value: unknown, schema: LlmJsonSchema): unknown {
    return stripNulls(value, schema, schema, new Set());
}

/**
 * **Finds first validation mismatch**: checks value against supported JSON Schema subset.
 *
 * - validates `type`, `enum`, `const`, `properties`, `required`, `additionalProperties: false`, `items`, `anyOf`, local `$ref`s
 * - `type: 'object'` rejects arrays and class instances (`Date`, `Map`...)
 * - ignores `allOf`, `oneOf`, `not`, `prefixItems`, schema-valued `additionalProperties`, formats, bounds and non-local refs
 * - returns formatted path error string (e.g. `$.items[2].kind`) or `undefined` if valid
 */
export function findSchemaMismatch(value: unknown, schema: LlmJsonSchema, path = '$'): string | undefined {
    return findMismatch(value, schema, path, schema, new Set());
}

/** Parses JSON text; strips surrounding markdown code fence if present. */
export function parseJsonText(text: string): { value: unknown } | undefined {
    const trimmed = text.trim();
    const fenced = MATCH_CODE_FENCE.exec(trimmed);
    try {
        return { value: JSON.parse(fenced === null ? trimmed : (fenced[1] as string)) };
    } catch {
        return undefined;
    }
}

// -------------------------------------------------------------------
// --- 3. Private Functions
// -------------------------------------------------------------------

/**
 * **Removes optional nulls against schema node**: implements `removeOptionalNulls`.
 *
 * - `root`: schema that local refs resolve against
 * - `seen`: refs followed without consuming value; stops cycles like `A -> $ref B -> $ref A`
 */
function stripNulls(value: unknown, schema: LlmJsonSchema, root: LlmJsonSchema, seen: ReadonlySet<string>): unknown {
    if (!isPlainObject(schema)) {
        return value;
    }
    const ref = schema.$ref;
    const target = typeof ref === 'string' && !seen.has(ref) ? resolveRef(ref, root) : undefined;
    const current = target === undefined ? value : stripNulls(value, target, root, new Set([...seen, ref as string]));
    if (Array.isArray(schema.anyOf)) {
        for (const branch of schema.anyOf as LlmJsonSchema[]) {
            const stripped = stripNulls(current, branch, root, seen);
            if (findMismatch(stripped, branch, '$', root, seen) === undefined) {
                return stripped;
            }
        }
        return current;
    }
    if (Array.isArray(current)) {
        const items = schema.items;
        return isPlainObject(items) ? current.map((item) => stripNulls(item, items, root, new Set())) : current;
    }
    if (!isJsonObject(current) || !isPlainObject(schema.properties)) {
        return current;
    }
    const properties = schema.properties as Record<string, LlmJsonSchema>;
    const required = new Set(Array.isArray(schema.required) ? (schema.required as string[]) : []);
    const result: [string, unknown][] = [];
    for (const [key, entry] of Object.entries(current)) {
        const property = ownProperty(properties, key);
        if (entry === null && property !== undefined && !required.has(key)) {
            continue;
        }
        result.push([key, property === undefined ? entry : stripNulls(entry, property, root, new Set())]);
    }
    // defines own properties so keys like __proto__ never modify prototype chain
    return Object.fromEntries(result);
}

/** Implements `findSchemaMismatch`; `root` and `seen` as in `stripNulls`. */
function findMismatch(
    value: unknown,
    schema: LlmJsonSchema,
    path: string,
    root: LlmJsonSchema,
    seen: ReadonlySet<string>,
): string | undefined {
    if (!isPlainObject(schema)) {
        return undefined;
    }
    const ref = schema.$ref;
    const target = typeof ref === 'string' && !seen.has(ref) ? resolveRef(ref, root) : undefined;
    if (target !== undefined) {
        const mismatch = findMismatch(value, target, path, root, new Set([...seen, ref as string]));
        if (mismatch !== undefined) {
            return mismatch;
        }
    }
    if (Array.isArray(schema.anyOf)) {
        const branches = schema.anyOf as LlmJsonSchema[];
        if (!branches.some((branch) => findMismatch(value, branch, path, root, seen) === undefined)) {
            return `${path} matches none of anyOf`;
        }
    }
    if (Array.isArray(schema.enum) && !schema.enum.some((entry) => deepEqual(entry, value))) {
        return `${path} is not one of ${JSON.stringify(schema.enum)}`;
    }
    if ('const' in schema && !deepEqual(schema.const, value)) {
        return `${path} is not ${JSON.stringify(schema.const)}`;
    }
    const types = typeof schema.type === 'string' ? [schema.type] : Array.isArray(schema.type) ? schema.type : [];
    if (types.length > 0 && !types.some((type) => hasType(value, type as string))) {
        return `${path} is not of type ${types.join(' | ')}`;
    }
    if (Array.isArray(value) && isPlainObject(schema.items)) {
        for (const [index, item] of value.entries()) {
            const mismatch = findMismatch(item, schema.items as LlmJsonSchema, `${path}[${index}]`, root, new Set());
            if (mismatch !== undefined) {
                return mismatch;
            }
        }
    }
    if (isJsonObject(value)) {
        return findObjectMismatch(value, schema, path, root);
    }
    return undefined;
}

/** Resolves local `#/$defs/<name>` or `#/definitions/<name>` ref against root; `undefined` otherwise. */
function resolveRef(ref: string, root: LlmJsonSchema): LlmJsonSchema | undefined {
    const match = MATCH_LOCAL_REF.exec(ref);
    if (match === null) {
        return undefined;
    }
    const defs = root[match[1] as string];
    const name = (match[2] as string).replaceAll('~1', '/').replaceAll('~0', '~');
    const target = isPlainObject(defs) && Object.hasOwn(defs, name) ? defs[name] : undefined;
    return isPlainObject(target) ? target : undefined;
}

/** Finds mismatch in object properties, required fields, or additionalProperties. */
function findObjectMismatch(
    value: Record<string, unknown>,
    schema: LlmJsonSchema,
    path: string,
    root: LlmJsonSchema,
): string | undefined {
    const properties = (isPlainObject(schema.properties) ? schema.properties : {}) as Record<string, LlmJsonSchema>;
    if (Array.isArray(schema.required)) {
        const missing = (schema.required as string[]).find((key) => !Object.hasOwn(value, key));
        if (missing !== undefined) {
            return `${path}.${missing} is missing`;
        }
    }
    for (const [key, entry] of Object.entries(value)) {
        const property = ownProperty(properties, key);
        if (property === undefined) {
            if (schema.additionalProperties === false) {
                return `${path}.${key} is not allowed`;
            }
            continue;
        }
        const mismatch = findMismatch(entry, property, `${path}.${key}`, root, new Set());
        if (mismatch !== undefined) {
            return mismatch;
        }
    }
    return undefined;
}

/** Gets declared property schema; ignores inherited Object.prototype properties. */
function ownProperty(properties: Record<string, LlmJsonSchema>, key: string): LlmJsonSchema | undefined {
    return Object.hasOwn(properties, key) ? properties[key] : undefined;
}

/**
 * **Applies transform to schema nodes post-order (children first)**: works on deep copy.
 *
 * - recurses into `properties`, `$defs`, `definitions`, `items`, object `additionalProperties`, `not`, `anyOf`, `allOf`, `oneOf`, `prefixItems`
 * - tuple `items` array: handled like `prefixItems` (also for `copyOnlyKeys`)
 * - `copyOnlyKeys`: subtrees under these keys copied without transform
 * - copies other values (`required`, `enum`, `const`...) without sharing arrays or objects with input
 */
function mapSchema(
    schema: LlmJsonSchema,
    transform: (node: LlmJsonSchema) => LlmJsonSchema,
    copyOnlyKeys: ReadonlySet<string> = new Set(),
): LlmJsonSchema {
    function visit(value: unknown): unknown {
        return isPlainObject(value) ? mapSchema(value, transform, copyOnlyKeys) : copyJson(value);
    }
    const node: LlmJsonSchema = Object.fromEntries(
        Object.entries(schema).map(([key, value]) => {
            // tuple `items` (draft-07) handled like `prefixItems`
            const kind = key === 'items' && Array.isArray(value) ? 'prefixItems' : key;
            if (copyOnlyKeys.has(kind)) {
                return [key, copyJson(value)];
            }
            if (SCHEMA_KEYS.has(kind)) {
                return [key, visit(value)];
            }
            if (SCHEMA_LIST_KEYS.has(kind) && Array.isArray(value)) {
                return [key, value.map(visit)];
            }
            if (SCHEMA_MAP_KEYS.has(key) && isPlainObject(value)) {
                return [key, Object.fromEntries(Object.entries(value).map(([name, entry]) => [name, visit(entry)]))];
            }
            return [key, copyJson(value)];
        }),
    );
    return transform(node);
}

/** Deep-copies arrays and JSON objects; returns other values as is. */
function copyJson(value: unknown): unknown {
    if (Array.isArray(value)) {
        return value.map(copyJson);
    }
    if (isJsonObject(value)) {
        return Object.fromEntries(Object.entries(value).map(([key, entry]) => [key, copyJson(entry)]));
    }
    return value;
}

/**
 * **Returns schema variant that also accepts null.**
 *
 * - null-only schema (`{ type: 'null' }`, `{ type: ['null'] }`, `{ const: null }`): unchanged
 * - `anyOf`: appends `{ type: 'null' }` branch unless null-only branch present
 * - no `type`/`enum` (`{}`, `$ref`, untyped `const`) or `const` next to `enum`: wraps as `anyOf: [node, { type: 'null' }]`
 * - otherwise adds `'null'` to `type` and `null` to `enum`; typed `const` becomes `enum: [const, null]`
 */
function nullable(schema: LlmJsonSchema): LlmJsonSchema {
    const { type, enum: values } = schema;
    if (isNullSchema(schema)) {
        return schema;
    }
    if (Array.isArray(schema.anyOf)) {
        return schema.anyOf.some(isNullSchema) ? schema : { ...schema, anyOf: [...schema.anyOf, { type: 'null' }] };
    }
    const hasConst = 'const' in schema;
    if ((type === undefined && values === undefined) || (hasConst && values !== undefined)) {
        return { anyOf: [schema, { type: 'null' }] };
    }
    const result: LlmJsonSchema = { ...schema };
    if (typeof type === 'string') {
        result.type = [type, 'null'];
    } else if (Array.isArray(type) && !type.includes('null')) {
        result.type = [...type, 'null'];
    }
    if (hasConst) {
        delete result.const;
        result.enum = [schema.const, null];
    } else if (Array.isArray(values) && !values.includes(null)) {
        result.enum = [...values, null];
    }
    return result;
}

/** Checks whether schema accepts only null: `{ type: 'null' }`, `{ type: ['null'] }` or `{ const: null }`. */
function isNullSchema(schema: unknown): boolean {
    if (!isPlainObject(schema)) {
        return false;
    }
    const { type } = schema;
    return type === 'null' || (Array.isArray(type) && type.length === 1 && type[0] === 'null') || schema.const === null;
}

/** Checks whether schema node represents an object. */
function isObjectNode(node: LlmJsonSchema): boolean {
    return node.type === 'object' || (Array.isArray(node.type) && node.type.includes('object'));
}

/** Checks whether value matches JSON Schema type. */
function hasType(value: unknown, type: string): boolean {
    switch (type) {
        case 'null':
            return value === null;
        case 'boolean':
            return typeof value === 'boolean';
        case 'string':
            return typeof value === 'string';
        case 'number':
            return typeof value === 'number' && Number.isFinite(value);
        case 'integer':
            return Number.isInteger(value);
        case 'array':
            return Array.isArray(value);
        case 'object':
            return isJsonObject(value);
        default:
            return true;
    }
}

/** Checks whether value is non-array, non-null object; loose check for schema nodes. */
function isPlainObject(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Checks whether value is JSON object (prototype `Object.prototype` or `null`); strict check for answer values. */
function isJsonObject(value: unknown): value is Record<string, unknown> {
    if (!isPlainObject(value)) {
        return false;
    }
    const proto = Object.getPrototypeOf(value);
    return proto === Object.prototype || proto === null;
}

/** Structural equality of JSON values: arrays by index (holes as `undefined`), objects by own keys, primitives with `===`. */
function deepEqual(left: unknown, right: unknown): boolean {
    if (Array.isArray(left) || Array.isArray(right)) {
        if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) {
            return false;
        }
        for (let index = 0; index < left.length; index++) {
            if (!deepEqual(left[index], right[index])) {
                return false;
            }
        }
        return true;
    }
    if (isJsonObject(left) && isJsonObject(right)) {
        const keys = Object.keys(left);
        return (
            keys.length === Object.keys(right).length &&
            keys.every((key) => Object.hasOwn(right, key) && deepEqual(left[key], right[key]))
        );
    }
    return left === right;
}
