// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { LlmJsonSchema } from './types.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Markdown code fence pattern enclosing JSON output. */
const MATCH_CODE_FENCE = /^```(?:json)?\s*\n?([\s\S]*?)\n?```$/;

/** Schema metadata keys stripped to prevent provider schema rejection. */
const META_KEYS: ReadonlySet<string> = new Set(['$schema', '$id']);

// -------------------------------------------------------------------
// --- 2. Exported Functions
// -------------------------------------------------------------------

/**
 * **Formats system instruction with JSON Schema**: guides models without native schema support.
 *
 * - injects schema as formatted JSON text
 * - instructs model to reply with single conforming JSON object
 */
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

/** Returns copy of schema with `$schema` and `$id` metadata properties removed. */
export function cleanSchema(schema: LlmJsonSchema): LlmJsonSchema {
    return mapSchema(schema, (node) => Object.fromEntries(Object.entries(node).filter(([key]) => !META_KEYS.has(key))));
}

/**
 * **Transforms schema for strict structured output**: required by OpenAI and Groq `strict: true`.
 *
 * - marks all object properties as required and sets `additionalProperties: false`
 * - converts optional properties to nullable types
 */
export function toStrictSchema(schema: LlmJsonSchema): LlmJsonSchema {
    return mapSchema(cleanSchema(schema), (node) => {
        const properties = node.properties as Record<string, LlmJsonSchema> | undefined;
        if (!isObjectNode(node) || properties === undefined) {
            return node;
        }
        const required = new Set(Array.isArray(node.required) ? (node.required as string[]) : []);
        const strictProperties = Object.fromEntries(
            Object.entries(properties).map(([key, value]) => [key, required.has(key) ? value : nullable(value)]),
        );
        return {
            ...node,
            properties: strictProperties,
            required: Object.keys(properties),
            additionalProperties: false,
        };
    });
}

/**
 * **Strips optional nulls**: recursively removes null values from fields schema does not require.
 *
 * - resolves matching branch for `anyOf` schemas
 * - preserves null values on required properties
 */
export function removeOptionalNulls(value: unknown, schema: LlmJsonSchema): unknown {
    if (Array.isArray(schema.anyOf)) {
        for (const branch of schema.anyOf as LlmJsonSchema[]) {
            const stripped = removeOptionalNulls(value, branch);
            if (findSchemaMismatch(stripped, branch) === undefined) {
                return stripped;
            }
        }
        return value;
    }
    if (Array.isArray(value)) {
        const items = schema.items as LlmJsonSchema | undefined;
        return items === undefined ? value : value.map((item) => removeOptionalNulls(item, items));
    }
    if (!isPlainObject(value)) {
        return value;
    }
    const properties = (schema.properties ?? {}) as Record<string, LlmJsonSchema>;
    const required = new Set(Array.isArray(schema.required) ? (schema.required as string[]) : []);
    const result: [string, unknown][] = [];
    for (const [key, entry] of Object.entries(value)) {
        const property = ownProperty(properties, key);
        if (entry === null && property !== undefined && !required.has(key)) {
            continue;
        }
        result.push([key, property === undefined ? entry : removeOptionalNulls(entry, property)]);
    }
    // defines own properties so keys like __proto__ never modify prototype chain
    return Object.fromEntries(result);
}

/**
 * **Finds first validation mismatch**: checks value against supported JSON Schema subset.
 *
 * - validates `type`, `enum`, `const`, `properties`, `required`, `additionalProperties`, `items`, and `anyOf`
 * - returns formatted path error string (e.g. `$.items[2].kind`) or `undefined` if valid
 */
export function findSchemaMismatch(value: unknown, schema: LlmJsonSchema, path = '$'): string | undefined {
    if (Array.isArray(schema.anyOf)) {
        const branches = schema.anyOf as LlmJsonSchema[];
        if (!branches.some((branch) => findSchemaMismatch(value, branch, path) === undefined)) {
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
            const mismatch = findSchemaMismatch(item, schema.items as LlmJsonSchema, `${path}[${index}]`);
            if (mismatch !== undefined) {
                return mismatch;
            }
        }
    }
    if (isPlainObject(value)) {
        return findObjectMismatch(value, schema, path);
    }
    return undefined;
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

/** Finds mismatch in object properties, required fields, or additionalProperties. */
function findObjectMismatch(value: Record<string, unknown>, schema: LlmJsonSchema, path: string): string | undefined {
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
        const mismatch = findSchemaMismatch(entry, property, `${path}.${key}`);
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

/** Applies transform to schema nodes post-order (children first). */
function mapSchema(schema: LlmJsonSchema, transform: (node: LlmJsonSchema) => LlmJsonSchema): LlmJsonSchema {
    const node: LlmJsonSchema = { ...schema };
    if (isPlainObject(node.properties)) {
        node.properties = Object.fromEntries(
            Object.entries(node.properties).map(([key, value]) => [key, mapSchema(value as LlmJsonSchema, transform)]),
        );
    }
    if (isPlainObject(node.items)) {
        node.items = mapSchema(node.items as LlmJsonSchema, transform);
    }
    if (Array.isArray(node.anyOf)) {
        node.anyOf = (node.anyOf as LlmJsonSchema[]).map((branch) => mapSchema(branch, transform));
    }
    return transform(node);
}

/** Returns schema variant that also accepts null. */
function nullable(schema: LlmJsonSchema): LlmJsonSchema {
    if (Array.isArray(schema.anyOf)) {
        return { ...schema, anyOf: [...schema.anyOf, { type: 'null' }] };
    }
    const result: LlmJsonSchema = { ...schema };
    if (typeof schema.type === 'string') {
        result.type = [schema.type, 'null'];
    } else if (Array.isArray(schema.type) && !schema.type.includes('null')) {
        result.type = [...schema.type, 'null'];
    }
    if (Array.isArray(schema.enum) && !schema.enum.includes(null)) {
        result.enum = [...schema.enum, null];
    }
    return result;
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
            return isPlainObject(value);
        default:
            return true;
    }
}

/** Checks whether value is non-array, non-null object. */
function isPlainObject(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Structural equality check for JSON values via serialization. */
function deepEqual(left: unknown, right: unknown): boolean {
    return JSON.stringify(left) === JSON.stringify(right);
}
