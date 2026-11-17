// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'bun:test';
import {
    cleanSchema,
    findSchemaMismatch,
    parseJsonText,
    removeOptionalNulls,
    schemaInstruction,
    toStrictSchema,
} from '../src/schema.js';

const schema = {
    $schema: 'https://json-schema.org/draft/2020-12/schema',
    type: 'object',
    properties: {
        kind: { type: 'string', enum: ['noun', 'verb'], description: 'Word class' },
        score: { type: 'number' },
        note: { type: 'string' },
        tags: { type: 'array', items: { type: 'object', properties: { label: { type: 'string' } } } },
    },
    required: ['kind', 'score'],
};

describe('toStrictSchema', () => {
    it('requires every field, closes objects and makes optional fields nullable', () => {
        expect(toStrictSchema(schema)).toEqual({
            type: 'object',
            properties: {
                kind: { type: 'string', enum: ['noun', 'verb'], description: 'Word class' },
                score: { type: 'number' },
                note: { type: ['string', 'null'] },
                tags: {
                    type: ['array', 'null'],
                    items: {
                        type: 'object',
                        properties: { label: { type: ['string', 'null'] } },
                        required: ['label'],
                        additionalProperties: false,
                    },
                },
            },
            required: ['kind', 'score', 'note', 'tags'],
            additionalProperties: false,
        });
    });

    it('adds null to enums and anyOf branches of optional fields', () => {
        const strict = toStrictSchema({
            type: 'object',
            properties: { a: { enum: ['x'] }, b: { anyOf: [{ type: 'string' }, { type: 'number' }] } },
        });
        expect(strict.properties).toEqual({
            a: { enum: ['x', null] },
            b: { anyOf: [{ type: 'string' }, { type: 'number' }, { type: 'null' }] },
        });
    });

    it('makes optional null, const, typeless and $ref fields nullable without duplicating null', () => {
        const strict = toStrictSchema({
            type: 'object',
            properties: {
                n: { type: 'null' },
                c: { const: 'dog' },
                t: { type: 'string', const: 'dog' },
                e: { const: 'x', enum: ['x', 'y'] },
                any: {},
                ref: { $ref: '#/$defs/D' },
                u: { anyOf: [{ type: 'string' }, { type: 'null' }] },
            },
        });
        expect(strict.properties).toEqual({
            n: { type: 'null' },
            c: { anyOf: [{ const: 'dog' }, { type: 'null' }] },
            t: { type: ['string', 'null'], enum: ['dog', null] },
            e: { anyOf: [{ const: 'x', enum: ['x', 'y'] }, { type: 'null' }] },
            any: { anyOf: [{}, { type: 'null' }] },
            ref: { anyOf: [{ $ref: '#/$defs/D' }, { type: 'null' }] },
            u: { anyOf: [{ type: 'string' }, { type: 'null' }] },
        });
        const props = strict.properties as Record<string, Record<string, unknown>>;
        for (const key of ['c', 't', 'e', 'any', 'ref']) {
            expect(findSchemaMismatch(null, props[key] as Record<string, unknown>)).toBeUndefined();
        }
        expect(findSchemaMismatch('dog', props.t as Record<string, unknown>)).toBeUndefined();
        expect(findSchemaMismatch('cat', props.t as Record<string, unknown>)).toBe('$ is not one of ["dog",null]');
    });

    it('applies strict rules inside $defs and definitions', () => {
        const inner = { type: 'object', properties: { a: { type: 'string' } } };
        const closed = {
            type: 'object',
            properties: { a: { type: ['string', 'null'] } },
            required: ['a'],
            additionalProperties: false,
        };
        expect(toStrictSchema({ $defs: { D: inner }, definitions: { E: inner } })).toEqual({
            $defs: { D: closed },
            definitions: { E: closed },
        });
    });

    it('cleans tuple items like prefixItems without strict rules and keeps boolean items', () => {
        const inner = { $id: 'x', type: 'object', properties: { a: { $schema: 'y', type: 'string' } } };
        expect(toStrictSchema({ type: 'array', items: [inner, true] })).toEqual({
            type: 'array',
            items: [{ type: 'object', properties: { a: { type: 'string' } } }, true],
        });
        expect(toStrictSchema({ type: 'array', items: false })).toEqual({ type: 'array', items: false });
        expect(toStrictSchema({ type: 'array', items: inner })).toEqual({
            type: 'array',
            items: {
                type: 'object',
                properties: { a: { type: ['string', 'null'] } },
                required: ['a'],
                additionalProperties: false,
            },
        });
    });

    it('only cleans allOf, oneOf, not, prefixItems and additionalProperties subtrees', () => {
        const inner = {
            $id: 'x',
            type: 'object',
            properties: { a: { $schema: 'y', type: 'string' }, b: { type: 'object', properties: {} } },
        };
        const cleaned = {
            type: 'object',
            properties: { a: { type: 'string' }, b: { type: 'object', properties: {} } },
        };
        expect(
            toStrictSchema({
                allOf: [inner],
                oneOf: [inner],
                not: inner,
                prefixItems: [inner],
                additionalProperties: inner,
            }),
        ).toEqual({
            allOf: [cleaned],
            oneOf: [cleaned],
            not: cleaned,
            prefixItems: [cleaned],
            additionalProperties: cleaned,
        });
    });

    it('does not duplicate null for optional fields with null-only anyOf branches', () => {
        const strict = toStrictSchema({
            type: 'object',
            properties: {
                a: { anyOf: [{ type: 'string' }, { type: ['null'] }] },
                b: { anyOf: [{ type: 'string' }, { const: null }] },
                c: { type: ['null'] },
                d: { const: null },
            },
        });
        expect(strict.properties).toEqual({
            a: { anyOf: [{ type: 'string' }, { type: ['null'] }] },
            b: { anyOf: [{ type: 'string' }, { const: null }] },
            c: { type: ['null'] },
            d: { const: null },
        });
    });

    it('leaves free-form objects unchanged and ignores malformed properties', () => {
        expect(toStrictSchema({ type: 'object' })).toEqual({ type: 'object' });
        expect(toStrictSchema({ type: 'object', properties: null })).toEqual({ type: 'object', properties: null });
        expect(toStrictSchema({ type: 'object', properties: { a: null, b: true } })).toEqual({
            type: 'object',
            properties: { a: null, b: true },
            required: ['a', 'b'],
            additionalProperties: false,
        });
    });
});

describe('removeOptionalNulls', () => {
    it('drops nulls of optional fields, also in arrays, and keeps required ones', () => {
        expect(removeOptionalNulls({ kind: 'noun', score: 1, note: null, tags: [{ label: null }] }, schema)).toEqual({
            kind: 'noun',
            score: 1,
            tags: [{}],
        });
        expect(removeOptionalNulls({ kind: null, score: 1 }, schema)).toEqual({ kind: null, score: 1 });
    });

    it('strips anyOf values against the branch they then match', () => {
        const pet = {
            anyOf: [
                {
                    type: 'object',
                    properties: { kind: { const: 'dog' }, name: { type: 'string' }, nick: { type: 'string' } },
                    required: ['kind', 'name'],
                },
                {
                    type: 'object',
                    properties: { kind: { const: 'cat' }, lives: { type: 'integer' } },
                    required: ['kind'],
                },
            ],
        };
        const union = { type: 'object', properties: { pet }, required: ['pet'] };
        const answer = { pet: { kind: 'dog', name: 'Rex', nick: null } };
        const stripped = removeOptionalNulls(answer, union);
        expect(stripped).toEqual({ pet: { kind: 'dog', name: 'Rex' } });
        expect(findSchemaMismatch(stripped, union)).toBeUndefined();
        expect(removeOptionalNulls({ pet: { kind: 'cat', lives: null } }, union)).toEqual({ pet: { kind: 'cat' } });
        expect(removeOptionalNulls({ pet: { kind: 'bird' } }, union)).toEqual({ pet: { kind: 'bird' } });
    });

    it('keeps a __proto__ key as an own field instead of setting the prototype', () => {
        const stripped = removeOptionalNulls(JSON.parse('{"a":"x","__proto__":{"isAdmin":true}}'), schema) as {
            isAdmin?: unknown;
        };
        expect(stripped.isAdmin).toBeUndefined();
        expect(Object.getPrototypeOf(stripped)).toBe(Object.prototype);
        expect(Object.hasOwn(stripped, '__proto__')).toBe(true);
    });

    it('keeps class instances instead of turning them into plain objects', () => {
        const date = new Date(0);
        const map = new Map([['a', 1]]);
        expect(removeOptionalNulls(date, { type: 'object' })).toBe(date);
        expect(removeOptionalNulls(map, { type: 'object' })).toBe(map);
    });

    it('ignores malformed items and property schemas instead of throwing', () => {
        expect(removeOptionalNulls([1], { type: 'array', items: null })).toEqual([1]);
        expect(removeOptionalNulls({ a: { b: null } }, { type: 'object', properties: { a: null } })).toEqual({
            a: { b: null },
        });
        expect(removeOptionalNulls({ a: 1 }, { type: 'object', properties: null })).toEqual({ a: 1 });
        expect(removeOptionalNulls(1, { anyOf: [null, { type: 'number' }] })).toBe(1);
    });

    it('keeps array values unchanged under tuple or boolean items', () => {
        const tuple = { type: 'array', items: [{ type: 'object', properties: { a: { type: 'string' } } }] };
        expect(removeOptionalNulls([{ a: null }], tuple)).toEqual([{ a: null }]);
        expect(removeOptionalNulls([{ a: null }], { type: 'array', items: true })).toEqual([{ a: null }]);
        expect(removeOptionalNulls([{ a: null }], { $ref: '#/$defs/T', $defs: { T: tuple } })).toEqual([{ a: null }]);
    });
});

describe('strict round trip', () => {
    const D = { type: 'object', properties: { a: { type: 'string' }, b: { type: 'string' } }, required: ['a'] };

    /** Validates answer against strict schema, strips nulls via original schema and validates again. */
    function roundTrip(original: Record<string, unknown>, answer: unknown): unknown {
        expect(findSchemaMismatch(answer, toStrictSchema(original))).toBeUndefined();
        const stripped = removeOptionalNulls(answer, original);
        expect(findSchemaMismatch(stripped, original)).toBeUndefined();
        expect(JSON.stringify(stripped)).not.toContain('null');
        return stripped;
    }

    it('removes nulls forced inside $ref to $defs', () => {
        const original = { type: 'object', properties: { d: { $ref: '#/$defs/D' } }, required: ['d'], $defs: { D } };
        expect(roundTrip(original, { d: { a: 'x', b: null } })).toEqual({ d: { a: 'x' } });
    });

    it('removes nulls forced inside $ref to definitions', () => {
        const original = {
            type: 'object',
            properties: { d: { type: 'array', items: { $ref: '#/definitions/D' } } },
            required: ['d'],
            definitions: { D },
        };
        expect(
            roundTrip(original, {
                d: [
                    { a: 'x', b: null },
                    { a: 'y', b: 'z' },
                ],
            }),
        ).toEqual({
            d: [{ a: 'x' }, { a: 'y', b: 'z' }],
        });
    });

    it('removes nulls in recursive refs', () => {
        const original = {
            $ref: '#/$defs/Node',
            $defs: {
                Node: {
                    type: 'object',
                    properties: {
                        v: { type: 'string' },
                        next: { $ref: '#/$defs/Node' },
                        kids: { type: 'array', items: { $ref: '#/$defs/Node' } },
                    },
                    required: ['v'],
                },
            },
        };
        const answer = {
            v: 'a',
            next: { v: 'b', next: null, kids: [{ v: 'c', next: null, kids: null }] },
            kids: null,
        };
        expect(roundTrip(original, answer)).toEqual({ v: 'a', next: { v: 'b', kids: [{ v: 'c' }] } });
    });

    it('removes nulls in optional $ref property wrapped in anyOf', () => {
        const original = { type: 'object', properties: { d: { $ref: '#/$defs/D' } }, $defs: { D } };
        expect(toStrictSchema(original).properties).toEqual({
            d: { anyOf: [{ $ref: '#/$defs/D' }, { type: 'null' }] },
        });
        expect(roundTrip(original, { d: null })).toEqual({});
        expect(roundTrip(original, { d: { a: 'x', b: null } })).toEqual({ d: { a: 'x' } });
    });

    it('removes nulls of optional untyped and typed const', () => {
        const original = { type: 'object', properties: { c: { const: 'dog' }, t: { type: 'string', const: 'dog' } } };
        expect(roundTrip(original, { c: null, t: null })).toEqual({});
        expect(roundTrip(original, { c: 'dog', t: 'dog' })).toEqual({ c: 'dog', t: 'dog' });
    });

    it('terminates on ref cycles that do not consume the value', () => {
        const original = { $ref: '#/$defs/A', $defs: { A: { $ref: '#/$defs/B' }, B: { $ref: '#/$defs/A' } } };
        expect(removeOptionalNulls({ a: null }, original)).toEqual({ a: null });
        expect(findSchemaMismatch({ a: null }, original)).toBeUndefined();
    });
});

describe('findSchemaMismatch', () => {
    it('accepts a matching value', () => {
        expect(findSchemaMismatch({ kind: 'verb', score: 2, tags: [{ label: 'x' }] }, schema)).toBeUndefined();
    });

    it('names the first mismatch with its path', () => {
        expect(findSchemaMismatch({ score: 1 }, schema)).toBe('$.kind is missing');
        expect(findSchemaMismatch({ kind: 'adj', score: 1 }, schema)).toBe('$.kind is not one of ["noun","verb"]');
        expect(findSchemaMismatch({ kind: 'noun', score: '1' }, schema)).toBe('$.score is not of type number');
        expect(findSchemaMismatch({ kind: 'noun', score: 1, tags: [{ label: 3 }] }, schema)).toBe(
            '$.tags[0].label is not of type string',
        );
        expect(findSchemaMismatch({ a: 1 }, { type: 'object', additionalProperties: false })).toBe(
            '$.a is not allowed',
        );
        expect(findSchemaMismatch(1.5, { type: 'integer' })).toBe('$ is not of type integer');
        expect(findSchemaMismatch(true, { anyOf: [{ type: 'string' }, { type: 'number' }] })).toBe(
            '$ matches none of anyOf',
        );
        expect(findSchemaMismatch(null, { type: ['string', 'null'] })).toBeUndefined();
    });

    it('treats keys named like Object.prototype members as unknown fields', () => {
        const closed = { type: 'object', properties: { a: { type: 'string' } }, additionalProperties: false };
        for (const key of ['constructor', 'toString', '__proto__']) {
            const value = JSON.parse(`{"a":"x","${key}":{"isAdmin":true}}`);
            expect(findSchemaMismatch(value, closed)).toBe(`$.${key} is not allowed`);
        }
    });

    it('rejects class instances as objects and accepts null-prototype objects', () => {
        for (const value of [new Date(0), new Map(), new Set(), /x/, new Uint8Array(1), new URL('https://x.test')]) {
            expect(findSchemaMismatch(value, { type: 'object' })).toBe('$ is not of type object');
        }
        expect(findSchemaMismatch(Object.create(null), { type: 'object' })).toBeUndefined();
    });

    it('compares object const and enum values structurally, independent of key order', () => {
        expect(findSchemaMismatch({ b: 2, a: 1 }, { const: { a: 1, b: 2 } })).toBeUndefined();
        expect(findSchemaMismatch({ b: [1, { d: 2, c: 1 }], a: 1 }, { enum: [{ a: 1, b: [1, { c: 1, d: 2 }] }] })).toBe(
            undefined,
        );
        expect(findSchemaMismatch({}, { const: { a: undefined } })).toBe('$ is not {}');
        expect(findSchemaMismatch(null, { const: Number.NaN })).toBe('$ is not null');
        expect(findSchemaMismatch([1, 2], { const: [1, 2, 3] })).toBe('$ is not [1,2,3]');
        expect(findSchemaMismatch({ 0: 1 }, { const: [1] })).toBe('$ is not [1]');
        expect(findSchemaMismatch({ a: 1, b: 2 }, { const: { a: 1, c: 2 } })).toBe('$ is not {"a":1,"c":2}');
    });

    it('ignores malformed items, property and anyOf schemas instead of throwing', () => {
        expect(findSchemaMismatch([1], { type: 'array', items: null })).toBeUndefined();
        expect(findSchemaMismatch({ a: 1 }, { type: 'object', properties: { a: null } })).toBeUndefined();
        expect(findSchemaMismatch({ a: 1 }, { type: 'object', properties: null })).toBeUndefined();
        expect(findSchemaMismatch(1, { anyOf: [null] })).toBeUndefined();
    });

    it('skips tuple and boolean items without throwing', () => {
        const tuple = { type: 'array', items: [{ type: 'string' }, { type: 'number' }] };
        expect(findSchemaMismatch([1, 'x'], tuple)).toBeUndefined();
        expect(findSchemaMismatch([1], { type: 'array', items: false })).toBeUndefined();
        expect(findSchemaMismatch([1], { $ref: '#/$defs/T', $defs: { T: tuple } })).toBeUndefined();
        expect(findSchemaMismatch('x', { $ref: '#/$defs/T', $defs: { T: tuple } })).toBe('$ is not of type array');
    });

    it('validates values under local $ref and skips unresolvable or non-local refs', () => {
        const defs = { D: { type: 'object', properties: { a: { type: 'string' } }, required: ['a'] } };
        const local = { type: 'object', properties: { d: { $ref: '#/$defs/D' } }, $defs: defs };
        expect(findSchemaMismatch({ d: { a: 1 } }, local)).toBe('$.d.a is not of type string');
        expect(findSchemaMismatch({ d: {} }, local)).toBe('$.d.a is missing');
        expect(findSchemaMismatch({ d: { a: 'x' } }, local)).toBeUndefined();
        for (const $ref of ['#/$defs/Missing', 'https://x.test/s.json', '#/properties/d']) {
            expect(findSchemaMismatch({ d: { a: 1 } }, { properties: { d: { $ref } }, $defs: defs })).toBeUndefined();
        }
    });

    it('compares sparse array const values by index', () => {
        // biome-ignore lint/suspicious/noSparseArray: sparse array is the case under test
        expect(findSchemaMismatch([1, null], { const: [1, ,] })).toBe('$ is not [1,null]');
    });
});

describe('parseJsonText', () => {
    it('parses JSON with or without one code fence', () => {
        expect(parseJsonText(' {"a":1} ')).toEqual({ value: { a: 1 } });
        expect(parseJsonText('```json\n{"a":1}\n```')).toEqual({ value: { a: 1 } });
        expect(parseJsonText('{"a":')).toBeUndefined();
    });

    it('accepts the json fence tag in any case', () => {
        for (const tag of ['JSON', 'Json', 'jSoN']) {
            expect(parseJsonText(`\`\`\`${tag}\n{"a":1}\n\`\`\``)).toEqual({ value: { a: 1 } });
        }
    });
});

describe('schemaInstruction and cleanSchema', () => {
    it('puts the schema with its descriptions into the text and names JSON', () => {
        const text = schemaInstruction(cleanSchema(schema));
        expect(text).toContain('JSON object');
        expect(text).toContain('"description": "Word class"');
        expect(text).not.toContain('$schema');
        expect(schemaInstruction(undefined)).toBe('Reply with only one JSON object and no other text.');
    });

    it('removes $id and $schema from every sub-schema', () => {
        const meta = { $id: 'x', $schema: 'y' };
        const cleaned = cleanSchema({
            ...meta,
            $defs: { D: { ...meta, type: 'string' } },
            definitions: { E: { ...meta, type: 'string' } },
            additionalProperties: { ...meta, type: 'string' },
            allOf: [{ ...meta, type: 'string' }],
            oneOf: [{ ...meta, type: 'string' }],
            not: { ...meta, type: 'string' },
            prefixItems: [{ ...meta, type: 'string' }],
            items: [{ ...meta, type: 'string' }, true],
            properties: { $id: { ...meta, type: 'string' } },
            const: { $id: 'kept' },
        });
        expect(cleaned).toEqual({
            $defs: { D: { type: 'string' } },
            definitions: { E: { type: 'string' } },
            additionalProperties: { type: 'string' },
            allOf: [{ type: 'string' }],
            oneOf: [{ type: 'string' }],
            not: { type: 'string' },
            prefixItems: [{ type: 'string' }],
            items: [{ type: 'string' }, true],
            properties: { $id: { type: 'string' } },
            const: { $id: 'kept' },
        });
    });

    it('returns a deep copy that shares no arrays or objects with the input', () => {
        const input = {
            type: ['object', 'null'],
            properties: { a: { enum: ['x', { y: 1 }] } },
            required: ['a'],
            $defs: { D: { type: 'string' } },
            allOf: [{ type: 'object' }],
            additionalProperties: false,
        };
        const snapshot = structuredClone(input);
        for (const copy of [cleanSchema(input), toStrictSchema(input)]) {
            const record = copy as Record<string, any>;
            record.type.push('array');
            record.required.push('zzz');
            record.properties.a.enum.push('z');
            record.properties.a.enum[1].y = 2;
            record.$defs.D.type = 'number';
            record.allOf[0].type = 'number';
        }
        expect(input).toEqual(snapshot);
    });
});
