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
});

describe('parseJsonText', () => {
    it('parses JSON with or without one code fence', () => {
        expect(parseJsonText(' {"a":1} ')).toEqual({ value: { a: 1 } });
        expect(parseJsonText('```json\n{"a":1}\n```')).toEqual({ value: { a: 1 } });
        expect(parseJsonText('{"a":')).toBeUndefined();
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
});
