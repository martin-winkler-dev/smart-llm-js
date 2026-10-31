// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const pkg = require('@smart-llm/core');

assert.equal(typeof pkg.createLlm, 'function');
assert.equal(typeof pkg.imageFromFile, 'function');
assert.equal(typeof pkg.imagePart, 'function');
assert.equal(typeof pkg.assistantMessage, 'function');
assert.equal(typeof pkg.toolMessage, 'function');
assert.equal(typeof pkg.cleanSchema, 'function');
assert.equal(typeof pkg.findSchemaMismatch, 'function');
assert.equal(typeof pkg.parseJsonText, 'function');
assert.equal(typeof pkg.removeOptionalNulls, 'function');
assert.equal(typeof pkg.schemaInstruction, 'function');
assert.equal(typeof pkg.toStrictSchema, 'function');

assert.deepEqual(pkg.cleanSchema({ type: 'string' }), { type: 'string' });
