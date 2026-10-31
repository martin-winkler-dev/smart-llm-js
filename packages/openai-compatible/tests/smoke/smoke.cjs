// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const pkg = require('@smart-llm/openai-compatible');

assert.equal(typeof pkg.createOpenAiCompatibleProvider, 'function');
assert.equal(typeof pkg.statusError, 'function');
assert.equal(typeof pkg.fetchJson, 'function');

const err = pkg.statusError(404, 'not found', null);
assert.equal(err.code, 'http');
