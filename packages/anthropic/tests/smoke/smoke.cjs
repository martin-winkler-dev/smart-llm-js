// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const pkg = require('@smart-llm/anthropic');

assert.equal(typeof pkg.createAnthropicProvider, 'function');
assert.equal(typeof pkg.ANTHROPIC_API_KEY_ENV, 'string');
assert.equal(typeof pkg.DEFAULT_MAX_TOKENS, 'number');
assert.equal(pkg.ANTHROPIC_SUPPORT.batch, true);
assert.equal(pkg.createAnthropicProvider({ apiKey: 'key' }).name, 'anthropic');
