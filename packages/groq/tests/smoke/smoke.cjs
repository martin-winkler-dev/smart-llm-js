// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const pkg = require('@smart-llm/groq');

assert.equal(typeof pkg.createGroqProvider, 'function');
assert.equal(pkg.GROQ_API_KEY_ENV, 'GROQ_API_KEY');
assert.equal(pkg.GROQ_SUPPORT.json, 'strict');

const provider = pkg.createGroqProvider({ apiKey: 'smoke-test-key' });
assert.equal(provider.name, 'groq');
