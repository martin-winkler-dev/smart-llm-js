// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const pkg = require('@smart-llm/gemini');

assert.equal(typeof pkg.createGeminiProvider, 'function');
assert.equal(pkg.GEMINI_SUPPORT.webSearch, true);

const provider = pkg.createGeminiProvider({ apiKey: 'smoke-test-key' });
assert.equal(provider.name, 'gemini');
