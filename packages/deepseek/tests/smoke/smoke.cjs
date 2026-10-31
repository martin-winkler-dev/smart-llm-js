// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const pkg = require('@smart-llm/deepseek');

assert.equal(typeof pkg.createDeepseekProvider, 'function');
assert.equal(pkg.DEEPSEEK_API_KEY_ENV, 'DEEPSEEK_API_KEY');
assert.equal(pkg.DEEPSEEK_SUPPORT.reasoning, true);

const provider = pkg.createDeepseekProvider({ apiKey: 'smoke-test-key' });
assert.equal(provider.name, 'deepseek');
