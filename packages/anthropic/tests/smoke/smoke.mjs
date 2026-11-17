// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import * as pkg from '@smart-llm/anthropic';

assert.equal(typeof pkg.createAnthropicProvider, 'function');
assert.equal(typeof pkg.ANTHROPIC_API_KEY_ENV, 'string');
assert.equal(typeof pkg.DEFAULT_MAX_TOKENS, 'number');
assert.equal(pkg.ANTHROPIC_SUPPORT.batch, true);
assert.equal(pkg.createAnthropicProvider({ apiKey: 'key' }).name, 'anthropic');
