// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import * as pkg from '@smart-llm/gemini';

assert.equal(typeof pkg.createGeminiProvider, 'function');
assert.equal(pkg.GEMINI_SUPPORT.webSearch, true);

const provider = pkg.createGeminiProvider({ apiKey: 'smoke-test-key' });
assert.equal(provider.name, 'gemini');
