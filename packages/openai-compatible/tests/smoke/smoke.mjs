// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import * as pkg from '@smart-llm/openai-compatible';

assert.equal(typeof pkg.createOpenAiCompatibleProvider, 'function');
assert.equal(typeof pkg.statusError, 'function');
assert.equal(typeof pkg.fetchJson, 'function');

const err = pkg.statusError(404, 'not found', null);
assert.equal(err.code, 'http');
