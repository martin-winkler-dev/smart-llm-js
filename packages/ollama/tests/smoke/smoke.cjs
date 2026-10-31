// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const pkg = require('@smart-llm/ollama');

assert.equal(typeof pkg.createOllamaProvider, 'function');
assert.equal(typeof pkg.fetchOllamaCapabilities, 'function');
assert.equal(pkg.OLLAMA_SUPPORT.json, 'grammar');

const provider = pkg.createOllamaProvider();
assert.equal(provider.name, 'ollama');
