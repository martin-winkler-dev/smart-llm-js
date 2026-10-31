// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

// Live: embeddings with bge-m3 on the local Ollama server (`ollama pull bge-m3`).
// Run with `bun run test:live:ollama` from the repository root; never part of `bun test`.

import { describe, expect, it } from 'bun:test';
import { createLlm } from '@smart-llm/core';
import { createOllamaProvider, fetchOllamaCapabilities } from '../../src/index.js';

/** Embedding model name (requires `ollama pull bge-m3`). */
const MODEL = 'bge-m3';

/** Test documents with distinct semantic topics. */
const DOCUMENTS = [
    'The bakery opens at six and sells rye bread and croissants.',
    'Mitochondria produce ATP through oxidative phosphorylation.',
    'The striker scored twice in the second half of the final.',
];

/** Test query matching the second document. */
const QUERY = 'Which organelle makes the energy carrier of the cell?';

/** Test corpus for batch embedding performance verification. */
const MANY = Array.from(
    { length: 150 },
    (_, index) => `Paragraph ${index}: samples were incubated for ${index} minutes before the second wash step.`,
);

/** Test LLM instance with Ollama embedding models. */
const llm = createLlm({
    providers: {
        ollama: {
            adapter: createOllamaProvider(),
            models: {
                bge: { model: MODEL, capabilities: { embeddings: true } },
                bgeSingle: { model: MODEL, capabilities: { embeddings: { supported: true, maxInputs: 1 } } },
            },
        },
    },
});

/** Computes cosine similarity between two numeric vectors. */
function cosine(a: number[], b: number[]): number {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let index = 0; index < a.length; index++) {
        dot += (a[index] ?? 0) * (b[index] ?? 0);
        normA += (a[index] ?? 0) ** 2;
        normB += (b[index] ?? 0) ** 2;
    }
    return dot / Math.sqrt(normA * normB);
}

describe('Ollama embeddings (live, bge-m3)', () => {
    it('reports the embedding capability of bge-m3', async () => {
        const capabilities = await fetchOllamaCapabilities();
        expect(capabilities.ok && capabilities.data[`${MODEL}:latest`]).toEqual({ embeddings: true });
    });

    it('returns normalized 1024-dimensional vectors, the query closest to its answer', async () => {
        const documents = await llm.embed({ model: 'bge', input: DOCUMENTS });
        const query = await llm.embed({ model: 'bge', input: QUERY });
        expect(documents.ok && query.ok).toBe(true);
        if (!(documents.ok && query.ok)) {
            console.log(JSON.stringify([documents, query]));
            return;
        }
        expect(documents.data).toMatchObject({ calls: 1, attempts: 1, model: MODEL });
        expect(documents.data.usage.inputTokens).toBeGreaterThan(0);
        const [queryVector = []] = query.data.embeddings;
        expect(queryVector).toHaveLength(1024);
        expect(Math.hypot(...queryVector)).toBeCloseTo(1, 3);
        const scores = documents.data.embeddings.map((vector) => cosine(queryVector, vector));
        console.log('bge-m3 scores', scores.map((score) => score.toFixed(3)).join(' '));
        expect(scores.indexOf(Math.max(...scores))).toBe(1);
    });

    it('embeds 150 texts in batches of 64 with the same vectors as one text per call', async () => {
        let started = performance.now();
        const batched = await llm.embed({ model: 'bge', input: MANY });
        const batchedMs = performance.now() - started;
        started = performance.now();
        const single = await llm.embed({ model: 'bgeSingle', input: MANY });
        const singleMs = performance.now() - started;
        console.log(`150 texts: batches of 64 ${Math.round(batchedMs)} ms, one per call ${Math.round(singleMs)} ms`);
        expect(batched).toMatchObject({ ok: true, data: { calls: 3 } });
        expect(single).toMatchObject({ ok: true, data: { calls: 150 } });
        if (batched.ok && single.ok) {
            expect(batched.data.embeddings).toHaveLength(150);
            for (const [index, vector] of batched.data.embeddings.entries()) {
                expect(cosine(vector, single.data.embeddings[index] ?? [])).toBeGreaterThan(0.999);
            }
        }
    }, 120_000);

    it('shortens vectors with dimensions and fails an unknown model with a 404', async () => {
        expect(await llm.embed({ model: 'bge', input: 'short', dimensions: 256 })).toMatchObject({ ok: true });
        const shortened = await llm.embed({ model: 'bge', input: 'short', dimensions: 256 });
        expect(shortened.ok && shortened.data.embeddings[0]).toHaveLength(256);
        const missing = createLlm({
            providers: {
                ollama: {
                    adapter: createOllamaProvider(),
                    models: { gone: { model: 'no-such-model', capabilities: { embeddings: true } } },
                },
            },
        });
        expect(await missing.embed({ model: 'gone', input: 'x' })).toMatchObject({
            ok: false,
            error: { code: 'http', status: 404, retryable: false },
        });
    });
});
