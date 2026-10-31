// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

// Live: embeddings on the Gemini Developer API with the free key in env GEMINI_API_KEY.
// Run with `bun run test:live:gemini` from the repository root; never part of `bun test`.

import { describe, expect, it } from 'bun:test';
import { createLlm } from '@smart-llm/core';
import { createGeminiProvider } from '../../src/index.js';

/** Test documents with section titles. */
const DOCUMENTS = [
    { text: 'The bakery opens at six and sells rye bread and croissants.', title: 'Shop' },
    { text: 'Mitochondria produce ATP through oxidative phosphorylation.', title: 'Cell biology' },
    { text: 'The striker scored twice in the second half of the final.', title: 'Sports' },
];

/** Test query corresponding to the second document. */
const QUERY = 'Which organelle makes the energy carrier of the cell?';

/** Test LLM instance with Gemini embedding models. */
const llm = createLlm({
    providers: {
        gemini: {
            adapter: createGeminiProvider({ timeoutMs: 30_000 }),
            models: {
                embed1: { model: 'gemini-embedding-001', capabilities: { embeddings: true } },
                embed2: { model: 'gemini-embedding-2', capabilities: { embeddings: true } },
                embed2Pairs: {
                    model: 'gemini-embedding-2',
                    capabilities: { embeddings: { supported: true, maxInputs: 2 } },
                },
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

describe('Gemini embeddings through createLlm (live, free key)', () => {
    for (const model of ['embed1', 'embed2'] as const) {
        it(`${model}: document and query vectors of 768 dimensions, the query closest to its answer`, async () => {
            // gemini-embedding-001 accepts one title per provider call; omit titles for batch
            const input = model === 'embed2' ? DOCUMENTS : DOCUMENTS.map((document) => document.text);
            const documents = await llm.embed({ model, input, task: 'document', dimensions: 768 });
            const query = await llm.embed({ model, input: QUERY, task: 'query', dimensions: 768 });
            expect(documents.ok && query.ok).toBe(true);
            if (!(documents.ok && query.ok)) {
                console.log(JSON.stringify([documents, query]));
                return;
            }
            expect(documents.data).toMatchObject({ calls: 1, usage: { inputTokens: null } });
            const [queryVector = []] = query.data.embeddings;
            expect(queryVector).toHaveLength(768);
            const scores = documents.data.embeddings.map((vector) => cosine(queryVector, vector));
            console.log(model, 'scores', scores.map((score) => score.toFixed(3)).join(' '));
            expect(scores.indexOf(Math.max(...scores))).toBe(1);
        }, 60_000);
    }

    it('splits into provider calls of maxInputs and keeps the input order', async () => {
        const whole = await llm.embed({ model: 'embed2', input: DOCUMENTS, task: 'document', dimensions: 256 });
        const pairs = await llm.embed({ model: 'embed2Pairs', input: DOCUMENTS, task: 'document', dimensions: 256 });
        expect(pairs).toMatchObject({ ok: true, data: { calls: 2 } });
        if (whole.ok && pairs.ok) {
            for (const [index, vector] of pairs.data.embeddings.entries()) {
                expect(cosine(vector, whole.data.embeddings[index] ?? [])).toBeGreaterThan(0.99);
            }
        }
    }, 60_000);
});
