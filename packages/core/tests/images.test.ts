// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { afterAll, beforeAll, describe, expect, it } from 'bun:test';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { imageFromFile, imagePart } from '../src/images.js';

describe('imagePart', () => {
    it('creates an image part from Uint8Array or ArrayBuffer', () => {
        const bytes = new Uint8Array([1, 2, 3]);
        expect(imagePart(bytes, 'image/png')).toEqual({
            type: 'image_url',
            image_url: { url: 'data:image/png;base64,AQID' },
        });

        expect(imagePart(bytes.buffer, 'image/jpeg', 'high')).toEqual({
            type: 'image_url',
            image_url: { url: 'data:image/jpeg;base64,AQID', detail: 'high' },
        });
    });

    it('throws TypeError if mimeType does not start with image/', () => {
        expect(() => imagePart(new Uint8Array([1]), 'text/plain')).toThrow(
            'Invalid value for option "mimeType": an image type such as image/png',
        );
    });

    it('lowercases MIME type in data URL', () => {
        expect(imagePart(new Uint8Array([1, 2, 3]), 'IMAGE/PNG').image_url.url).toBe('data:image/png;base64,AQID');
    });

    it('throws TypeError for a malformed or non-string MIME type, empty data or an invalid detail', () => {
        const bytes = new Uint8Array([1]);
        const mime = 'Invalid value for option "mimeType": an image type such as image/png';
        for (const type of ['image/', 'image/png; charset=x', 'image/png ', 5, undefined]) {
            expect(() => imagePart(bytes, type as string)).toThrow(mime);
        }
        expect(() => imagePart(new Uint8Array(0), 'image/png')).toThrow(
            'Invalid value for option "data": non-empty image bytes',
        );
        expect(() => imagePart(new ArrayBuffer(0), 'image/png')).toThrow('option "data"');
        expect(() => imagePart(bytes, 'image/png', 'ultra' as never)).toThrow(
            'Invalid value for option "detail": auto, low or high',
        );
    });

    it('throws TypeError for non-buffer data', () => {
        const message = 'Invalid value for option "data": a Uint8Array, Buffer or ArrayBuffer';
        for (const data of [5, 'AQID', [1, 2], null, undefined, { byteLength: 1 }]) {
            expect(() => imagePart(data as never, 'image/png')).toThrow(message);
        }
        expect(imagePart(Buffer.from([1, 2, 3]), 'image/png').image_url.url).toBe('data:image/png;base64,AQID');
        expect(imagePart(new Uint8Array([1, 2, 3]).buffer, 'image/png').image_url.url).toBe(
            'data:image/png;base64,AQID',
        );
    });
});

describe('imageFromFile', () => {
    let testDir: string;

    beforeAll(async () => {
        testDir = await mkdtemp(join(tmpdir(), 'smart-llm-images-'));
    });

    afterAll(async () => {
        if (testDir !== undefined) {
            await rm(testDir, { recursive: true, force: true });
        }
    });

    it('loads an image from disk and derives mime type from extension case-insensitively', async () => {
        const cases = [
            ['sample.png', 'image/png'],
            ['sample.jpg', 'image/jpeg'],
            ['sample.jpeg', 'image/jpeg'],
            ['sample.webp', 'image/webp'],
            ['sample.gif', 'image/gif'],
            ['sample.PNG', 'image/png'],
        ] as const;

        for (const [filename, expectedMime] of cases) {
            const filePath = join(testDir, filename);
            await writeFile(filePath, new Uint8Array([1, 2, 3]));
            const part = await imageFromFile(filePath);
            expect(part).toEqual({
                type: 'image_url',
                image_url: { url: `data:${expectedMime};base64,AQID` },
            });
        }
    });

    it('accepts an explicit mimeType override', async () => {
        const filePath = join(testDir, 'raw.bin');
        await writeFile(filePath, new Uint8Array([4, 5, 6]));
        const part = await imageFromFile(filePath, 'image/webp');
        expect(part).toEqual({
            type: 'image_url',
            image_url: { url: 'data:image/webp;base64,BAUG' },
        });
    });

    it('throws TypeError for unknown image extension when mimeType is omitted', async () => {
        await expect(imageFromFile('file.unknown')).rejects.toThrow(
            'Invalid value for option "path": unknown image type of file.unknown',
        );
    });
});
