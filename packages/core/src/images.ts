// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { readFile } from 'node:fs/promises';
import { extname } from 'node:path';
import type { LlmImagePart } from './types.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Image types by file extension; formats every provider takes. */
const MIME_TYPES: Readonly<Record<string, string>> = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.webp': 'image/webp',
    '.gif': 'image/gif',
};

// -------------------------------------------------------------------
// --- 2. Exported Functions
// -------------------------------------------------------------------

/**
 * Formats image byte buffer as OpenAI image part (base64 data URL).
 * @throws {TypeError} If `mimeType` does not start with `image/`.
 */
export function imagePart(
    data: Uint8Array | ArrayBuffer,
    mimeType: string,
    detail?: 'auto' | 'low' | 'high',
): LlmImagePart {
    if (!mimeType.startsWith('image/')) {
        throw new TypeError('Invalid value for option "mimeType": an image type such as image/png');
    }
    const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
    const url = `data:${mimeType};base64,${Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString('base64')}`;
    const image = detail === undefined ? { url } : { url, detail };
    return {
        type: 'image_url',
        // biome-ignore lint/style/useNamingConvention: OpenAI field name
        image_url: image,
    };
}

/**
 * Reads image from disk and formats as image part.
 * @throws {TypeError} If file extension is unsupported and `mimeType` is omitted.
 */
export async function imageFromFile(path: string, mimeType?: string): Promise<LlmImagePart> {
    const type = mimeType ?? MIME_TYPES[extname(path).toLowerCase()];
    if (type === undefined) {
        throw new TypeError(`Invalid value for option "path": unknown image type of ${path}`);
    }
    return imagePart(await readFile(path), type);
}
