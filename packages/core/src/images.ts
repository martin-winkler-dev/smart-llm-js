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

/** Image MIME type pattern: `image/<subtype>`, any case, no parameters. */
const MATCH_IMAGE_TYPE = /^image\/[\w.+-]+$/i;

/** Values of image `detail`; also used by request validation. */
export const IMAGE_DETAILS: ReadonlySet<unknown> = new Set(['auto', 'low', 'high']);

// -------------------------------------------------------------------
// --- 2. Exported Functions
// -------------------------------------------------------------------

/**
 * Creates an image content part from a binary buffer.
 * @throws {TypeError} If data is empty/non-buffer, mimeType is not an image type, or detail is invalid.
 */
export function imagePart(
    data: Uint8Array | ArrayBuffer,
    mimeType: string,
    detail?: 'auto' | 'low' | 'high',
): LlmImagePart {
    if (!isByteBuffer(data)) {
        throw new TypeError('Invalid value for option "data": a Uint8Array, Buffer or ArrayBuffer');
    }
    if (typeof mimeType !== 'string' || !MATCH_IMAGE_TYPE.test(mimeType)) {
        throw new TypeError('Invalid value for option "mimeType": an image type such as image/png');
    }
    if (detail !== undefined && !IMAGE_DETAILS.has(detail)) {
        throw new TypeError('Invalid value for option "detail": auto, low or high');
    }
    const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
    if (bytes.byteLength === 0) {
        throw new TypeError('Invalid value for option "data": non-empty image bytes');
    }
    const base64 = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString('base64');
    const url = `data:${mimeType.toLowerCase()};base64,${base64}`;
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

// -------------------------------------------------------------------
// --- 3. Private Functions
// -------------------------------------------------------------------

/** Checks whether value is a byte array or an array buffer. */
function isByteBuffer(value: unknown): value is Uint8Array | ArrayBuffer {
    return (
        value instanceof Uint8Array ||
        value instanceof ArrayBuffer ||
        (typeof SharedArrayBuffer !== 'undefined' && value instanceof SharedArrayBuffer)
    );
}
