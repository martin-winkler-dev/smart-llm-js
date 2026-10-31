// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Topological dependency order for publishing workspace packages. */
const PACKAGES = ['core', 'openai-compatible', 'gemini', 'deepseek', 'groq', 'ollama'] as const;

/** Source repository root path. */
const REPO_ROOT = resolve(import.meta.dir, '..');

/** Destination directory for packed publish tarballs. */
const DIST_PACK = join(REPO_ROOT, 'node_modules', '.cache', 'dist-pack');

/** Regex pattern matching npm web authorization URLs. */
const MATCH_AUTH_URL = /https:\/\/(?:www\.)?npmjs\.com\/auth\/cli\/[A-Za-z0-9_-]+/;

// -------------------------------------------------------------------
// --- 2. Main
// -------------------------------------------------------------------

/**
 * Packs workspace packages into standalone tarballs and publishes to npm.
 * Converts workspace:^ dependencies to explicit version strings during pack.
 * Automatically opens browser authorization URLs when prompted by npm.
 */
async function main() {
    const args = process.argv.slice(2);
    const dryRun = args.includes('--dry-run');
    const skipBuild = args.includes('--skip-build');

    console.log('=== @smart-llm Publish Suite ===');

    if (!skipBuild) {
        console.log('\n1. Building workspace packages...');
        const build = Bun.spawnSync(['bun', 'run', 'build'], {
            cwd: REPO_ROOT,
            stdio: ['inherit', 'inherit', 'inherit'],
        });
        if (build.exitCode !== 0) {
            console.error('Build step failed.');
            process.exit(1);
        }
    }

    console.log('\n2. Packing tarballs with resolved workspace dependencies...');
    mkdirSync(DIST_PACK, { recursive: true });

    const tarballs: Record<string, string> = {};
    for (const pkg of PACKAGES) {
        const tarball = join(DIST_PACK, `${pkg}.tgz`);
        const pack = Bun.spawnSync(['bun', 'pm', 'pack', '--cwd', `packages/${pkg}`, '--filename', tarball], {
            cwd: REPO_ROOT,
        });
        if (pack.exitCode !== 0) {
            console.error(`Packing packages/${pkg} failed: ${pack.stderr.toString()}`);
            process.exit(1);
        }
        tarballs[pkg] = tarball;
        console.log(`  ✓ Packed @smart-llm/${pkg} -> ${tarball}`);
    }

    console.log(`\n3. Publishing ${PACKAGES.length} packages${dryRun ? ' (DRY RUN)' : ''}...`);

    for (const pkg of PACKAGES) {
        const name = `@smart-llm/${pkg}`;
        const tarball = tarballs[pkg];
        console.log(`\n--> Publishing ${name}...`);

        const publishArgs = ['publish', tarball, '--access', 'public'];
        if (dryRun) {
            publishArgs.push('--dry-run');
        }

        const proc = Bun.spawn(['npm', ...publishArgs], {
            cwd: REPO_ROOT,
            stdin: 'inherit',
            stdout: 'pipe',
            stderr: 'pipe',
        });

        // Stream and inspect both stdout and stderr for npm browser auth URLs
        const stdoutDone = pipeWithUrlDetection(proc.stdout, process.stdout);
        const stderrDone = pipeWithUrlDetection(proc.stderr, process.stderr);

        const exitCode = await proc.exited;
        await Promise.all([stdoutDone, stderrDone]);

        if (exitCode !== 0) {
            console.error(`\nFailed to publish ${name}. Exit code: ${exitCode}`);
            process.exit(1);
        }
        console.log(`  ✓ Successfully published ${name}`);
    }

    console.log('\nAll 6 packages published successfully.');
}

/** Streams output stream to target and automatically opens detected npm auth URLs in the default browser. */
async function pipeWithUrlDetection(source: ReadableStream<Uint8Array>, target: NodeJS.WriteStream): Promise<void> {
    const reader = source.getReader();
    const decoder = new TextDecoder();
    const openedUrls = new Set<string>();

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                break;
            }
            target.write(value);
            const text = decoder.decode(value);
            const match = MATCH_AUTH_URL.exec(text);
            if (match && !openedUrls.has(match[0])) {
                openedUrls.add(match[0]);
                openBrowser(match[0]);
            }
        }
    } finally {
        reader.releaseLock();
    }
}

/** Opens URL in the default web browser. */
function openBrowser(url: string): void {
    console.log(`\n[Auto-launching browser approval: ${url}]`);
    try {
        if (process.platform === 'win32') {
            Bun.spawn(['cmd', '/c', 'start', '', url]);
        } else if (process.platform === 'darwin') {
            Bun.spawn(['open', url]);
        } else {
            Bun.spawn(['xdg-open', url]);
        }
    } catch {
        // Fallback: URL is already written to terminal
    }
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
