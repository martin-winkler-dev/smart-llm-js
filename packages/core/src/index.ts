// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

export type {
    Llm,
    LlmAnswer,
    LlmCallOptions,
    LlmCapabilities,
    LlmConfig,
    LlmContentPart,
    LlmDefaults,
    LlmEmbedInput,
    LlmEmbedOutput,
    LlmEmbedRequest,
    LlmEmbedTask,
    LlmError,
    LlmErrorCode,
    LlmImagePart,
    LlmJsonMode,
    LlmJsonSchema,
    LlmLimiterOptions,
    LlmLimits,
    LlmMessage,
    LlmModel,
    LlmModelCall,
    LlmModelCapabilities,
    LlmModelInfo,
    LlmModelInput,
    LlmModelNames,
    LlmModelOptions,
    LlmModelSpec,
    LlmOutput,
    LlmProvider,
    LlmProviderEmbedOutput,
    LlmProviderEmbedRequest,
    LlmProviderError,
    LlmProviderOutput,
    LlmProviderRequest,
    LlmProviderSpec,
    LlmProviderSupport,
    LlmReasoningEffort,
    LlmRequest,
    LlmReservedName,
    LlmResponseFormat,
    LlmRole,
    LlmSettings,
    LlmSource,
    LlmTextPart,
    LlmTool,
    LlmToolCall,
    LlmToolChoice,
    LlmUsage,
    Result,
} from './types.js';

// -------------------------------------------------------------------
// --- 2. Exported Functions
// -------------------------------------------------------------------

export { imageFromFile, imagePart } from './images.js';
export { createLlm } from './llm.js';
export { assistantMessage, toolMessage } from './messages.js';
export {
    cleanSchema,
    findSchemaMismatch,
    parseJsonText,
    removeOptionalNulls,
    schemaInstruction,
    toStrictSchema,
} from './schema.js';
