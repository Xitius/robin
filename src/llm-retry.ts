import {
  DEFAULT_LLM_COMPLETION_ATTEMPTS,
  DEFAULT_LLM_RETRY_DELAY_MS,
  DEFAULT_LLM_ROUTER_COMPLETION_ATTEMPTS,
  DEFAULT_LLM_ROUTER_RETRY_DELAY_MS,
  DEFAULT_LLM_ROUTER_TIMEOUT_MS,
  DEFAULT_LLM_TIMEOUT_MS,
} from "./config";

export interface LlmRetryContext {
  model?: string;
}

/** OpenRouter routers (e.g. openrouter/free) pick models dynamically — no secret updates needed. */
export function resolveLlmTimeoutMs(model: string | undefined, timeoutMs: number): number {
  if (timeoutMs !== DEFAULT_LLM_TIMEOUT_MS) return timeoutMs;
  return isOpenRouterRouterModel(model) ? DEFAULT_LLM_ROUTER_TIMEOUT_MS : timeoutMs;
}

export function isOpenRouterRouterModel(model: string | undefined): boolean {
  if (!model) return false;
  const normalized = model.trim().toLowerCase();
  return (
    normalized === "openrouter/free" ||
    normalized === "openrouter/auto" ||
    normalized.startsWith("openrouter/") && normalized.endsWith("/free")
  );
}

export function isOpenRouterProviderError(error: unknown): boolean {
  const message = (error instanceof Error ? error.message : String(error)).toLowerCase();
  return message.includes("provider returned error");
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (
    typeof error === "object" &&
    error !== null &&
    typeof (error as { message?: unknown }).message === "string"
  ) {
    return (error as { message: string }).message;
  }
  return String(error);
}

/** Provider phrases meaning the extra parameter itself is unknown, not that its value is bad. */
const UNSUPPORTED_PARAMETER_PHRASES: RegExp[] = [
  /(?:unsupported|unknown|unrecognized|unrecognised|unexpected)(?:\s+\w+){0,2}\s+(?:parameter|argument|field|property|option|input|feature)\b/i,
  /\bunknown\s+name\b/i,
  /\bcannot\s+(?:bind|find)\s+(?:the\s+)?(?:field|property|parameter)\b/i,
  /(?:parameter|argument|field|property|option|input|feature)\b[^.!?]{0,40}\b(?:unsupported|unknown|unrecognized|unrecognised|unexpected)\b/i,
  /\b(?:is|are|was|were)\s+(?:not\s+supported|unsupported)\b/i,
  /\bnot\s+supported\s+(?:by|for|with|in|on)\b/i,
  /\b(?:does|do|did)\s+not\s+support\b/i,
  /\b(?:parameter|argument|field|property|option|feature)\b[^.!?]{0,30}\b(?:is|are|was|were)\s+not\s+(?:allowed|permitted|recognized|recognised)\b/i,
  /\bextra\s+(?:inputs?|fields?|properties|arguments?|parameters?)\b/i,
];

/** Schema/shape complaints about the reasoning field itself, not about its configured value. */
const SHAPE_MISMATCH_PHRASES: RegExp[] = [
  /\binput should be (?:a|an)\s+(?:valid\s+)?(?:string|object|boolean|number|array)\b/i,
];

/** Malformed-value signals: these must keep failing rather than mask a configuration typo. */
const INVALID_VALUE_PHRASES: RegExp[] = [
  /\binvalid\s+(?:value|type|format)\b/i,
  /\b(?:must|should|needs?\s+to)\s+be\s+(?:one\s+of|between|greater|less|at\s+most|at\s+least|a|an)\b/i,
  /\b(?:expected|not)\s+one\s+of\b/i,
  /\bout\s+of\s+range\b/i,
  /\b(?:valid|allowed)\s+values?\s+(?:are|is)\b/i,
  /\bnot\s+a\s+valid\b/i,
];

/**
 * True only for a client validation response (400/422) that reports the reasoning
 * configuration itself as unknown, unsupported, or of the wrong shape — the cases where
 * dropping the reasoning parameter and retrying is safe. Invalid effort values, missing
 * values, and generic validation errors must surface normally: a rejection that repeats the
 * configured effort value is a value complaint, not an unknown-parameter report.
 */
export function isUnsupportedReasoningEffortError(error: unknown, sentEffort?: string): boolean {
  if (!error || typeof error !== "object") return false;
  const status = Number((error as { status?: unknown }).status);
  if (status !== 400 && status !== 422) return false;
  const message = errorMessage(error);
  if (!/\b(?:reasoning|effort|exclude)/i.test(message)) return false;
  if (sentEffort && mentionsEffortValue(message, sentEffort)) return false;
  if (SHAPE_MISMATCH_PHRASES.some((pattern) => pattern.test(message))) return true;
  if (INVALID_VALUE_PHRASES.some((pattern) => pattern.test(message))) return false;
  return UNSUPPORTED_PARAMETER_PHRASES.some((pattern) => pattern.test(message));
}

/** Word-boundary match so short values like `low` or `max` cannot hit `follow` or `maximum`. */
function mentionsEffortValue(message: string, effort: string): boolean {
  const escaped = effort.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(?:^|\\W)${escaped}(?:$|\\W)`, "i").test(message);
}

export function isRetriableLlmError(error: unknown, context: LlmRetryContext = {}): boolean {
  if (!error) return false;

  const routerModel = isOpenRouterRouterModel(context.model);

  if (typeof error === "object" && error !== null && "status" in error) {
    const status = Number((error as { status?: number }).status);
    if (status === 429 || (Number.isFinite(status) && status >= 500)) {
      return true;
    }
    if (status === 404 && routerModel) {
      return true;
    }
    if (Number.isFinite(status) && status >= 400 && status < 500) {
      return false;
    }
  }

  const message = (error instanceof Error ? error.message : String(error)).toLowerCase();
  if (routerModel && (message.includes("404") || isOpenRouterProviderError(error))) {
    return true;
  }

  return (
    message.includes("timeout") ||
    message.includes("timed out") ||
    message.includes("econnreset") ||
    message.includes("econnrefused") ||
    message.includes("network") ||
    message.includes("socket hang up") ||
    message.includes("rate limit") ||
    message.includes("overloaded") ||
    message.includes("empty response from llm") ||
    message.includes("openrouter stall")
  );
}

export function shouldUseJsonResponseMode(
  attempt: number,
  jsonResponseMode: boolean
): boolean {
  return jsonResponseMode && attempt === 1;
}

export function computeRetryDelayMs(
  attempt: number,
  context: LlmRetryContext = {},
  baseDelayMs = isOpenRouterRouterModel(context.model)
    ? DEFAULT_LLM_ROUTER_RETRY_DELAY_MS
    : DEFAULT_LLM_RETRY_DELAY_MS
): number {
  return baseDelayMs * attempt;
}

export function getLlmCompletionAttemptCount(
  maxAttempts = DEFAULT_LLM_COMPLETION_ATTEMPTS,
  model?: string
): number {
  const resolved =
    maxAttempts === DEFAULT_LLM_COMPLETION_ATTEMPTS && isOpenRouterRouterModel(model)
      ? DEFAULT_LLM_ROUTER_COMPLETION_ATTEMPTS
      : maxAttempts;
  return Math.max(1, Math.floor(resolved));
}

export async function delayMs(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

export function openRouterStallError(firstChunkMs: number): Error {
  return new Error(`OpenRouter stall: no first response within ${firstChunkMs} ms`);
}
