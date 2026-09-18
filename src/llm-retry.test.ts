import {
  computeRetryDelayMs,
  getLlmCompletionAttemptCount,
  isOpenRouterRouterModel,
  isRetriableLlmError,
  isUnsupportedReasoningEffortError,
  openRouterStallError,
  resolveLlmTimeoutMs,
  shouldUseJsonResponseMode,
} from "./llm-retry";
import {
  DEFAULT_LLM_COMPLETION_ATTEMPTS,
  DEFAULT_LLM_ROUTER_COMPLETION_ATTEMPTS,
  DEFAULT_LLM_ROUTER_RETRY_DELAY_MS,
  DEFAULT_LLM_ROUTER_TIMEOUT_MS,
  DEFAULT_LLM_TIMEOUT_MS,
} from "./config";

describe("openRouterStallError", () => {
  it("produces a retriable stall message", () => {
    const error = openRouterStallError(45000);
    expect(error.message).toContain("OpenRouter stall");
    expect(isRetriableLlmError(error, { model: "openrouter/free" })).toBe(true);
  });
});

describe("resolveLlmTimeoutMs", () => {
  it("shortens the default timeout for OpenRouter routers", () => {
    expect(resolveLlmTimeoutMs("openrouter/free", DEFAULT_LLM_TIMEOUT_MS)).toBe(
      DEFAULT_LLM_ROUTER_TIMEOUT_MS
    );
    expect(resolveLlmTimeoutMs("gpt-4o", DEFAULT_LLM_TIMEOUT_MS)).toBe(DEFAULT_LLM_TIMEOUT_MS);
  });

  it("keeps an explicit consumer override", () => {
    expect(resolveLlmTimeoutMs("openrouter/free", 300000)).toBe(300000);
  });
});

describe("isOpenRouterRouterModel", () => {
  it("detects OpenRouter free and auto routers", () => {
    expect(isOpenRouterRouterModel("openrouter/free")).toBe(true);
    expect(isOpenRouterRouterModel("openrouter/auto")).toBe(true);
    expect(isOpenRouterRouterModel("gpt-4o")).toBe(false);
  });
});

describe("isRetriableLlmError", () => {
  it("retries rate limits and server errors", () => {
    expect(isRetriableLlmError({ status: 429 })).toBe(true);
    expect(isRetriableLlmError({ status: 502 })).toBe(true);
  });

  it("does not retry client auth or validation errors", () => {
    expect(isRetriableLlmError({ status: 401 })).toBe(false);
    expect(isRetriableLlmError({ status: 400 })).toBe(false);
  });

  it("retries network and timeout messages", () => {
    expect(isRetriableLlmError(new Error("Request timed out"))).toBe(true);
    expect(isRetriableLlmError(new Error("ECONNRESET"))).toBe(true);
    expect(
      isRetriableLlmError(new Error("OpenRouter stall: no first response within 45000 ms"), {
        model: "openrouter/free",
      })
    ).toBe(true);
  });

  it("retries OpenRouter provider 404s for router models", () => {
    expect(
      isRetriableLlmError(new Error("404 Provider returned error"), {
        model: "openrouter/free",
      })
    ).toBe(true);
    expect(isRetriableLlmError({ status: 404 }, { model: "openrouter/free" })).toBe(true);
    expect(isRetriableLlmError({ status: 404 }, { model: "gpt-4o" })).toBe(false);
  });
});

describe("isUnsupportedReasoningEffortError", () => {
  it("detects 400/422 responses that report the reasoning/effort parameter as unknown", () => {
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "Unsupported parameter: 'reasoning' is not supported with this model",
      })
    ).toBe(true);
    expect(
      isUnsupportedReasoningEffortError(
        Object.assign(new Error("reasoning_effort is not supported"), { status: 422 })
      )
    ).toBe(true);
    expect(
      isUnsupportedReasoningEffortError({ status: 400, message: "Unknown parameter: effort" })
    ).toBe(true);
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "Unrecognized request argument supplied: reasoning",
      })
    ).toBe(true);
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "This model does not support reasoning",
      })
    ).toBe(true);
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "The reasoning effort control is not supported for this model",
      })
    ).toBe(true);
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "reasoning not supported by this model",
      })
    ).toBe(true);
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "The reasoning parameter is not allowed for this model",
      })
    ).toBe(true);
    expect(
      isUnsupportedReasoningEffortError({
        status: 422,
        message: "reasoning: Extra inputs are not permitted",
      })
    ).toBe(true);
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: 'Unknown name "reasoning": Cannot bind field.',
      })
    ).toBe(true);
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message:
          "Invalid JSON payload received. Unknown name \"reasoning\" at 'reasoning': Cannot find field.",
      })
    ).toBe(true);
    expect(
      isUnsupportedReasoningEffortError({
        status: 422,
        message: "reasoning: Input should be a valid string",
      })
    ).toBe(true);
  });

  it("lets an explicit parameter rejection win over value words elsewhere", () => {
    expect(
      isUnsupportedReasoningEffortError(
        {
          status: 400,
          message: "Unsupported parameter: reasoning; valid values are low, medium, high",
        },
        "low"
      )
    ).toBe(true);
    expect(
      isUnsupportedReasoningEffortError(
        { status: 400, message: "This model does not support high-effort reasoning" },
        "high"
      )
    ).toBe(true);
    expect(
      isUnsupportedReasoningEffortError(
        { status: 400, message: "reasoning is not supported with this model for effort high" },
        "high"
      )
    ).toBe(true);
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "reasoning is not one of the supported parameters",
      })
    ).toBe(true);
  });

  it("keeps a mixed invalid-value message on the value path", () => {
    const mixed = {
      status: 400,
      message: "Unsupported value for parameter reasoning: must be one of low, medium, high",
    };
    expect(isUnsupportedReasoningEffortError(mixed, "extreme")).toBe(false);
    expect(isUnsupportedReasoningEffortError(mixed, "low")).toBe(false);
  });

  it("matches explicit parameter rejections with underscored and hyphenated names", () => {
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "Unsupported parameter: reasoning_effort",
      })
    ).toBe(true);
    expect(
      isUnsupportedReasoningEffortError({
        status: 422,
        message: "reasoning-effort is not supported",
      })
    ).toBe(true);
  });

  it("uses a structured param naming the reasoning field when message text is inconclusive", () => {
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "Request validation failed",
        param: "reasoning_effort",
      })
    ).toBe(true);
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "Invalid value for 'reasoning_effort': 'extreme'",
        param: "reasoning_effort",
      })
    ).toBe(false);
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "Request validation failed",
        param: "temperature",
      })
    ).toBe(false);
  });

  it("treats a message repeating the configured effort value as a value complaint", () => {
    const valueRejection = {
      status: 400,
      message: "reasoning effort 'extreme' is not supported by this model",
    };
    expect(isUnsupportedReasoningEffortError(valueRejection, "extreme")).toBe(false);
    expect(
      isUnsupportedReasoningEffortError(
        { status: 400, message: "reasoning is not supported with this model" },
        "extreme"
      )
    ).toBe(true);
    expect(
      isUnsupportedReasoningEffortError(
        { status: 400, message: "Unsupported parameter: reasoning; follow the docs" },
        "low"
      )
    ).toBe(true);
  });

  it("detects a rejected exclude sub-key of the reasoning request", () => {
    expect(
      isUnsupportedReasoningEffortError({ status: 400, message: "Unsupported parameter: exclude" })
    ).toBe(true);
    expect(
      isUnsupportedReasoningEffortError({ status: 400, message: "Invalid API key" })
    ).toBe(false);
  });

  it("ignores auth, rate-limit, server, timeout, and unrelated validation errors", () => {
    expect(
      isUnsupportedReasoningEffortError({ status: 401, message: "Invalid API key" })
    ).toBe(false);
    expect(
      isUnsupportedReasoningEffortError({ status: 429, message: "reasoning rate limit" })
    ).toBe(false);
    expect(
      isUnsupportedReasoningEffortError({ status: 500, message: "reasoning backend failed" })
    ).toBe(false);
    expect(
      isUnsupportedReasoningEffortError({ status: 400, message: "Invalid temperature" })
    ).toBe(false);
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "temperature 2 is not supported; reasoning models require 1",
      })
    ).toBe(false);
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "temperature 2 is not supported for reasoning models",
      })
    ).toBe(false);
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "reasoning models do not support temperature 0.1",
      })
    ).toBe(false);
    expect(isUnsupportedReasoningEffortError(new Error("reasoning rejected"))).toBe(false);
    expect(isUnsupportedReasoningEffortError(undefined)).toBe(false);
  });

  it("does not treat invalid, out-of-range, or missing reasoning values as unsupported", () => {
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "reasoning effort must be one of low, medium, high",
      })
    ).toBe(false);
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "reasoning effort should be one of low, medium, high",
      })
    ).toBe(false);
    expect(
      isUnsupportedReasoningEffortError({
        status: 422,
        message: "reasoning_effort: invalid value; expected one of low, medium, high",
      })
    ).toBe(false);
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "Invalid value for 'reasoning_effort': 'extreme' is not one of [low, medium, high]",
      })
    ).toBe(false);
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "reasoning_effort 'extreme' is not allowed; use low, medium, or high",
      })
    ).toBe(false);
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "reasoning effort out of range: allowed values are low, medium, high",
      })
    ).toBe(false);
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "Missing required parameter: reasoning_effort",
      })
    ).toBe(false);
    expect(
      isUnsupportedReasoningEffortError({
        status: 400,
        message: "reasoning effort is required",
      })
    ).toBe(false);
  });
});

describe("shouldUseJsonResponseMode", () => {
  it("uses JSON only on the first attempt", () => {
    expect(shouldUseJsonResponseMode(1, true)).toBe(true);
    expect(shouldUseJsonResponseMode(2, true)).toBe(false);
    expect(shouldUseJsonResponseMode(1, false)).toBe(false);
  });
});

describe("computeRetryDelayMs", () => {
  it("backs off linearly by attempt", () => {
    expect(computeRetryDelayMs(1, {}, 1000)).toBe(1000);
    expect(computeRetryDelayMs(2, {}, 1000)).toBe(2000);
  });

  it("uses longer base delay for router models", () => {
    expect(computeRetryDelayMs(1, { model: "openrouter/free" })).toBe(
      DEFAULT_LLM_ROUTER_RETRY_DELAY_MS
    );
  });
});

describe("getLlmCompletionAttemptCount", () => {
  it("clamps invalid values to at least one", () => {
    expect(getLlmCompletionAttemptCount(0)).toBe(1);
    expect(getLlmCompletionAttemptCount(2.7)).toBe(2);
  });

  it("uses more attempts for OpenRouter router models by default", () => {
    expect(getLlmCompletionAttemptCount(DEFAULT_LLM_COMPLETION_ATTEMPTS, "openrouter/free")).toBe(
      DEFAULT_LLM_ROUTER_COMPLETION_ATTEMPTS
    );
    expect(getLlmCompletionAttemptCount(DEFAULT_LLM_COMPLETION_ATTEMPTS, "gpt-4o")).toBe(
      DEFAULT_LLM_COMPLETION_ATTEMPTS
    );
  });
});
