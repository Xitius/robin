jest.mock("@actions/core", () => ({
  info: jest.fn(),
  warning: jest.fn(),
  error: jest.fn(),
}));

import * as core from "@actions/core";
import { LLMClient } from "./llm-client";

const warningMock = core.warning as unknown as jest.Mock;

interface StubbedOpenAI {
  chat: { completions: { create: jest.Mock } };
}

function buildRequest(client: LLMClient, jsonResponseMode = true) {
  return (
    client as unknown as {
      buildRequest(
        systemPrompt: string,
        userContent: string,
        jsonResponseMode: boolean,
      ): Record<string, unknown>;
    }
  ).buildRequest("system", "user", jsonResponseMode);
}

function stubOpenAI(client: LLMClient): jest.Mock {
  const create = jest.fn();
  (client as unknown as { client: StubbedOpenAI }).client = {
    chat: { completions: { create } },
  };
  return create;
}

function completionResponse(content: string) {
  return {
    model: "resolved-model",
    choices: [{ message: { content }, finish_reason: "stop" }],
  };
}

function reasoningRejection(
  status = 400,
  message = "Unsupported parameter: reasoning is not supported with this model",
) {
  return Object.assign(new Error(message), { status });
}

function streamOf(chunks: unknown[]) {
  return {
    [Symbol.asyncIterator]: async function* () {
      for (const chunk of chunks) {
        yield chunk;
      }
    },
  };
}

function fallbackWarnings(): unknown[][] {
  return warningMock.mock.calls.filter(([message]) =>
    String(message).includes("Retrying once without the reasoning"),
  );
}

describe("LLMClient reasoning request shape", () => {
  it("preserves the default request shape when effort is unset", () => {
    const client = new LLMClient("https://example.test/v1", "test-key", "model");
    const request = buildRequest(client);

    expect(request).not.toHaveProperty("reasoning");
    expect(request).toMatchObject({
      model: "model",
      temperature: 0.1,
      response_format: { type: "json_object" },
    });
    expect(request).not.toHaveProperty("max_tokens");
  });

  it.each(["", "   "])("does not emit reasoning for whitespace effort %j", (effort) => {
    const client = new LLMClient(
      "https://example.test/v1",
      "test-key",
      "model",
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      effort,
    );
    expect(buildRequest(client)).not.toHaveProperty("reasoning");
  });

  it("adds trimmed effort with hidden reasoning excluded", () => {
    const client = new LLMClient(
      "https://example.test/v1",
      "test-key",
      "model",
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      "  high  ",
    );
    expect(buildRequest(client).reasoning).toEqual({ effort: "high", exclude: true });
  });

  it("passes provider-specific effort names through unchanged", () => {
    const client = new LLMClient(
      "https://example.test/v1",
      "test-key",
      "model",
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      "provider-custom",
    );
    expect(buildRequest(client).reasoning).toEqual({
      effort: "provider-custom",
      exclude: true,
    });
  });
});

describe("LLMClient reasoning fallback", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("retries once without reasoning when the provider rejects the parameter", async () => {
    const client = new LLMClient(
      "https://example.test/v1",
      "test-key",
      "model",
      undefined,
      undefined,
      1,
      undefined,
      undefined,
      "high",
    );
    const create = stubOpenAI(client);
    create
      .mockRejectedValueOnce(reasoningRejection())
      .mockResolvedValueOnce(completionResponse("review text"));

    const result = await client.chatCompletion("system", "user");

    expect(result.content).toBe("review text");
    expect(create).toHaveBeenCalledTimes(2);
    expect(create.mock.calls[0][0]).toMatchObject({
      reasoning: { effort: "high", exclude: true },
    });
    expect(create.mock.calls[1][0]).not.toHaveProperty("reasoning");
    expect(fallbackWarnings()).toHaveLength(1);
    expect(client.getReasoningFallbackReason()).toBe("unsupported");
  });

  it("keeps reasoning off for later completions after one fallback", async () => {
    const client = new LLMClient(
      "https://example.test/v1",
      "test-key",
      "model",
      undefined,
      undefined,
      1,
      undefined,
      undefined,
      "high",
    );
    const create = stubOpenAI(client);
    create
      .mockRejectedValueOnce(reasoningRejection(422, "reasoning_effort is not supported"))
      .mockResolvedValueOnce(completionResponse("first"))
      .mockResolvedValueOnce(completionResponse("second"));

    await client.chatCompletion("system", "user");
    await client.chatCompletion("system", "user");

    expect(create).toHaveBeenCalledTimes(3);
    expect(create.mock.calls[2][0]).not.toHaveProperty("reasoning");
    expect(fallbackWarnings()).toHaveLength(1);
  });

  it("does not fall back on auth errors", async () => {
    const client = new LLMClient(
      "https://example.test/v1",
      "test-key",
      "model",
      undefined,
      undefined,
      1,
      undefined,
      undefined,
      "high",
    );
    const create = stubOpenAI(client);
    create.mockRejectedValue(Object.assign(new Error("Invalid API key"), { status: 401 }));

    await expect(client.chatCompletion("system", "user")).rejects.toThrow(
      "Failed to get response from LLM",
    );
    expect(create).toHaveBeenCalledTimes(1);
    expect(fallbackWarnings()).toHaveLength(0);
  });

  it("does not fall back on unrelated 400 validation errors", async () => {
    const client = new LLMClient(
      "https://example.test/v1",
      "test-key",
      "model",
      undefined,
      undefined,
      1,
      undefined,
      undefined,
      "high",
    );
    const create = stubOpenAI(client);
    create.mockRejectedValue(
      Object.assign(new Error("Invalid temperature: only 1 is allowed"), { status: 400 }),
    );

    await expect(client.chatCompletion("system", "user")).rejects.toThrow(
      "Failed to get response from LLM",
    );
    expect(create).toHaveBeenCalledTimes(1);
    expect(fallbackWarnings()).toHaveLength(0);
  });

  it("retries without reasoning when the configured effort is invalid", async () => {
    const client = new LLMClient(
      "https://example.test/v1",
      "test-key",
      "model",
      undefined,
      undefined,
      1,
      undefined,
      undefined,
      "extreme",
    );
    const create = stubOpenAI(client);
    create
      .mockRejectedValueOnce(
        Object.assign(new Error("reasoning effort must be one of low, medium, high"), {
          status: 400,
        }),
      )
      .mockResolvedValueOnce(completionResponse("review text"));

    const result = await client.chatCompletion("system", "user");

    expect(result.content).toBe("review text");
    expect(create).toHaveBeenCalledTimes(2);
    expect(create.mock.calls[0][0]).toMatchObject({
      reasoning: { effort: "extreme", exclude: true },
    });
    expect(create.mock.calls[1][0]).not.toHaveProperty("reasoning");
    expect(fallbackWarnings()).toHaveLength(1);
    expect(client.getReasoningFallbackReason()).toBe("invalid-value");
  });

  it("retries without reasoning when a value rejection repeats the configured value", async () => {
    const client = new LLMClient(
      "https://example.test/v1",
      "test-key",
      "model",
      undefined,
      undefined,
      1,
      undefined,
      undefined,
      "extreme",
    );
    const create = stubOpenAI(client);
    create
      .mockRejectedValueOnce(
        reasoningRejection(400, "reasoning effort 'extreme' is not supported by this model"),
      )
      .mockResolvedValueOnce(completionResponse("review text"));

    const result = await client.chatCompletion("system", "user");

    expect(result.content).toBe("review text");
    expect(create).toHaveBeenCalledTimes(2);
    expect(create.mock.calls[1][0]).not.toHaveProperty("reasoning");
    expect(fallbackWarnings()).toHaveLength(1);
    expect(client.getReasoningFallbackReason()).toBe("invalid-value");
  });

  it("falls back when the provider rejects the exclude sub-key", async () => {
    const client = new LLMClient(
      "https://example.test/v1",
      "test-key",
      "model",
      undefined,
      undefined,
      1,
      undefined,
      undefined,
      "high",
    );
    const create = stubOpenAI(client);
    create
      .mockRejectedValueOnce(reasoningRejection(400, "Unsupported parameter: exclude"))
      .mockResolvedValueOnce(completionResponse("review text"));

    const result = await client.chatCompletion("system", "user");

    expect(result.content).toBe("review text");
    expect(create).toHaveBeenCalledTimes(2);
    expect(create.mock.calls[1][0]).not.toHaveProperty("reasoning");
    expect(fallbackWarnings()).toHaveLength(1);
  });

  it("propagates the error when the fallback retry also rejects", async () => {
    const client = new LLMClient(
      "https://example.test/v1",
      "test-key",
      "model",
      undefined,
      undefined,
      1,
      undefined,
      undefined,
      "high",
    );
    const create = stubOpenAI(client);
    create.mockRejectedValue(reasoningRejection());

    await expect(client.chatCompletion("system", "user")).rejects.toThrow(
      "Failed to get response from LLM",
    );
    expect(create).toHaveBeenCalledTimes(2);
    expect(fallbackWarnings()).toHaveLength(1);
  });

  it("keeps reasoning off for the outer retry after a failed fallback", async () => {
    const client = new LLMClient(
      "https://example.test/v1",
      "test-key",
      "model",
      undefined,
      undefined,
      2,
      undefined,
      undefined,
      "high",
    );
    const create = stubOpenAI(client);
    create
      .mockRejectedValueOnce(reasoningRejection())
      .mockRejectedValueOnce(Object.assign(new Error("backend unavailable"), { status: 500 }))
      .mockResolvedValueOnce(completionResponse("review text"));

    const result = await client.chatCompletion("system", "user");

    expect(result.content).toBe("review text");
    expect(create).toHaveBeenCalledTimes(3);
    expect(create.mock.calls[0][0]).toMatchObject({
      reasoning: { effort: "high", exclude: true },
    });
    expect(create.mock.calls[1][0]).not.toHaveProperty("reasoning");
    expect(create.mock.calls[2][0]).not.toHaveProperty("reasoning");
    expect(fallbackWarnings()).toHaveLength(1);
  });

  it("does not fall back on server errors", async () => {
    const client = new LLMClient(
      "https://example.test/v1",
      "test-key",
      "model",
      undefined,
      undefined,
      1,
      undefined,
      undefined,
      "high",
    );
    const create = stubOpenAI(client);
    create.mockRejectedValue(Object.assign(new Error("reasoning backend failed"), { status: 500 }));

    await expect(client.chatCompletion("system", "user")).rejects.toThrow(
      "Failed to get response from LLM",
    );
    expect(create).toHaveBeenCalledTimes(1);
    expect(fallbackWarnings()).toHaveLength(0);
  });

  it("does not fall back when no effort was configured", async () => {
    const client = new LLMClient(
      "https://example.test/v1",
      "test-key",
      "model",
      undefined,
      undefined,
      1,
    );
    const create = stubOpenAI(client);
    create.mockRejectedValue(reasoningRejection());

    await expect(client.chatCompletion("system", "user")).rejects.toThrow(
      "Failed to get response from LLM",
    );
    expect(create).toHaveBeenCalledTimes(1);
    expect(fallbackWarnings()).toHaveLength(0);
  });

  it("falls back for a rejected reasoning parameter on the streaming router path", async () => {
    const client = new LLMClient(
      "https://example.test/v1",
      "test-key",
      "openrouter/free",
      undefined,
      undefined,
      1,
      undefined,
      undefined,
      "high",
    );
    const create = stubOpenAI(client);
    create
      .mockRejectedValueOnce(reasoningRejection(422, "reasoning_effort is not supported"))
      .mockResolvedValueOnce(
        streamOf([
          { model: "vendor/model", choices: [{ delta: { content: "streamed review" } }] },
        ]),
      );

    const result = await client.chatCompletion("system", "user");

    expect(result.content).toBe("streamed review");
    expect(create).toHaveBeenCalledTimes(2);
    expect(create.mock.calls[0][0]).toMatchObject({
      reasoning: { effort: "high", exclude: true },
    });
    expect(create.mock.calls[1][0]).not.toHaveProperty("reasoning");
    expect(fallbackWarnings()).toHaveLength(1);
  });

  it("logs the provider message for plain-object rejections", async () => {
    const client = new LLMClient(
      "https://example.test/v1",
      "test-key",
      "model",
      undefined,
      undefined,
      1,
      undefined,
      undefined,
      "high",
    );
    const create = stubOpenAI(client);
    create
      .mockRejectedValueOnce({
        status: 400,
        message: "Unsupported parameter: reasoning is not supported",
      })
      .mockResolvedValueOnce(completionResponse("review text"));

    const result = await client.chatCompletion("system", "user");

    expect(result.content).toBe("review text");
    const warnings = fallbackWarnings();
    expect(warnings).toHaveLength(1);
    expect(String(warnings[0][0])).toContain("Unsupported parameter: reasoning is not supported");
    expect(String(warnings[0][0])).not.toContain("[object Object]");
  });

  it("surfaces an unrecognized reasoning rejection on the streaming router path", async () => {
    const client = new LLMClient(
      "https://example.test/v1",
      "test-key",
      "openrouter/free",
      undefined,
      undefined,
      1,
      undefined,
      undefined,
      "high",
    );
    const create = stubOpenAI(client);
    create.mockRejectedValue(
      Object.assign(new Error("reasoning controls rejected by the selected provider"), {
        status: 422,
      }),
    );

    await expect(client.chatCompletion("system", "user")).rejects.toThrow(
      "reasoning controls rejected by the selected provider",
    );
    expect(create).toHaveBeenCalledTimes(1);
    expect(fallbackWarnings()).toHaveLength(0);
  });

  it("surfaces an unrelated validation error that mentions reasoning context", async () => {
    const client = new LLMClient(
      "https://example.test/v1",
      "test-key",
      "openrouter/free",
      undefined,
      undefined,
      1,
      undefined,
      undefined,
      "high",
    );
    const create = stubOpenAI(client);
    create.mockRejectedValue(
      Object.assign(new Error("temperature must be 1 for reasoning models"), { status: 400 }),
    );

    await expect(client.chatCompletion("system", "user")).rejects.toThrow(
      "temperature must be 1 for reasoning models",
    );
    expect(create).toHaveBeenCalledTimes(1);
    expect(fallbackWarnings()).toHaveLength(0);
  });

  it("keeps the stall retry path after a fallback when a reasoning-flavored 400 arrives", async () => {
    const client = new LLMClient(
      "https://example.test/v1",
      "test-key",
      "openrouter/free",
      undefined,
      undefined,
      1,
      undefined,
      undefined,
      "high",
    );
    const create = stubOpenAI(client);
    create
      .mockRejectedValueOnce(reasoningRejection(422, "reasoning_effort is not supported"))
      .mockResolvedValueOnce(
        streamOf([
          { model: "vendor/model", choices: [{ delta: { content: "first review" } }] },
        ]),
      )
      .mockRejectedValueOnce(reasoningRejection());

    await client.chatCompletion("system", "user");
    await expect(client.chatCompletion("system", "user")).rejects.toThrow("OpenRouter stall");

    expect(create).toHaveBeenCalledTimes(3);
    expect(create.mock.calls[2][0]).not.toHaveProperty("reasoning");
    expect(fallbackWarnings()).toHaveLength(1);
  });
});
