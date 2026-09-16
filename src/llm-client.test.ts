import { LLMClient } from "./llm-client";

function buildRequest(client: LLMClient) {
  return (client as unknown as {
    buildRequest(
      systemPrompt: string,
      userContent: string,
      jsonResponseMode: boolean,
    ): Record<string, unknown>;
  }).buildRequest("system", "user", true);
}

describe("LLMClient reasoning request controls", () => {
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

  it("adds high reasoning with hidden output excluded", () => {
    const client = new LLMClient(
      "https://example.test/v1", "test-key", "model", 16384, undefined,
      undefined, undefined, undefined, "high", true,
    );
    expect(buildRequest(client).reasoning).toEqual({ effort: "high", exclude: true });
  });

  it("supports visible reasoning when explicitly requested", () => {
    const client = new LLMClient(
      "https://example.test/v1", "test-key", "model", undefined, undefined,
      undefined, undefined, undefined, "high", false,
    );
    expect(buildRequest(client).reasoning).toEqual({ effort: "high", exclude: false });
  });

  it.each(["", "   "])("does not emit reasoning for whitespace effort %j", (effort) => {
    const client = new LLMClient(
      "https://example.test/v1", "test-key", "model", undefined, undefined,
      undefined, undefined, undefined, effort, false,
    );
    expect(buildRequest(client)).not.toHaveProperty("reasoning");
  });

  it("passes provider-specific effort names through unchanged", () => {
    const client = new LLMClient(
      "https://example.test/v1", "test-key", "model", undefined, undefined,
      undefined, undefined, undefined, "provider-custom", true,
    );
    expect(buildRequest(client).reasoning).toEqual({ effort: "provider-custom", exclude: true });
  });
});
