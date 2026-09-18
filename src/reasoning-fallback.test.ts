import { buildReasoningFallbackNotice } from "./reasoning-fallback";

describe("buildReasoningFallbackNotice", () => {
  it("keeps an invalid-effort correction in the final status text", () => {
    expect(buildReasoningFallbackNotice("invalid-value")).toBe(
      ":warning: The configured `reasoning-effort` was rejected as invalid. " +
        "Robin completed this run without a reasoning override. Update `.github/robin.yml` " +
        "or the workflow `with: reasoning-effort` value."
    );
  });

  it("distinguishes an unsupported reasoning control", () => {
    expect(buildReasoningFallbackNotice("unsupported")).toContain("rejected as unsupported");
  });

  it("adds nothing when no fallback occurred", () => {
    expect(buildReasoningFallbackNotice()).toBeUndefined();
  });
});
