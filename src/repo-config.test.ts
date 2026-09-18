import {
  DEFAULT_MAX_COMMENTS,
  DEFAULT_ACTION_MAX_DIFF_SIZE,
  parseRepoConfigYaml,
  resolveJsonResponseMode,
  resolveMaxComments,
  resolveMaxDiffSize,
  resolveReasoningEffort,
  resolveRequestChanges,
} from "./repo-config";

describe("parseRepoConfigYaml", () => {
  it("parses supported keys", () => {
    const config = parseRepoConfigYaml(`
max-diff-size: 25000
max-comments: 8
json-response-mode: false
skip-paths:
  - "**/generated/**"
  - vendor/**
`);

    expect(config.maxDiffSize).toBe(25000);
    expect(config.maxComments).toBe(8);
    expect(config.jsonResponseMode).toBe(false);
    expect(config.skipPaths).toEqual(["**/generated/**", "vendor/**"]);
  });

  it("parses reasoning-effort as a case-preserving string", () => {
    expect(parseRepoConfigYaml("reasoning-effort: high").reasoningEffort).toBe("high");
    expect(parseRepoConfigYaml('reasoning-effort: "xhigh"').reasoningEffort).toBe("xhigh");
    expect(parseRepoConfigYaml("reasoning-effort: 'medium'").reasoningEffort).toBe("medium");
    expect(parseRepoConfigYaml("reasoning-effort: ProviderCustom").reasoningEffort).toBe(
      "ProviderCustom"
    );
  });

  it("leaves reasoning-effort unset when the repo config value is empty", () => {
    expect(parseRepoConfigYaml("reasoning-effort:").reasoningEffort).toBeUndefined();
    expect(parseRepoConfigYaml('reasoning-effort: ""').reasoningEffort).toBeUndefined();
  });

  it("tolerates the inline comments the shipped examples use", () => {
    expect(
      parseRepoConfigYaml("reasoning-effort: high   # provider-dependent; unset sends none")
        .reasoningEffort
    ).toBe("high");
    expect(
      parseRepoConfigYaml("request-changes: false # advisor mode").requestChanges
    ).toBe(false);
  });

  it("keeps a hash that is part of a quoted value", () => {
    expect(parseRepoConfigYaml('reasoning-effort: "provider#custom"').reasoningEffort).toBe(
      "provider#custom"
    );
  });
});

describe("resolveMaxDiffSize", () => {
  it("uses repo config when action input is still the default", () => {
    expect(
      resolveMaxDiffSize(String(DEFAULT_ACTION_MAX_DIFF_SIZE), { maxDiffSize: 25000 })
    ).toBe(25000);
  });

  it("keeps explicit action input over repo config", () => {
    expect(resolveMaxDiffSize("12000", { maxDiffSize: 25000 })).toBe(12000);
  });
});

describe("resolveMaxComments", () => {
  it("uses repo config when action input is still the default", () => {
    expect(
      resolveMaxComments(String(DEFAULT_MAX_COMMENTS), { maxComments: 8 })
    ).toBe(8);
  });

  it("honors an explicit non-default action input over repo config", () => {
    expect(resolveMaxComments("5", { maxComments: 8 })).toBe(5);
  });

  it("honors max-comments 0 from repo config", () => {
    expect(resolveMaxComments(String(DEFAULT_MAX_COMMENTS), { maxComments: 0 })).toBe(0);
  });
});

describe("resolveJsonResponseMode", () => {
  it("prefers explicit action input, then repo config, then default true", () => {
    expect(resolveJsonResponseMode("false", { jsonResponseMode: true })).toBe(false);
    expect(resolveJsonResponseMode("true", { jsonResponseMode: false })).toBe(true);
    expect(resolveJsonResponseMode("", { jsonResponseMode: false })).toBe(false);
    expect(resolveJsonResponseMode("", undefined)).toBe(true);
  });
});

describe("resolveRequestChanges", () => {
  it("prefers explicit action input, then repo config, then default true", () => {
    expect(resolveRequestChanges("false", { requestChanges: true })).toBe(false);
    expect(resolveRequestChanges("true", { requestChanges: false })).toBe(true);
    expect(resolveRequestChanges("", { requestChanges: false })).toBe(false);
    expect(resolveRequestChanges("", undefined)).toBe(true);
  });
});

describe("resolveReasoningEffort", () => {
  it("prefers a non-empty action input over repo config and trims it", () => {
    expect(resolveReasoningEffort("  low ", { reasoningEffort: "high" })).toBe("low");
  });

  it("falls back to repo config when the input is empty or whitespace", () => {
    expect(resolveReasoningEffort("", { reasoningEffort: "high" })).toBe("high");
    expect(resolveReasoningEffort("   ", { reasoningEffort: "high" })).toBe("high");
  });

  it("stays unset when neither the input nor repo config sets it", () => {
    expect(resolveReasoningEffort("", undefined)).toBeUndefined();
    expect(resolveReasoningEffort("  ", {})).toBeUndefined();
  });
});
