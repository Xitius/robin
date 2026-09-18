export type ReasoningFallbackReason = "unsupported" | "invalid-value";

export function buildReasoningFallbackNotice(
  reason?: ReasoningFallbackReason
): string | undefined {
  if (!reason) return undefined;
  const rejection = reason === "invalid-value" ? "rejected as invalid" : "rejected as unsupported";
  return (
    `:warning: The configured \`reasoning-effort\` was ${rejection}. ` +
    "Robin completed this run without a reasoning override. Update `.github/robin.yml` " +
    "or the workflow `with: reasoning-effort` value."
  );
}
