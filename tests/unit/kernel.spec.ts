import { describe, it, expect } from "vitest";
import "../../src/kernel/index.js"; // triggers boot
import { getStatus, processCommand } from "../../src/prime.js";

describe("HADRA-PRIME Kernel", () => {
  it("boots without errors", () => {
    expect(true).toBe(true);
  });

  it("exposes PRIME status API", () => {
    const status = getStatus();
    expect(status).toHaveProperty("state");
    expect(status).toHaveProperty("cognitiveLoad");
    expect(status.state).toBe("online");
  });

  it("processes commands", async () => {
    const result = await processCommand("status");
    expect(result.type).toBe("status");
    expect(result.data).toHaveProperty("state");
  });
});

