// src/shared/hash.ts

import crypto from "crypto";

export function cryptoHash(data: string) {
  return crypto
    .createHash("sha256")
    .update(data)
    .digest("hex");
}

