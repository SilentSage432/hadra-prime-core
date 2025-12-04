// ESM requires explicit .ts extension
import PRIME, { getStatus, processCommand, subscribeLogs } from "./prime.ts";

// Export default PRIME instance
export default PRIME;

// Export named functions for gateway compatibility
export { getStatus, processCommand, subscribeLogs };

// Export expression types
export * from "./expression/types.ts";

