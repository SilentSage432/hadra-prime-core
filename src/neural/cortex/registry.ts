// src/neural/cortex/registry.ts
// A91: Neural Cortex Registration Table
// A92: Added Embedding Model Slot #1
// Defines available neural modules for PRIME's cortex

export const NeuralRegistry = [
  {
    name: "embedding_model",
    path: "../../models/embedding_model.ts",
    enabled: true
  },
  {
    name: "example_model",
    path: "../../models/example_model.ts",
    enabled: false
  }
];

