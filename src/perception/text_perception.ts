// src/perception/text_perception.ts
// A92: Text Perception with Embedding Support
// A93: Neural Memory Encoding Integration
// A95: Concept Formation Integration
// Connects PRIME's perception layer to neural embeddings

import { getEmbeddingAdapter } from "../kernel/index.ts";
import { getNeuralMemory } from "../kernel/index.ts";
import { fuse } from "../memory/fusion/episodic_neural_fusion.ts";

export async function perceiveText(text: string) {
  const embeddingAdapter = getEmbeddingAdapter();
  const neuralMem = getNeuralMemory();
  
  const embedding = embeddingAdapter 
    ? await embeddingAdapter.embedText(text)
    : { embedding: [], meta: { error: "adapter_offline" }};

  // A93: Store embedding in neural memory
  if (neuralMem && embedding.embedding.length > 0) {
    neuralMem.saveEmbedding(
      `text-${Date.now()}`,
      embedding.embedding,
      ["text", "raw_input"]
    );
  }

  // A95: Create fused memory with concept formation
  let fusedMemory = null;
  if (embedding.embedding.length > 0) {
    fusedMemory = fuse(
      {
        type: "text_observation",
        raw: text
      },
      {
        embedding: embedding.embedding,
        meaningVector: embedding.embedding // Use embedding as meaning vector for now
      }
    );
  }

  return {
    raw: text,
    embedding: embedding.embedding,
    meta: embedding.meta,
    concept: fusedMemory?.concept || null
  };
}

