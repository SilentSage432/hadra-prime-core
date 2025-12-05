// src/perception/text_perception.ts
// A92: Text Perception with Embedding Support
// A93: Neural Memory Encoding Integration
// Connects PRIME's perception layer to neural embeddings

import { getEmbeddingAdapter } from "../kernel/index.ts";
import { getNeuralMemory } from "../kernel/index.ts";

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

  return {
    raw: text,
    embedding: embedding.embedding,
    meta: embedding.meta
  };
}

