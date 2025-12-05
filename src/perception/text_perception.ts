// src/perception/text_perception.ts
// A92: Text Perception with Embedding Support
// Connects PRIME's perception layer to neural embeddings

import { getEmbeddingAdapter } from "../kernel/index.ts";

export async function perceiveText(text: string) {
  const embeddingAdapter = getEmbeddingAdapter();
  
  const embedding = embeddingAdapter 
    ? await embeddingAdapter.embedText(text)
    : { embedding: [], meta: { error: "adapter_offline" }};

  return {
    raw: text,
    embedding: embedding.embedding,
    meta: embedding.meta
  };
}

