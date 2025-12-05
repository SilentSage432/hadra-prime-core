// src/shared/types.ts
// Shared type definitions for PRIME core

// --- Neural Context Contracts ---
export interface CognitiveState {
  activeGoal?: { type: string };
  confidence?: number;
  uncertainty?: number;
  lastReflection?: { reason: string; pressure: number };
  recall?: {
    intuition: number;
    reference: any | null;
  };
  intentModifiers?: Array<{
    type: string;
    weight: number;
    note?: string;
  }>;
  safetyFlags?: string[];
}

export interface MotivationState {
  urgency: number;
  curiosity: number;
  claritySeeking: number;
  consolidation: number;
  goalBias: number;
  stabilityPressure: number;
  direction: string | null;
}

