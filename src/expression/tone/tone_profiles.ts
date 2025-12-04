// src/expression/tone/tone_profiles.ts

export interface OperatorProfile {
  prefersDirectness: boolean;
  prefersWarmth: boolean;
  learningMode: boolean;
}

export const DefaultOperatorProfile: OperatorProfile = {
  prefersDirectness: true,
  prefersWarmth: true,
  learningMode: false,
};

// Later phases will let PRIME build this profile automatically from interactions.

