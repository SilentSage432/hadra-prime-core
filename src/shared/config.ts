// src/shared/config.ts

export const PRIMEConfig = {
  cognition: {
    idlePollRate: 600,             // how often PRIME checks for stimulus
    baseCooldownMs: 1500,          // base cooldown after recursion
    cooldownMs: 1500,              // dynamically adjusted
  },
  runtime: {
    hasStimulus: false,            // gateway toggles this when a message arrives
  },
};

