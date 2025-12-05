// A109 — Strategic Autonomy Types
// Types for strategic reasoning and outcome evaluation

export interface StrategicScenario {
  goal: string;
  subgoal: string;
  evidence: string[];
}

export interface StrategicOutcome extends StrategicScenario {
  score: number;
}

