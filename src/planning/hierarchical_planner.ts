// src/planning/hierarchical_planner.ts

import { StrategicPlanner } from "./levels/strategic.ts";
import { OperationalPlanner } from "./levels/operational.ts";
import { TacticalPlanner } from "./levels/tactical.ts";
import { AtomicPlanner } from "./levels/atomic.ts";

export class HierarchicalPlanner {
  strategic: StrategicPlanner;
  operational: OperationalPlanner;
  tactical: TacticalPlanner;
  atomic: AtomicPlanner;

  constructor() {
    this.strategic = new StrategicPlanner();
    this.operational = new OperationalPlanner();
    this.tactical = new TacticalPlanner();
    this.atomic = new AtomicPlanner();
  }

  buildPlan(intent: string, context: any) {
    const strategicGoal = this.strategic.formulate(intent, context);
    const operationalPlan = this.operational.expand(strategicGoal, context);
    const tacticalSteps = this.tactical.resolve(operationalPlan, context);
    const atomicActions = this.atomic.decompose(tacticalSteps);

    return {
      intent,
      strategicGoal,
      operationalPlan,
      tacticalSteps,
      atomicActions,
      confidence: this.strategic.confidence,
      requiresPermission: this.atomic.requiresPermission,
      requiresYubiKey: this.atomic.requiresYubiKey
    };
  }
}

