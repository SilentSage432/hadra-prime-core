// src/prediction/predict_engine.ts

export function predictEngine(state: any) {
  if (state.intent === null) {
    return {
      ...state,
      prediction: {
        horizon: "idle",
        likelyNextIntent: null,
        recursionRisk: 0,
      }
    };
  }

  return {
    ...state,
    prediction: {
      horizon: "short",
      likelyNextIntent: state.intent,
      recursionRisk: 0
    }
  };
}

