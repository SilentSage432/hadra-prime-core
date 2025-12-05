// src/shared/symbolic_packet.ts
// A101: Symbolic Packet Interface
// A102: Attention & Relevance Estimator Integration
// Defines the structure for symbolic data that can be encoded/decoded through neural bridge

export interface SymbolicPacket {
  type: string;
  payload: any;
  neural_enhancement?: any;  // A101: Neural enhancement from neural bridge
  attention?: any;  // A102: Attention scores from attention estimator
}

