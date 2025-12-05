// src/memory/embedding/kmeans.ts
// A97: K-Means Clustering Utility
// Used for concept splitting

export function kMeans(vectors: number[][], k: number = 2, iterations: number = 6) {
  if (vectors.length === 0 || vectors.length < k) {
    return {
      centroids: [],
      clusters: []
    };
  }

  const dim = vectors[0].length;
  
  // Initialize centroids randomly
  let centroids: number[][] = [];
  for (let i = 0; i < k; i++) {
    const randomIdx = Math.floor(Math.random() * vectors.length);
    centroids.push([...vectors[randomIdx]]);
  }

  let clusters: number[] = [];

  for (let iter = 0; iter < iterations; iter++) {
    // Assign vectors to nearest centroid
    clusters = vectors.map(vec => {
      let minDist = Infinity;
      let nearest = 0;
      
      for (let i = 0; i < centroids.length; i++) {
        const dist = euclideanDistance(vec, centroids[i]);
        if (dist < minDist) {
          minDist = dist;
          nearest = i;
        }
      }
      
      return nearest;
    });

    // Update centroids
    for (let i = 0; i < k; i++) {
      const clusterVectors = vectors.filter((_, idx) => clusters[idx] === i);
      if (clusterVectors.length > 0) {
        centroids[i] = clusterVectors[0].map((_, dimIdx) => {
          const sum = clusterVectors.reduce((s, v) => s + v[dimIdx], 0);
          return sum / clusterVectors.length;
        });
      }
    }
  }

  return {
    centroids,
    clusters
  };
}

function euclideanDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.pow(a[i] - b[i], 2);
  }
  return Math.sqrt(sum);
}

