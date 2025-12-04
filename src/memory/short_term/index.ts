let buffer: any[] = [];

export function remember(evt: any) {
  buffer.push(evt);
  if (buffer.length > 50) buffer.shift();
}

export function recall() {
  return buffer;
}

