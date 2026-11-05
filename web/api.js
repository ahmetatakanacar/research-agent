const KEY_STORAGE = "ra-api-key";

let apiKey = "";
try { apiKey = sessionStorage.getItem(KEY_STORAGE) || ""; } catch { /* storage unavailable */ }

let onUnauthorized = null;

export function setApiKey(key) {
  apiKey = key;
  try { sessionStorage.setItem(KEY_STORAGE, key); } catch { /* storage unavailable */ }
}

export function hasApiKey() { return Boolean(apiKey); }

export function handleUnauthorized(callback) { onUnauthorized = callback; }

export class ApiError extends Error {
  constructor(message, status, retryAfter) {
    super(message);
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

function headers(extra = {}) {
  return apiKey ? { "X-API-Key": apiKey, ...extra } : extra;
}

async function errorFrom(response) {
  let message = `Server error (${response.status})`;
  try {
    const { detail } = await response.json();
    if (typeof detail === "string") message = detail;
    else if (Array.isArray(detail) && detail[0]?.msg) message = detail[0].msg;
  } catch { /* body is not JSON */ }
  return new ApiError(message, response.status, response.headers.get("Retry-After"));
}

async function send(path, { method = "GET", body, signal } = {}, retried = false) {
  const init = { method, signal, headers: headers() };
  if (body !== undefined) {
    init.headers = headers({ "Content-Type": "application/json" });
    init.body = JSON.stringify(body);
  }

  let response;
  try {
    response = await fetch(path, init);
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError("network", 0);
  }

  if (response.status === 401 && onUnauthorized && !retried && await onUnauthorized()) {
    return send(path, { method, body, signal }, true);
  }
  if (!response.ok) throw await errorFrom(response);
  return response;
}

export async function api(path, options) {
  const response = await send(path, options, Boolean(options?.noRetry));
  return response.status === 204 ? null : response.json();
}

export async function streamEvents(path, { after = 0, signal, onEvent }) {
  const response = await send(`${path}?after=${after}`, { signal });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);

      let event = "message";
      let id = null;
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("id: ")) id = Number(line.slice(4));
        else if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (!data) continue;
      onEvent(event, JSON.parse(data), id);
    }
  }
}
