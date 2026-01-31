import type { AskRequest, AskResponse } from "./types";

export const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) ??
  "http://localhost:7860";
const NORMALISED_BASE = API_BASE.endsWith("/") ? API_BASE : `${API_BASE}/`;
const ASK_ENDPOINT = new URL("ask", NORMALISED_BASE).toString();

export async function askQuestion(payload: AskRequest): Promise<AskResponse> {
  const response = await fetch(ASK_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Backend responded with an error");
  }

  return response.json() as Promise<AskResponse>;
}

export function resolveViewerUrl(
  relativeUrl: string | null | undefined
): string | null {
  if (!relativeUrl) {
    return null;
  }
  try {
    return new URL(relativeUrl, NORMALISED_BASE).toString();
  } catch (error) {
    console.warn("Failed to resolve viewer URL", error);
    return relativeUrl;
  }
}
