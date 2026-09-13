import { Case } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

export async function fetchPending(): Promise<Case[]> {
  const res = await fetch(`${API_BASE}/listings/pending`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch pending listings");
  return res.json();
}

export async function fetchResolved(): Promise<Case[]> {
  const res = await fetch(`${API_BASE}/listings/resolved`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch resolved listings");
  return res.json();
}

export async function decideCase(
  threadId: string,
  decision: "confirmed_fraud" | "false_positive"
): Promise<void> {
  const form = new FormData();
  form.append("decision", decision);
  const res = await fetch(`${API_BASE}/listings/${encodeURIComponent(threadId)}/decide`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error("Failed to record decision");
}

export function imageUrl(threadId: string): string {
  return `${API_BASE}/listings/${encodeURIComponent(threadId)}/image`;
}
