export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export type DocumentRecord = {
  id: string;
  title: string;
  filename: string;
  source_url?: string | null;
  file_type: string;
  created_at: string;
  chunk_count: number;
};

export type Citation = {
  source_id: string;
  document_id: string;
  document_title: string;
  filename: string;
  source_url?: string | null;
  page?: number | null;
  chunk_index: number;
  snippet: string;
  score: number;
};

export type RetrievedChunk = Citation & {
  chunk_id: string;
  text: string;
};

export type AskResponse = {
  answer: string;
  supported: boolean;
  citations: Citation[];
  retrieved_chunks: RetrievedChunk[];
};

export async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set("Content-Type", "application/json");

  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.detail || `Request failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}
