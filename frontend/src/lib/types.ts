export interface SourceInfo {
  document_id: string;
  title: string;
  published_on: string | null;
  original_filename: string;
  signed_url: string | null;
  storage_path: string;
  similarity: number;
}

export interface QAResponse {
  status: "answered" | "escalated" | string;
  answer: string;
  sources: SourceInfo[];
}

export interface CalendarEvent {
  id?: string;
  title: string;
  event_date: string;
  end_date: string | null;
  audience: string[];
  source?: string | null;
  summary?: string;
}

