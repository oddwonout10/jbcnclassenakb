"use client";

import { useEffect, useState } from "react";

type CircularItem = {
  id: string;
  title: string;
  displayTitle: string;
  publishedOn?: string | null;
  signedUrl?: string | null;
  isTimetable?: boolean;
};

function formatDate(value?: string | null) {
  if (!value) return null;
  try {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return null;
    return date.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return null;
  }
}

export function CircularsInfo() {
  const [items, setItems] = useState<CircularItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    async function loadCirculars() {
      try {
        setIsLoading(true);
        setLoadError(null);
        const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;
        if (!backendUrl) {
          throw new Error("Backend URL is not configured.");
        }

        const response = await fetch(`${backendUrl}/documents/recent`, {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        });

        if (!response.ok) {
          const detail = await response.text();
          throw new Error(detail || "Failed to load circulars.");
        }

        const data = (await response.json()) as Array<{
          document_id: string;
          title: string;
          display_title: string;
          published_on?: string | null;
          signed_url?: string | null;
          is_timetable?: boolean;
        }>;

        const mapped: CircularItem[] = (data ?? []).map((item) => ({
          id: item.document_id,
          title: item.title,
          displayTitle: item.display_title,
          publishedOn: item.published_on ?? null,
          signedUrl: item.signed_url ?? null,
          isTimetable: Boolean(item.is_timetable),
        }));

        setItems(mapped);
      } catch (error) {
        console.warn("Failed to load circulars:", error);
        setLoadError("Unable to load circulars right now.");
      } finally {
        setIsLoading(false);
      }
    }

    loadCirculars();
  }, []);

  return (
    <section className="flex h-full flex-col gap-4 rounded-3xl border border-[#dff7f0] bg-[#fbfffe] p-6 shadow-[0_15px_30px_rgba(67,192,246,0.2)]">
      <div className="flex items-center gap-3">
        <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#4cc6a8]/20 text-2xl">
          📚
        </div>
        <div>
          <h2 className="text-xl font-semibold text-[#2f3142]">
            Fresh from the Circular Shelf
          </h2>
          <p className="text-sm text-[#4e5d78]">
            Latest uploads for Grade 3 Ena, pulled straight from Supabase Storage.
          </p>
        </div>
      </div>

      {isLoading ? (
        <p className="text-sm text-[#4e5d78]">Loading circulars…</p>
      ) : loadError ? (
        <p className="text-sm text-[#d63f2f]">{loadError}</p>
      ) : items.length === 0 ? (
        <p className="text-sm text-[#4e5d78]">
          No circulars are available yet. Check back soon!
        </p>
      ) : (
        <div className="space-y-3 text-sm text-[#394155]">
          {items.map((item) => {
            const dateLabel = formatDate(item.publishedOn);
            return (
              <article
                key={item.id}
                className={`rounded-2xl bg-white/90 p-4 shadow-inner ${
                  item.isTimetable
                    ? "shadow-[#4cc6a8]/20 border border-[#4cc6a8]/40"
                    : "shadow-[#43c0f6]/15 border border-white/60"
                }`}
              >
                <div className="flex items-center justify-between gap-3">
                  <p className="text-base font-semibold text-[#21576f]">
                    {item.displayTitle}
                  </p>
                  {item.signedUrl ? (
                    <a
                      href={item.signedUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 rounded-full bg-[#43c0f6]/10 px-3 py-1 text-xs font-semibold text-[#1f5670] transition hover:bg-[#43c0f6]/20"
                    >
                      Download
                    </a>
                  ) : null}
                </div>
                {dateLabel ? (
                  <p className="mt-1 text-xs font-medium text-[#4e5d78]">
                    Published {dateLabel}
                  </p>
                ) : null}
                {!item.isTimetable ? (
                  <p className="mt-2 text-sm text-[#51607c]">
                    Tap download to open the PDF and review the latest update.
                  </p>
                ) : (
                  <p className="mt-2 text-sm text-[#51607c]">
                    Keep this handy for quick reference to daily schedules.
                  </p>
                )}
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
