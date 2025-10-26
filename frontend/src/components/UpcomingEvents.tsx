"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabaseClient";
import { CalendarEvent } from "@/lib/types";

function formatDateRange(start: string, end: string | null) {
  const startDate = new Date(start);
  if (Number.isNaN(startDate.getTime())) return start;
  if (!end) {
    return startDate.toLocaleDateString(undefined, {
      weekday: "short",
      month: "short",
      day: "numeric",
    });
  }

  const endDate = new Date(end);
  if (Number.isNaN(endDate.getTime())) return startDate.toLocaleDateString();

  const sameMonth =
    startDate.getMonth() === endDate.getMonth() &&
    startDate.getFullYear() === endDate.getFullYear();

  const startLabel = startDate.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
  const endLabel = endDate.toLocaleDateString(undefined, {
    month: sameMonth ? undefined : "short",
    day: "numeric",
  });
  return `${startLabel} – ${endLabel}`;
}

export function UpcomingEvents() {
  const [events, setEvents] = useState<CalendarEvent[]>([]);
  const [error, setError] = useState<string | null>(null);

  const allowedAudience = new Set<string>([
    "whole_school",
    "whole_school_holiday",
    "primary",
    "primary_secondary",
    "general",
  ]);
  const audiencePriority: Record<string, number> = {
    primary: 0,
    primary_secondary: 1,
    whole_school: 2,
    whole_school_holiday: 3,
    general: 4,
    secondary: 5,
    pre_primary: 6,
  };

  function scoreEventAudience(audience: string[] | null | undefined): number {
    if (!audience || audience.length === 0) {
      return audiencePriority.general;
    }
    let best = audiencePriority.general;
    for (const entry of audience) {
      const normalized = entry.toLowerCase();
      if (normalized in audiencePriority) {
        best = Math.min(best, audiencePriority[normalized]);
      }
    }
    return best;
  }

  useEffect(() => {
    async function loadEvents() {
      if (!supabase) return;
      const today = new Date().toISOString().split("T")[0];
      const { data, error: fetchError } = await supabase
        .from("calendar_events")
        .select("id,title,event_date,end_date,audience,description,source")
        .gte("event_date", today)
        .order("event_date", { ascending: true })
        .limit(40);

      if (fetchError) {
        setError("Unable to load the calendar right now.");
        return;
      }

      const relevant = (data ?? []).filter((event) => {
        const rawAudience = Array.isArray(event.audience)
          ? event.audience
          : [];
        const tags = rawAudience;
        if (tags.length === 0) {
          return true;
        }
        return tags.some((entry: string) => {
          const normalized = entry.toLowerCase();
          return allowedAudience.has(normalized);
        });
      });

      const grouped = new Map<string, CalendarEvent[]>();
      relevant.forEach((event) => {
        const key = `${event.event_date}:${event.title.toLowerCase()}`;
        if (!grouped.has(key)) {
          grouped.set(key, []);
        }
        grouped.get(key)!.push(event);
      });

      const orderedKeys = Array.from(grouped.keys()).sort((a, b) => {
        const [dateA] = a.split(":");
        const [dateB] = b.split(":");
        return dateA.localeCompare(dateB);
      });
      const MAX_ITEMS = 6;
      const selected: CalendarEvent[] = [];
      for (const key of orderedKeys) {
        const dayEvents = grouped.get(key);
        if (!dayEvents) continue;
        const sorted = [...dayEvents].sort((a, b) => {
          return scoreEventAudience(a.audience) - scoreEventAudience(b.audience);
        });
        const primaryEvent = sorted[0];
        selected.push(primaryEvent);
        if (selected.length >= MAX_ITEMS) {
          break;
        }
      }

      setEvents(selected);
    }

    loadEvents();
  }, []);

  return (
    <section className="flex h-full flex-col gap-4 rounded-3xl border border-[#dcecff] bg-[#f5faff] p-6 shadow-[0_18px_28px_rgba(51,132,223,0.18)]">
      <div className="flex items-center gap-3">
        <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#43c0f6]/20 text-2xl">
          📅
        </div>
        <div>
          <h2 className="text-xl font-semibold text-[#2f3142]">
            Save the Date!
          </h2>
          <p className="text-sm text-[#51607c]">
            Straight from the official calendar—perfect for planning backpacks and celebrations.
          </p>
        </div>
      </div>

      <div className="mt-4 flex-1 space-y-3 overflow-auto">
        {error ? <p className="text-sm text-[#d63f2f]">{error}</p> : null}
        {!error && events.length === 0 ? (
          <p className="text-sm text-[#94a3b8]">No upcoming events recorded.</p>
        ) : null}

        {events.map((event) => {
          const dateRange = formatDateRange(
            event.event_date,
            event.end_date ?? null
          );
          const audiences = Array.isArray(event.audience)
            ? event.audience
            : [];
          const tagRaw =
            audiences.find((entry) =>
              allowedAudience.has(entry.toLowerCase())
            ) ?? "general";
          const tag = tagRaw.toLowerCase();

          const badgeMap: Record<string, string> = {
            whole_school: "bg-[#ff6f61]/20 text-[#c74335]",
            whole_school_holiday: "bg-[#f9c846]/25 text-[#a46b00]",
            holiday: "bg-[#f9c846]/25 text-[#a46b00]",
            primary: "bg-[#43c0f6]/20 text-[#1f5670]",
            secondary: "bg-[#4cc6a8]/25 text-[#256e5c]",
            primary_secondary: "bg-[#b387fa]/20 text-[#6048a5]",
            pre_primary: "bg-[#9fd5ff]/30 text-[#27638f]",
            general: "bg-[#cde6ff]/40 text-[#2f3142]",
          };

          const badgeStyle = badgeMap[tag] ?? badgeMap.general;
          const tagLabel = tag.replaceAll("_", " ");

          return (
            <article
              key={`${event.event_date}-${event.title}`}
              className="relative overflow-hidden rounded-3xl border border-white/60 bg-white/90 p-4 shadow-[0_12px_22px_rgba(47,82,120,0.12)]"
            >
              <span
                className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold capitalize ${badgeStyle}`}
              >
                {tagLabel}
              </span>
              <p className="mt-2 text-base font-semibold text-[#2f3142]">
                {event.title}
              </p>
              {event.description &&
              event.description.trim().toLowerCase() !==
                event.title.trim().toLowerCase() ? (
                <p className="mt-1 text-sm text-[#51607c]">
                  {event.description}
                </p>
              ) : null}
              <p className="text-xs font-medium text-[#4e5d78]">
                {dateRange}
              </p>
            </article>
          );
        })}
      </div>
    </section>
  );
}
