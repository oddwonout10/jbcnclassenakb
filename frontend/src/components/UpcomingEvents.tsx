"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabaseClient";
import { CalendarEvent } from "@/lib/types";

function formatDateRange(start: string, end: string | null) {
  const startDate = new Date(start);
  if (Number.isNaN(startDate.getTime())) return start;
  if (!end) return startDate.toLocaleDateString();

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

  useEffect(() => {
    async function loadEvents() {
      if (!supabase) return;
      const today = new Date().toISOString().split("T")[0];
      const { data, error: fetchError } = await supabase
        .from("calendar_events")
        .select("*")
        .gte("event_date", today)
        .order("event_date", { ascending: true })
        .limit(40);

      if (fetchError) {
        setError("Unable to load the calendar right now.");
        return;
      }

      const grouped = new Map<string, CalendarEvent[]>();
      (data ?? []).forEach((event) => {
        if (!grouped.has(event.event_date)) {
          grouped.set(event.event_date, []);
        }
        grouped.get(event.event_date)!.push(event);
      });

      const orderedDates = Array.from(grouped.keys()).sort();
      const MAX_ITEMS = 6;
      const selected: CalendarEvent[] = [];
      for (const date of orderedDates) {
        const dayEvents = grouped.get(date);
        if (!dayEvents) continue;
        selected.push(...dayEvents);
        if (selected.length >= MAX_ITEMS) {
          break;
        }
      }

      setEvents(selected);
    }

    loadEvents();
  }, []);

  return (
    <section className="flex h-full flex-col rounded-xl border border-slate-200 bg-white p-6 shadow-lg shadow-slate-200/40">
      <h2 className="text-xl font-semibold text-slate-900">Upcoming Events</h2>
      <p className="mt-1 text-sm text-slate-500">
        These dates are taken directly from the yearly calendar. Whole-school
        events are highlighted for quick reference.
      </p>

      <div className="mt-4 flex-1 space-y-3 overflow-auto">
        {error ? <p className="text-sm text-rose-600">{error}</p> : null}
        {!error && events.length === 0 ? (
          <p className="text-sm text-slate-400">No upcoming events recorded.</p>
        ) : null}

        {events.map((event) => {
          const dateRange = formatDateRange(
            event.event_date,
            event.end_date ?? null
          );
          const tag =
            event.audience?.length && event.audience[0]
              ? event.audience[0]
              : "general";

          return (
            <article
              key={`${event.event_date}-${event.title}`}
              className="rounded-lg border border-slate-200 bg-slate-50 p-3"
            >
              <p className="text-xs uppercase tracking-wide text-indigo-500">
                {tag.replaceAll("_", " ")}
              </p>
              <p className="mt-1 text-sm font-medium text-slate-900">
                {event.title}
              </p>
              <p className="text-xs text-slate-600">{dateRange}</p>
            </article>
          );
        })}
      </div>
    </section>
  );
}
