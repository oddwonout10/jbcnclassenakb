/* eslint-disable react-hooks/exhaustive-deps */
"use client";

import { FormEvent, useMemo, useState } from "react";
import { QAResponse, SourceInfo } from "@/lib/types";

interface ConversationTurn {
  role: "user" | "assistant";
  text: string;
  sources?: SourceInfo[];
}

function SourceList({ sources }: { sources: SourceInfo[] }) {
  if (!sources?.length) return null;

  return (
    <div className="mt-3 rounded-md border border-indigo-100 bg-indigo-50 p-3 text-sm text-indigo-900">
      <p className="font-medium">Sources</p>
      <ul className="mt-1 space-y-1">
        {sources.map((source, index) => (
          <li key={`${source.document_id}-${index}`} className="flex flex-col">
            <span className="font-medium">{source.title}</span>
            <span className="text-xs text-indigo-700">
              {source.published_on ? `Updated ${source.published_on}` : ""}
            </span>
            {source.signed_url ? (
              <a
                href={source.signed_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-indigo-600 underline hover:text-indigo-700"
              >
                View document
              </a>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function AssistantPanel() {
  const [question, setQuestion] = useState("");
  const [conversation, setConversation] = useState<ConversationTurn[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function askAssistant(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) {
      setErrorMessage("Ask a question before submitting.");
      return;
    }

    setErrorMessage(null);
    setIsLoading(true);

    setConversation((prev) => [...prev, { role: "user", text: trimmed }]);
    setQuestion("");

    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;
      if (!backendUrl) {
        throw new Error("Backend URL is not configured.");
      }

      const response = await fetch(`${backendUrl}/qa`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: trimmed }),
      });

      if (!response.ok) {
        throw new Error(
          response.status === 503
            ? "The knowledge base is temporarily unavailable. The class parent has been notified."
            : "The assistant could not process the question right now."
        );
      }

      const data: QAResponse = await response.json();
      setConversation((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.answer,
          sources: data.sources,
        },
      ]);
    } catch (error) {
      setConversation((prev) => [
        ...prev,
        {
          role: "assistant",
          text:
            error instanceof Error
              ? error.message
              : "Something went wrong while contacting the assistant.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  const conversationGroups = useMemo(() => {
    const groups: ConversationTurn[][] = [];
    let index = conversation.length - 1;

    while (index >= 0) {
      const current = conversation[index];
      const previous = conversation[index - 1];

      if (
        current?.role === "assistant" &&
        previous?.role === "user"
      ) {
        groups.push([previous, current]);
        index -= 2;
      } else {
        groups.push([current]);
        index -= 1;
      }
    }

    return groups;
  }, [conversation]);

  return (
    <section className="flex w-full flex-col gap-4 rounded-xl border border-slate-200 bg-white p-6 shadow-lg shadow-slate-200/40">
      <h2 className="text-xl font-semibold text-slate-900">
        Ask the Class Assistant
      </h2>
      <p className="text-sm text-slate-500">
        Ask anything about schedules, breaks, events or circulars. When the
        assistant cannot find an answer it will automatically alert the class
        parent.
      </p>

      <form onSubmit={askAssistant} className="flex flex-col gap-3">
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="e.g. When does the Diwali break start? When do classes resume?"
          className="min-h-[96px] w-full rounded-lg border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
        />
        {errorMessage ? (
          <p className="text-sm text-rose-600">{errorMessage}</p>
        ) : null}
        <div className="flex items-center justify-between gap-3">
          <button
            type="submit"
            disabled={isLoading}
            className="inline-flex items-center justify-center rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-indigo-300"
          >
            {isLoading ? "Checking…" : "Ask"}
          </button>
          <button
            type="button"
            onClick={() => setConversation([])}
            disabled={isLoading || conversation.length === 0}
            className="text-sm font-medium text-slate-500 hover:text-slate-700 disabled:text-slate-300"
          >
            Clear conversation
          </button>
        </div>
      </form>

      {conversation.length > 0 ? (
        <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
          Most recent response at the top
        </p>
      ) : null}
      <div className="mt-2 space-y-4">
        {conversationGroups.map((group, groupIndex) => (
          <div key={`group-${groupIndex}`} className="space-y-3">
            {group.map((turn, turnIndex) => (
              <div
                key={`turn-${groupIndex}-${turnIndex}-${turn.role}`}
                className={`rounded-lg border p-3 text-sm ${
                  turn.role === "user"
                    ? "border-slate-200 bg-slate-50"
                    : "border-indigo-100 bg-indigo-50"
                }`}
              >
                <p className="font-medium text-slate-600">
                  {turn.role === "user" ? "You" : "Assistant"}
                </p>
                <p className="mt-1 whitespace-pre-wrap text-slate-800">
                  {turn.text}
                </p>
                {turn.role === "assistant" ? (
                  <SourceList sources={turn.sources ?? []} />
                ) : null}
              </div>
            ))}
          </div>
        ))}
      </div>
    </section>
  );
}
