"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { QAResponse, SourceInfo } from "@/lib/types";

type TurnstileRenderOptions = {
  sitekey: string;
  callback: (token: string) => void;
  "expired-callback"?: () => void;
  "error-callback"?: () => void;
  theme?: "light" | "dark" | "auto";
};

declare global {
  interface Window {
    turnstile?: {
      render: (container: HTMLElement, options: TurnstileRenderOptions) => string;
      reset: (widgetId?: string) => void;
    };
  }
}

interface ConversationTurn {
  role: "user" | "assistant";
  text: string;
  sources?: SourceInfo[];
}

function SourceList({ sources }: { sources: SourceInfo[] }) {
  if (!sources?.length) return null;

  return (
    <div className="mt-3 rounded-2xl border border-[#b7e6f9] bg-[#e7f8ff] p-3 text-sm text-[#1f5670] shadow-inner shadow-white/40">
      <p className="font-semibold">Sources to explore</p>
      <ul className="mt-1 space-y-1">
        {sources.map((source, index) => (
          <li key={`${source.document_id}-${index}`} className="flex flex-col">
            <span className="font-semibold">{source.title}</span>
            <span className="text-xs text-[#31718d]">
              {source.published_on ? `Updated ${source.published_on}` : ""}
            </span>
            {source.signed_url ? (
              <a
                href={source.signed_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-[#ff6f61] underline hover:text-[#ff4b40]"
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
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const siteKey = process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY;
  const [captchaToken, setCaptchaToken] = useState<string | null>(null);
  const widgetRef = useRef<HTMLDivElement | null>(null);
  const widgetIdRef = useRef<string | null>(null);

  const suggestions = [
    "When is Diwali vacation?",
    "When is Nashik trip?",
    "Who is on the PTA?",
  ];

  useEffect(() => {
    if (!siteKey) {
      return;
    }

    const renderWidget = () => {
      if (!widgetRef.current || !window.turnstile) {
        return;
      }
      if (widgetIdRef.current) {
        window.turnstile.reset(widgetIdRef.current);
      }
      widgetIdRef.current = window.turnstile.render(widgetRef.current, {
        sitekey: siteKey,
        callback: (token: string) => setCaptchaToken(token),
        "error-callback": () => setCaptchaToken(null),
        "expired-callback": () => setCaptchaToken(null),
        theme: "light",
      });
    };

    if (typeof window !== "undefined" && window.turnstile) {
      renderWidget();
      return;
    }

    const scriptId = "turnstile-script";
    let script = document.getElementById(scriptId) as HTMLScriptElement | null;

    if (script) {
      script.addEventListener("load", renderWidget);
      return () => {
        script?.removeEventListener("load", renderWidget);
      };
    }

    script = document.createElement("script");
    script.id = scriptId;
    script.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
    script.async = true;
    script.defer = true;
    script.onload = renderWidget;
    document.head.appendChild(script);

    return () => {
      script?.removeEventListener("load", renderWidget);
    };
  }, [siteKey]);

  function handleSuggestion(text: string) {
    setQuestion(text);
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  }

  async function askAssistant(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) {
      setErrorMessage("Ask a question before submitting.");
      return;
    }
    if (siteKey && !captchaToken) {
      setErrorMessage("Please complete the quick verification above first.");
      return;
    }

    setErrorMessage(null);
    setIsLoading(true);

    setConversation((prev) => [...prev, { role: "user", text: trimmed }]);
    setQuestion("");

    const resetCaptcha = () => {
      if (!siteKey) return;
      setCaptchaToken(null);
      if (widgetIdRef.current && window.turnstile) {
        window.turnstile.reset(widgetIdRef.current);
      }
    };

    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;
      if (!backendUrl) {
        throw new Error("Backend URL is not configured.");
      }

      const payload: {
        question: string;
        captcha_token?: string;
      } = { question: trimmed };
      if (captchaToken) {
        payload.captcha_token = captchaToken;
      }

      const response = await fetch(`${backendUrl}/qa`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      let parsed: QAResponse | { detail?: string } | null = null;
      try {
        parsed = (await response.json()) as QAResponse | { detail?: string };
      } catch {
        parsed = null;
      }

      const maybeDetail = parsed && !("sources" in parsed) ? parsed.detail : undefined;

      if (!response.ok || !parsed || !("sources" in parsed)) {
        const detail =
          typeof maybeDetail === "string"
            ? maybeDetail
            : response.status === 503
              ? "The knowledge base is temporarily unavailable. The class parent has been notified."
              : response.status === 429
                ? "Too many questions back-to-back. Please wait a moment before trying again."
                : "The assistant could not process the question right now.";
        throw new Error(detail);
      }

      setConversation((prev) => [
        ...prev,
        {
          role: "assistant",
          text: parsed.answer,
          sources: parsed.sources,
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
      resetCaptcha();
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
    <section className="flex w-full flex-col gap-5 rounded-3xl border border-[#ffe4c4] bg-[#fffdf7] p-6 shadow-[0_18px_35px_rgba(255,175,109,0.25)]">
      <div className="flex flex-col gap-3 rounded-2xl bg-gradient-to-r from-[#fff0d4] via-white to-[#e8fffb] p-5 shadow-inner shadow-white/60 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-4">
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-white text-3xl shadow-inner shadow-black/10">
            🦉
          </div>
          <div>
            <h2 className="text-xl font-semibold text-[#2f3142] sm:text-2xl">
              Ask the Class Owl
            </h2>
            <p className="text-sm text-[#4e4f63]">
              We keep caregivers and curious learners in the loop with the newest circulars.
            </p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {suggestions.map((text) => (
            <button
              key={text}
              type="button"
              onClick={() => handleSuggestion(text)}
              className="rounded-full bg-white/80 px-3 py-1 text-xs font-semibold text-[#21576f] shadow-sm transition hover:bg-white/90 hover:shadow"
            >
              {text}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={askAssistant} className="flex flex-col gap-4">
        <textarea
          ref={textareaRef}
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="e.g. When does the Diwali break start? When do classes resume?"
          className="min-h-[120px] w-full rounded-3xl border border-[#ffd8ad] bg-white px-4 py-3 text-sm text-[#2f3142] shadow focus:border-[#ffb07a] focus:outline-none focus:ring-4 focus:ring-[#ffe2ca]"
        />
        {errorMessage ? (
          <p className="text-sm text-[#d63f2f]">{errorMessage}</p>
        ) : null}
        {siteKey ? (
          <div className="flex justify-end">
            <div
              ref={widgetRef}
              className="rounded-2xl border border-[#ffd8ad]/70 bg-white/80 px-2 py-2 shadow-inner shadow-white/40"
            />
          </div>
        ) : null}
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <button
            type="submit"
            disabled={isLoading}
            className="inline-flex items-center justify-center rounded-full bg-gradient-to-r from-[#ff6f61] to-[#ffb86c] px-6 py-2 text-sm font-semibold text-white shadow-lg shadow-[#ff9d7a]/40 transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-70"
          >
            {isLoading ? "Checking…" : "Send question"}
          </button>
          <button
            type="button"
            onClick={() => setConversation([])}
            disabled={isLoading || conversation.length === 0}
            className="text-sm font-semibold text-[#4e5d78] underline decoration-dotted underline-offset-4 hover:text-[#2f3142] disabled:text-[#c1c5d0]"
          >
            Clear conversation
          </button>
        </div>
      </form>

      {conversation.length > 0 ? (
        <p className="text-xs font-semibold uppercase tracking-wide text-[#a3adb6]">
          Most recent response appears first
        </p>
      ) : null}
      <div className="mt-2 space-y-4">
        {isLoading ? (
          <div
            className="flex items-center gap-3 rounded-3xl border border-[#b7e6f9] bg-[#f3fbff] p-4 text-sm text-[#1f5670] shadow-inner"
            aria-live="polite"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white text-2xl shadow">
              🦉
            </div>
            <div className="flex flex-col gap-1">
              <p className="font-semibold">Checking the latest circulars…</p>
              <div className="flex items-center gap-1">
                <span
                  className="h-2 w-2 rounded-full bg-[#ff6f61] animate-bounce"
                  style={{ animationDelay: "0ms" }}
                />
                <span
                  className="h-2 w-2 rounded-full bg-[#f9c846] animate-bounce"
                  style={{ animationDelay: "150ms" }}
                />
                <span
                  className="h-2 w-2 rounded-full bg-[#43c0f6] animate-bounce"
                  style={{ animationDelay: "300ms" }}
                />
              </div>
            </div>
          </div>
        ) : null}
        {conversationGroups.map((group, groupIndex) => (
          <div key={`group-${groupIndex}`} className="space-y-3">
            {group.map((turn, turnIndex) => (
              <div
                key={`turn-${groupIndex}-${turnIndex}-${turn.role}`}
                className={`relative rounded-3xl border p-4 text-sm shadow-sm transition ${
                  turn.role === "user"
                    ? "border-[#ffd8ad] bg-white/90 text-[#2f3142]"
                    : "border-[#b7e6f9] bg-[#f3fbff] text-[#1f5670]"
                }`}
              >
                <span className="absolute -top-3 left-4 flex items-center gap-1 rounded-full bg-white px-3 py-1 text-xs font-semibold text-[#4e5d78] shadow">
                  {turn.role === "user" ? "👩‍👧 You asked" : "🦉 Owl replied"}
                </span>
                <p className="mt-3 whitespace-pre-wrap text-base leading-relaxed">
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
