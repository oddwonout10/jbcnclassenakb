import { AssistantPanel } from "@/components/AssistantPanel";
import { CircularsInfo } from "@/components/CircularsInfo";
import { UpcomingEvents } from "@/components/UpcomingEvents";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col gap-6 bg-slate-100 px-4 py-6 sm:px-6 lg:px-8">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6">
        <section className="rounded-xl border border-slate-200 bg-gradient-to-r from-indigo-600 via-indigo-500 to-indigo-400 p-6 text-white shadow-lg">
          <h1 className="text-xl font-semibold sm:text-2xl">
            Welcome, Grade 3 Ena Families!
          </h1>
          <p className="mt-2 text-sm text-indigo-50 sm:text-base">
            Use the assistant to get answers from recent circulars and the
            school calendar. Scroll for upcoming events and quick reminders.
          </p>
        </section>

        <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
          <div className="flex flex-col gap-6">
            <AssistantPanel />
          </div>
          <div className="flex flex-col gap-6">
            <UpcomingEvents />
            <CircularsInfo />
          </div>
        </div>
      </div>
    </main>
  );
}
