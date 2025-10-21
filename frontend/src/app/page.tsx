import { AssistantPanel } from "@/components/AssistantPanel";
import { CircularsInfo } from "@/components/CircularsInfo";
import { UpcomingEvents } from "@/components/UpcomingEvents";
import { AboutContact } from "@/components/AboutContact";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col gap-8 px-4 py-8 sm:px-6 lg:px-8">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8">
        <section className="rounded-3xl border border-white/60 bg-white/80 p-6 shadow-[0_20px_40px_rgba(255,176,122,0.18)] backdrop-blur">
          <h2 className="text-2xl font-semibold text-[#2f3142] sm:text-3xl">
            Let’s make school updates fun &amp; easy!
          </h2>
          <p className="mt-1 text-sm text-[#4e5d78] sm:text-base">
            Follow these quick steps to stay in the loop together as a family.
          </p>
          <div className="mt-6 grid gap-4 sm:grid-cols-3">
            {[
              {
                icon: "💬",
                title: "Ask the Owl",
                description:
                  "Type a question and get the latest answer pulled straight from official circulars.",
              },
              {
                icon: "🎉",
                title: "Check Upcoming Fun",
                description:
                  "See holidays, celebrations, and activities so backpacks are ready on time.",
              },
              {
                icon: "📎",
                title: "Review Highlights",
                description:
                  "Skim the quick reminders or jump into the PDFs if you want all the details.",
              },
            ].map((card) => (
              <div
                key={card.title}
                className="flex flex-col gap-2 rounded-2xl bg-[#fff9f2] p-4 shadow-inner shadow-white/70"
              >
                <span className="text-3xl">{card.icon}</span>
                <p className="text-base font-semibold text-[#2f3142]">
                  {card.title}
                </p>
                <p className="text-sm text-[#4e5d78]">{card.description}</p>
              </div>
            ))}
          </div>
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
        <AboutContact />
      </div>
    </main>
  );
}
