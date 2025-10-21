"use client";

export function AboutContact() {
  return (
    <section className="mx-auto w-full max-w-6xl rounded-3xl border border-[#e6dff7] bg-gradient-to-r from-[#f6f0ff] via-white to-[#e8f9ff] p-6 shadow-[0_18px_32px_rgba(123,104,238,0.12)]">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div className="max-w-2xl space-y-2">
          <p className="inline-flex items-center gap-2 rounded-full bg-white/70 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-[#5b4ab4] shadow-sm">
            <span role="img" aria-hidden="true">
              💡
            </span>
            About this assistant
          </p>
          <h2 className="text-2xl font-semibold text-[#2f3142]">
            Built on official Toddle announcements
          </h2>
          <p className="text-sm text-[#4e5d78]">
            Every answer and download you see here comes from the announcements posted
            in the Toddle app. Circulars are ingested as soon as they are uploaded on a
            best-effort basis, so expect a delay when I&apos;m super busy. This site is for
            convenience—always refer to the Toddle app for the latest information.
          </p>
          <p className="text-sm text-[#4e5d78]">
            If you spot a bug, can&apos;t find an update, or have ideas to make this
            portal better, please ping Shreerang on WhatsApp. Quick feedback helps the
            class parent keep the knowledge base accurate for everyone.
          </p>
        </div>
      </div>
    </section>
  );
}
