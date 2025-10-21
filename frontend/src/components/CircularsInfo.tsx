export function CircularsInfo() {
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
            Check the WhatsApp spotlight for the newest uploads—everything here stays in sync.
          </p>
        </div>
      </div>
      <div className="grid gap-3 text-sm text-[#394155]">
        <div className="rounded-2xl bg-white/90 p-3 shadow-inner shadow-[#4cc6a8]/10">
          <p className="font-semibold text-[#21576f]">Weekly Highlights</p>
          <p>Look out for the class parent’s summary on Fridays with the top need-to-knows.</p>
        </div>
        <div className="rounded-2xl bg-white/90 p-3 shadow-inner shadow-[#ffb86c]/15">
          <p className="font-semibold text-[#ff6f61]">Ask &amp; Explore</p>
          <p>The assistant links every answer to its circular so you can double-check in one tap.</p>
        </div>
        <div className="rounded-2xl bg-white/90 p-3 shadow-inner shadow-[#43c0f6]/15">
          <p className="font-semibold text-[#2f7fa1]">New Uploads</p>
          <p>Admins drop new PDFs via the dashboard—once processed, they pop up in this feed automatically.</p>
        </div>
      </div>
    </section>
  );
}
