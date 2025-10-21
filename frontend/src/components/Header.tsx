import Image from "next/image";

export function Header() {
  return (
    <header className="relative w-full bg-gradient-to-r from-[#43c0f6] via-[#73d6f5] to-[#f9c846] text-white shadow-[0_25px_50px_-12px_rgba(36,91,151,0.4)]">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute left-8 top-6 h-12 w-12 rounded-full bg-white/30 blur-xl" />
        <div className="absolute bottom-0 right-16 h-16 w-16 rounded-full bg-white/20 blur-2xl" />
      </div>
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-8 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:py-10">
        <div className="space-y-3 text-balance">
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2 rounded-full bg-white/70 px-3 py-1 text-xs font-semibold text-[#21576f] shadow-sm">
              <Image
                src="/logo.png"
                alt="JBCN Grade 3 Ena crest"
                width={28}
                height={28}
                className="h-7 w-7 rounded-full object-contain shadow-sm"
                priority
              />
              <span>JBCN Grade 3 Ena</span>
            </div>
            <p className="inline-flex items-center gap-2 rounded-full bg-white/30 px-4 py-1 text-sm font-semibold text-[#21576f] shadow-sm">
              <span role="img" aria-hidden="true">
                🌈
              </span>
              Class companion crafted by Thea&apos;s dad, Shreerang Sunkersett
            </p>
          </div>
          <h1 className="text-3xl font-semibold sm:text-4xl lg:text-5xl">
            Hello, Super Learners & Families!
          </h1>
          <p className="max-w-xl text-base sm:text-lg">
            Explore circulars i.e. Announcements on Toddle, discover exciting events, and ask the class owl anything. We keep the latest updates right at your fingertips.
          </p>
        </div>
        <div className="flex w-full max-w-sm items-center gap-4 rounded-3xl bg-white/80 p-4 text-[#21576f] shadow-lg backdrop-blur-sm lg:w-auto">
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-[#f9c846]/80 text-2xl shadow">
            🦉
          </div>
          <p className="text-sm font-medium">
            “Ask me anything about school life and I will find the latest answer for you!”
          </p>
        </div>
      </div>
    </header>
  );
}
