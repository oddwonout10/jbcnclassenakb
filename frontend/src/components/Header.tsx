import Image from "next/image";

export function Header() {
  return (
    <header className="w-full border-b border-slate-200 bg-white/70 backdrop-blur supports-[backdrop-filter]:bg-white/50">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <Image
            src="/logo.png"
            alt="JBCN School Ena"
            width={56}
            height={56}
            priority
            className="h-12 w-12 rounded-full object-contain"
          />
          <div>
            <p className="text-lg font-semibold text-slate-900">
              Grade 3 Ena – Class Assistant
            </p>
            <p className="text-sm text-slate-500">
              Ask a question or review the latest circulars and events.
            </p>
          </div>
        </div>
      </div>
    </header>
  );
}

