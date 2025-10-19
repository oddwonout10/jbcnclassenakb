export function CircularsInfo() {
  return (
    <section className="flex h-full flex-col rounded-xl border border-slate-200 bg-white p-6 shadow-lg shadow-slate-200/40">
      <h2 className="text-xl font-semibold text-slate-900">
        Latest Circulars & Resources
      </h2>
      <p className="mt-1 text-sm text-slate-500">
        The five most recent circulars for Grade 3 are highlighted in the class
        WhatsApp group. Uploads are also available in the Supabase dashboard
        (admin view) and will appear here as soon as guardian logins are
        enabled.
      </p>
      <ul className="mt-4 list-disc space-y-2 pl-5 text-sm text-slate-600">
        <li>Keep an eye on the class parent&apos;s weekly summary.</li>
        <li>
          Use the &ldquo;Ask the Class Assistant&rdquo; panel for quick answers
          linked to circulars.
        </li>
        <li>
          Admins can upload new PDFs via the FastAPI dashboard; the assistant
          picks them up automatically.
        </li>
      </ul>
    </section>
  );
}

