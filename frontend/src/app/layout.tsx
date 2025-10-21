import type { Metadata } from "next";
import { Baloo_2, Nunito } from "next/font/google";
import "./globals.css";
import { Header } from "@/components/Header";

const playfulHeading = Baloo_2({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const friendlyBody = Nunito({
  variable: "--font-body",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_URL ?? "https://jbcnena.vercel.app"
  ),
  title: "JBCN Grade 3 Assistant",
  description: "Ask questions and review circular updates for Grade 3 Ena.",
  icons: {
    icon: "/logo.png",
    shortcut: "/logo.png",
    apple: "/logo.png",
  },
  openGraph: {
    title: "JBCN Grade 3 Assistant",
    description: "Ask questions and review circular updates for Grade 3 Ena.",
    url: "/",
    siteName: "JBCN Ena Assistant",
    images: [
      {
        url: "/logo.png",
        width: 512,
        height: 512,
        alt: "JBCN Ena crest",
      },
    ],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "JBCN Grade 3 Assistant",
    description: "Ask questions and review circular updates for Grade 3 Ena.",
    images: ["/logo.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${playfulHeading.variable} ${friendlyBody.variable} app-body antialiased`}
      >
        <Header />
        <main className="flex min-h-screen flex-col">{children}</main>
      </body>
    </html>
  );
}
