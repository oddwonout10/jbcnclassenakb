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
  title: "JBCN Grade 3 Assistant",
  description: "Ask questions and review circular updates for Grade 3 Ena.",
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
