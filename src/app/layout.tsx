import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ArchiDE - Visual ML IDE",
  description: "Visual node editor for PyTorch models",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
