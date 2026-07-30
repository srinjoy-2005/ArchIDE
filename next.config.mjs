import { fileURLToPath } from 'url';

// Resolve the directory of this config file (the actual project root)
// This fixes Next.js 16 Turbopack mis-detecting the workspace root
// when a parent package-lock.json exists at /Users/sampad/dev/
const projectRoot = fileURLToPath(new URL('.', import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  turbopack: {
    root: projectRoot,
  },
  allowedDevOrigins: ['172.28.18.2'],
};

export default nextConfig;
