import dotenv from 'dotenv';
dotenv.config({ path: './.env' });
/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    images: {
      remotePatterns: [{hostname:'cdn.brawlify.com'}],
    },
  }

export default nextConfig;
