import dotenv from 'dotenv';
dotenv.config({ path: './.env' });
/** @type {import('next').NextConfig} */
console.log('NODE_ENV during build:', process.env.NODE_ENV);

const nextConfig = {
    reactStrictMode: true,
    images: {
        remotePatterns: [{ hostname: 'cdn.brawlify.com' }],
    },
};

export default nextConfig;
