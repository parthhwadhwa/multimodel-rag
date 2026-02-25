import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",  // enables lean multi-stage Docker build
};

export default nextConfig;
