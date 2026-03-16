module.exports = {
  apps: [
    {
      name: "brawl-backend",
      cwd: "./backend/src",
      script: "./start-backend.sh",
      interpreter: "bash",
      env: {
        BRAWL_AI_BACKEND_VENV: "/home/oleg/brawl-ai/.venv-runtime",
        BRAWL_AI_CONFIG: "/home/oleg/brawl-ai/backend/src/config.ini",
      },
    },
    {
      name: "frontend",
      cwd: "./frontend",
      script: "./node_modules/next/dist/bin/next",
      args: "start -p 3003",
      interpreter: "node",
      env: {
        NODE_ENV: "production",
      },
    },
  ],
};
