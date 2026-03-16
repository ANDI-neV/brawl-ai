module.exports = {
  apps: [
    {
      name: "brawl-backend",
      cwd: "./backend/src",
      script: "./start-backend.sh",
      interpreter: "bash",
      env: {
        BRAWL_AI_CONFIG: "/home/oleg/brawl-ai/backend/src/config.ini",
      },
    },
    {
      name: "frontend",
      cwd: "./frontend",
      script: "npm",
      args: "start",
      env: {
        NODE_ENV: "production",
      },
    },
  ],
};
