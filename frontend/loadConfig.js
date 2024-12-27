const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
const ini = require('ini');

const env = process.env.NODE_ENV || 'development';
const configPath = path.resolve(__dirname, 'frontend_config.ini');

if (fs.existsSync(configPath)) {
    const config = ini.parse(fs.readFileSync(configPath, 'utf-8'));
    const envConfig = config[env] || {};

    const dotenvContent = Object.entries(envConfig)
        .map(([key, value]) => `${key}=${value}`)
        .join('\n');

    fs.writeFileSync(path.resolve(__dirname, '.env'), dotenvContent);
    console.log(`Loaded configuration for ${env} environment.`);
} else {
    console.error(`Config file not found at ${configPath}`);
    process.exit(1);
}
