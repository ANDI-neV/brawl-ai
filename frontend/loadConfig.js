const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
const ini = require('ini');

console.log('NODE_ENV:', process.env.NODE_ENV);

const env = process.env.NODE_ENV?.trim() || 'development';

const configPath = path.resolve(__dirname, 'frontend_config.ini');

if (fs.existsSync(configPath)) {
    const config = ini.parse(fs.readFileSync(configPath, 'utf-8'));
    console.log('Parsed Config:', config);

    const envConfig = config[env] || {};
    if (Object.keys(envConfig).length === 0) {
        console.error(`No configuration found for environment: ${env}`);
        process.exit(1);
    }
    console.log('Selected Environment Config:', envConfig);

    const dotenvContent = Object.entries(envConfig)
        .map(([key, value]) => `${key}=${value}`)
        .join('\n');
    console.log('Generated .env Content:', dotenvContent);

    fs.writeFileSync(path.resolve(__dirname, '.env'), dotenvContent);
    console.log(`Loaded configuration for ${env} environment.`);
} else {
    console.error(`Config file not found at ${configPath}`);
    process.exit(1);
}
