<div align="center">

# BrawlAI

A solution for optimizing ranked drafting sequences in the mobile game Brawl Stars by Supercell.

<p align="center">
  <img src="https://private-user-images.githubusercontent.com/89548801/380198324-7b6584da-c97f-42ba-9fca-af95e479c42f.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mjk5NTYzNDksIm5iZiI6MTcyOTk1NjA0OSwicGF0aCI6Ii84OTU0ODgwMS8zODAxOTgzMjQtN2I2NTg0ZGEtYzk3Zi00MmJhLTlmY2EtYWY5NWU0NzljNDJmLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEwMjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMDI2VDE1MjA0OVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWUyMjEwYjdmNjMxNGUzOTM1NTE2ZjI2ZmEwMThjYmJhM2M2YTlmYjUwODM3ZDE1ODAwZTAyNjYwY2UxZGU3NDMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.WrzrEj6WZ-9H7yRL8-OKmM69uP6_ri7D4_kKUINiVTM" width="600" alt="BrawlAI Screenshot">
</p>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)
![Next JS](https://img.shields.io/badge/Next-black?style=for-the-badge&logo=next.js&logoColor=white)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)

</div>

## ✨ Features
- 🎯 Optimized drafting sequences for competitive play
- ⚡ Real-time analysis and suggestions
- 📱 Responsive, mobile-friendly interface
- 🤖 AI-powered decision making
- 🔄 Regular updates based on meta/balance changes

## 🚀 Usage
Free usage ***coming soon*** at [brawl-ai.com](https://brawl-ai.com)

## 💻 Running Locally

### Prerequisites
- Node.js (v18 or higher)
- Python 3.8+
- PostgreSQL (optional, for training)
- Brawl Stars API key

### Quick Start
1. Clone the repository
```bash
git clone https://github.com/ANDI-neV/brawl-ai.git
cd brawl-ai
```
2. Set up configuration, create `config.ini` in `backend/src`:
```ini
[Credentials]
api = YOUR_API_KEY
```
3. Install dependencies:
```bash
# Frontend
cd frontend
npm install

# Backend
cd ../backend
pip install -r requirements.txt
```
4. Start the application
```bash
# Run frontend
npm run dev

# Run backend
python src/web_server.py
```

## 🛠 Development
### Training a custom model
The project includes a pre-trained model at `backend/src/out/models/ai_model.pth` (model will be regularly updated in relation to new updates/balance changes), but you can train your own:
1. Set up database configuration in `config.ini` (and adjust URL parameters/database loading according to your setup)
```ini
[Credentials]
username = YOUR_USERNAME
password = YOUR_PASSWORD
host = YOUR_HOST_URL
database = YOUR_DATABASE_NAME
api = YOUR_API_KEY
```
2. Create the required tables in the database (`battles` and `players`)
3. Feed the database with and adjust parameters to your liking:
```bash
python src/feeding.py
```
5. Configure training parameters in `ai.py`
6. Run the training script:
```bash
python src/ai.py
```
## 📝 License
MIT

## 🙏 Acknowledgments
- Brawl Stars API for providing game data
- [BrawlAPI](https://brawlapi.com/#/) for providing high-quality images for brawlers and maps
