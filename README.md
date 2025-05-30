<div align="center">

# BrawlAI

A solution for optimizing ranked drafting sequences in the mobile game Brawl Stars by Supercell.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a06d8375-7905-4995-bd9d-9b874c3810a5">
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
Free usage at [brawl-ai.com](https://brawl-ai.com)

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
GPLv3

## 🙏 Acknowledgments
- Brawl Stars API for providing game data
- [BrawlAPI](https://brawlapi.com/#/) for providing high-quality images for brawlers and maps
