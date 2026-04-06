# 🎨 Image Style Transformation Web Application

A free, fast, and user-friendly web application for artistic image style transfer. Upload your photos and transform them into eight different artistic styles using deep learning models.

---

## ✨ Features

* **8 Artistic Styles**: Sketch, Ukiyo-e, Cyberpunk, Hayao (Ghibli-style anime), Shinkai, Paprika, Fauvism, Pointillism
* **Free & Unlimited**: No watermarks, no usage quotas, no subscription fees
* **Fast Processing**: 3–5 seconds per image on standard CPU hardware
* **Local Processing**: All models run locally, no external API dependencies
* **User System**: Registration, login, personal gallery, favorites, and portfolio management
* **Community Features**: Weekly challenges, likes, and public gallery
* **Responsive Design**: Works on desktop, tablet, and mobile devices

---

## 🛠 Tech Stack

| Layer            | Technology                   |
| ---------------- | ---------------------------- |
| Backend          | Python 3.11.9+ / FastAPI        |
| AI Framework     | PyTorch 2.10.0               |
| Image Processing | OpenCV 4.8.1 + PIL           |
| Database         | MongoDB (Motor async driver) |
| Frontend         | HTML5 + CSS3 + JavaScript    |
| Authentication   | JWT + bcrypt                 |

---

## 🐍 Requirements

* Python 3.11.9 or higher
* MongoDB (local or cloud)
* 4GB RAM minimum (8GB recommended)
* CPU only (no GPU required)

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/tuyi3008/final-year-project.git
cd final-year-project
```

---

### 2. Create and Activate Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

Create a `.env` file in the project root directory:

```env
# JWT Secret Key
JWT_SECRET_KEY=your-secret-key-here-change-this-in-production

# MongoDB Configuration
MONGODB_URL=mongodb://your_username:your_password@localhost:27018/
DB_NAME=styletrans_db
```

⚠️ **Security Notes**

* Generate your own key using: `openssl rand -hex 32`
* Do NOT commit `.env` to GitHub
* Change default database credentials

---

### 5. Start MongoDB

#### Using Docker

```bash
docker run -d \
  --name mongodb \
  -p 27018:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=your_username \
  -e MONGO_INITDB_ROOT_PASSWORD=your_password \
  mongo:latest
```

---


### 6. Download Model Files

Place all model files inside the `models/` directory:

```text
models/
├── sketch.pt          # Self-trained (Sketch)
├── ukiyoe.pt          # Self-trained (Ukiyo-e)
├── cyberpunk.pt       # Self-trained (Cyberpunk)
├── anime.pt           # Adapted from fast-neural-style (Fauvism)
├── ink.pt             # Adapted from fast-neural-style (Pointillism)
├── hayao.pt           # From AnimeGANv2 (Hayao/Ghibli style)
├── shinkai.pt         # From AnimeGANv2 (Shinkai style)
└── paprika.pt         # From AnimeGANv2 (Paprika style)
```


#### Model Sources

**1. Self-Trained Models (Provided by Author)**

Download from: [Google Drive](https://drive.google.com/drive/folders/1Wsw3iWFyzr4qA5K9Wz4yHjhf9YtIwmOu)

| File | Style |
| --- | --- |
| sketch.pt | Sketch |
| ukiyoe.pt | Ukiyo-e |
| cyberpunk.pt | Cyberpunk |

**2. Pre-Trained Models (Third Party)**

| File(s) | Style | Source | License |
| --- | --- | --- | --- |
| anime.pt, ink.pt | Fauvism, Pointillism | [fast-neural-style](https://github.com/pytorch/examples/tree/main/fast_neural_style) | BSD 3-Clause |
| hayao.pt, shinkai.pt, paprika.pt | Hayao, Shinkai, Paprika | [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2) | Non-commercial |

> **Note**: The PyTorch fast-neural-style models (`anime.pt`, `ink.pt`) have been renamed from the original `.pth` files for use in this project. Please refer to their official repository for more details.

### 7. Run the Application

```bash
uvicorn app:app --reload
```

👉 Open in browser:
http://localhost:8000

---

## 📁 Project Structure

```text
final-year-project/
├── app.py                 # Main FastAPI application
├── auth.py                # JWT authentication
├── database.py            # MongoDB connection
├── model.py               # Style transfer model loader
├── requirements.txt       # Python dependencies
├── README.md
├── .env                   # Environment variables
├── .gitignore
├── docker-compose.yml     # MongoDB container setup
├── public/                # Frontend files
├── models/                # 8 pretrained style models (.pt)
├── tests/                 # Unit tests
├── uploads/               # Temporary uploads
└── tf_env/                # Virtual environment (local only)
```

---

## 📡 API Endpoints

### Authentication
| Method | Endpoint | Description | Auth |
| --- | --- | --- | --- |
| POST | `/register` | User registration | ❌ |
| POST | `/login` | User login | ❌ |
| GET | `/profile` | Get user profile | ✅ |

### Style Transfer
| Method | Endpoint | Description | Auth |
| --- | --- | --- | --- |
| POST | `/stylize` | Apply artistic style to image | ❌ |
| GET | `/history` | Get user's transfer history | ✅ |

### Gallery & Social
| Method | Endpoint | Description | Auth |
| --- | --- | --- | --- |
| POST | `/gallery/publish` | Publish image to gallery | ✅ |
| GET | `/gallery/images` | View public gallery | ❌ |
| POST | `/gallery/like/{id}` | Like an image | ✅ |
| GET | `/favorites` | Get user's favorite images | ✅ |
| POST | `/favorites/{id}` | Save to favorites | ✅ |

### Challenges
| Method | Endpoint | Description | Auth |
| --- | --- | --- | --- |
| GET | `/challenges` | Get weekly challenges | ❌ |
| POST | `/challenges/submit` | Submit to challenge | ✅ |
---

## 🧪 Testing

Run tests:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=. --cov-report=html
```

---

## ⚡ Performance

*Test environment: Intel i7-10750H, 16GB RAM (CPU only)*

| Style | Avg Time (s) |
| --- | --- |
| Sketch | ~3.4s |
| Cyberpunk | ~3.3s |
| Hayao | ~3.9s |
| Shinkai | ~3.8s |
| Ukiyo-e | ~4.2s |
| Fauvism | ~3.6s |
| Pointillism | ~3.4s |
| Paprika | ~3.9s |

---

## 🛠 Troubleshooting

### MongoDB Connection Error

* Check MongoDB is running
* Verify `.env` config
* Confirm port (27018)

### Model Loading Error

* Ensure models exist in `/models`
* Check filenames match

### Port Already in Use

```bash
uvicorn app:app --reload --port 8001
```

---

## 🚀 Future Work

* Cloud deployment
* Video style transfer
* More artistic styles
* Mobile app version

---

## 🙏 Acknowledgments

* Supervisor: Basel Magableh
* TU Dublin School of Computer Science
* Beta testers

---

## 👨‍💻 Author

**Yi Tu**
Technological University Dublin, 2026
