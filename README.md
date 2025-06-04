<h1 align="center">🚀 ML-Enhanced Training Portal</h1>

---

## 📖 Overview

The **ML-Enhanced Training Portal** uses a Soft Actor-Critic (SAC) reinforcement learning model with LSTM to deliver personalized learning paths for students.  
It adapts module difficulty (Easy, Medium, Hard) based on performance metrics like score, time spent, and attempts — promoting both technical and soft skill growth.

---

## ✨ Features

- ✅ **Login System:** Secure access via Streamlit  
- ✅ **Skill Quiz:** 5-question personalized quiz  
- ✅ **Adaptive Learning:** Module difficulty adapted via SAC model  
- ✅ **Performance Dashboard:** Visual progress insights using Altair  
- ✅ **Reports:** Exportable CSV performance data  

---

<h2 align="center">📁 Project Structure</h2>

adaptive_learning_project/
├── ML-Enhanced-Training-Portal/
│ ├── data/
│ │ ├── adaptive_learning_dataset.csv # Training dataset
│ │ └── questions.json # Quiz questions
│ ├── model/
│ │ ├── sac_adaptive_learning_model_actor.keras # SAC actor
│ │ ├── sac_adaptive_learning_model_critic_1.keras # SAC critic 1
│ │ └── sac_adaptive_learning_model_critic_2.keras # SAC critic 2
│ ├── adaptive_test.py # Streamlit App
│ ├── TRain.py # SAC Training Script
│ └── requirements.txt # Dependencies
├── app/ # Additional app files
├── train/ # Additional training files
├── data/ # Top-level duplicate?
├── model/ # Top-level duplicate?
├── requirements.txt # Top-level dependencies
└── README.md # Project documentation

yaml
Copy
Edit

---

<h2 align="center">⚙️ Installation & Setup</h2>

### 📌 Prerequisites

- Python 3.8+
- pip (Python package manager)
- (Optional) Virtual environment

---

### 🧱 Step 1: Clone the Repository

```bash
git clone https://github.com/KerulKidecha234/ML-Enhanced-Training-Portal.git
cd ML-Enhanced-Training-Portal
🧪 Step 2: Set Up Virtual Environment
bash
Copy
Edit
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
📦 Step 3: Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
🚀 Step 4: Run the App
bash
Copy
Edit
streamlit run adaptive_test.py
🧑‍💻 Usage
Login: Enter Student ID and job preference

Select Domain: Choose Aptitude, Technical, or Soft Skills

Take Quiz: Answer 5 questions for tailored recommendations

Learn: Get dynamic module suggestions

Track: View dashboards, export performance reports

🧠 Technologies
ML Model: SAC with LSTM (TensorFlow/Keras)

Data Processing: Pandas, NumPy, Scikit-learn

Frontend: Streamlit, Altair

Backend: Python

Storage: JSON (quiz questions), CSV (training data)

📊 Dataset
adaptive_learning_dataset.csv:
Contains 1000 records with Student_ID, Previous_Score, Time_Spent, Attempt_Count, Module_Difficulty, and Next_Recommended_Module.

questions.json:
Structured quiz questions based on domain and difficulty.

🧪 Model Training
File: TRain.py

Architecture:

Actor: LSTM (128 units)

Critics: Dense (256 units)

Training:

Episodes: 1500

Replay Buffer: 10,000

Batch Size: 64

Hyperparameters:

Gamma: 0.99

Alpha: 0.1

Tau: 0.002

Rewards:

+2 for correct, -1 for incorrect

Use python TRain.py to retrain the model
Models will be saved in the /model directory

🚧 Future Plans
Add soft skill analytics

Integrate VR/AR-based learning

Enable peer collaboration and feedback

👥 Contributors
Bhavya Jappi (60009220004)

Aniket Waghela (60009220033)

Kerul Kidecha (60009220064)

Guide: Prof. Poonam Jadhav

📄 License
This project is licensed under the MIT License. See LICENSE for details.

🙏 Acknowledgements
Special thanks to Dr. Jadhav, Dr. Srivastava, Kriti, Dr. Vasudevan, and the CS Department of Dwarkadas J. Sanghvi College of Engineering.

