# ML-Enhanced Training Portal

The ML-Enhanced Training Portal uses a Soft Actor-Critic (SAC) reinforcement learning model with LSTM to provide personalized learning paths for students.  
It dynamically recommends modules (Easy, Medium, Hard) based on performance metrics like scores, time spent, and attempts — enhancing technical and soft skills for better employability.

---

## 🔐 Features

- **Login System:** Secure access via Streamlit  
- **Skill Quiz:** 5-question assessment for tailored recommendations  
- **Adaptive Modules:** SAC-driven difficulty adjustments  
- **Progress Tracking:** Visual dashboards with Altair  
- **Reports:** Export performance data as CSV

---

## 🗂 Repository Structure

ML-Enhanced-Training-Portal/
├── adaptive_learning_project/
│ ├── data/
│ │ ├── adaptive_learning_dataset.csv # Training dataset
│ │ └── questions.json # Quiz questions
│ ├── model/
│ │ ├── sac_adaptive_learning_model_actor.keras # SAC actor
│ │ ├── sac_adaptive_learning_model_critic_1.keras # SAC critic 1
│ │ └── sac_adaptive_learning_model_critic_2.keras # SAC critic 2
│ ├── train/
│ │ └── train_dqn.py # SAC training script
│ ├── app/
│ │ └── adaptive_test.py # Streamlit app
│ ├── TRain.py
│ ├── requirements.txt # Dependencies
│ └── README.md # Documentation
| └──


▶️ Run the App
bash
Copy
Edit
streamlit run adaptive_test.py
💡 Usage
Login: Enter Student ID and job preference

Select Domain: Aptitude, Technical, or Soft Skills

Take Quiz: Answer 5 questions for recommendations

Learn: Get adaptive modules based on performance

Track: View dashboards and export reports

🧪 Technologies Used
ML: SAC with LSTM (TensorFlow/Keras)

Data: Pandas, NumPy, Scikit-learn

Frontend: Streamlit, Altair

Backend: Python

Storage: JSON (questions), CSV (data)

📊 Dataset
adaptive_learning_dataset.csv: 1000 records with:

Student_ID

Previous_Score

Time_Spent

Attempt_Count

Module_Difficulty (0=Easy, 1=Medium, 2=Hard)

Next_Recommended_Module

questions.json: Quiz questions by domain and difficulty

🧠 Model: SAC Agent (TRain.py)
Actor: LSTM with 128 units

Critics: Dense layers with 256 units

Training:

Episodes: 1500

Replay Buffer: 10,000

Batch Size: 64

Hyperparameters:

Gamma = 0.99

Alpha = 0.1

Tau = 0.002

Reward Function:

+2 for correct answer

-1 for incorrect answer

Inference:

adaptive_test.py uses SAC actor

Includes fallback rules for unseen cases

🔁 Retraining the Model
bash
Copy
Edit
python TRain.py
Models are saved in the model/ directory.

📈 Future Plans
Add soft skill analytics

Introduce VR/AR-based content

Enable peer collaboration features

👥 Contributors
Bhavya Jappi (60009220004)

Aniket Waghela (60009220033)

Kerul Kidecha (60009220064)

Guide: Prof. Poonam Jadhav

📜 License
MIT License — see LICENSE

🙏 Acknowledgements
Thanks to:

Dr. Poonam Jadhav

Dr. Srivastava

Kriti

Dr. Vasudevan

CS Dept., Dwarkadas J. Sanghvi College

yaml
Copy
Edit

---

