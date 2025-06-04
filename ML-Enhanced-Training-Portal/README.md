ML-Enhanced Training Portal
Overview
The ML-Enhanced Training Portal uses a Soft Actor-Critic (SAC) reinforcement learning model with LSTM to provide personalized learning paths for students. It dynamically recommends modules (Easy, Medium, Hard) based on performance metrics like scores, time spent, and attempts, enhancing technical and soft skills for better employability.
Features

Login System: Secure access via Streamlit.
Skill Quiz: 5-question assessment for tailored recommendations.
Adaptive Modules: SAC-driven difficulty adjustments.
Progress Tracking: Visual dashboards with Altair.
Reports: Export performance data as CSV.

Repository
ML-Enhanced-Training-Portal
Structure
adaptive_learning_project/
├── ML-Enhanced-Training-Portal/
│   ├── data/
│   │   ├── adaptive_learning_dataset.csv  # Training dataset
│   │   └── questions.json                 # Quiz questions
│   ├── model/
│   │   ├── sac_adaptive_learning_model_actor.keras   # SAC actor
│   │   ├── sac_adaptive_learning_model_critic_1.keras # SAC critic 1
│   │   ├── sac_adaptive_learning_model_critic_2.keras # SAC critic 2
│   ├── adaptive_test.py                  # Streamlit app
│   ├── TRain.py                          # SAC training script
│   └── requirements.txt                  # Dependencies
└── README.md                             # Documentation

Installation

Clone the repo:git clone https://github.com/KerulKidecha234/ML-Enhanced-Training-Portal.git


Navigate to the directory:cd ML-Enhanced-Training-Portal


Set up a virtual environment:python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt


Run the app:streamlit run adaptive_test.py



Usage

Login: Enter Student ID and job preference.
Select Domain: Choose Aptitude, Technical, or Soft Skills.
Take Quiz: Answer 5 questions for recommendations.
Learn: Engage with adaptive modules.
Track: View dashboards and export reports.

Technologies

ML: SAC with LSTM (TensorFlow/Keras)
Data: Pandas, NumPy, Scikit-learn
Frontend: Streamlit, Altair
Backend: Python
Storage: JSON (questions), CSV (data)

Dataset

adaptive_learning_dataset.csv: 1000 records with Student_ID, Previous_Score, Time_Spent, Attempt_Count, Module_Difficulty (0=Easy, 1=Medium, 2=Hard), Next_Recommended_Module.
questions.json: Quiz questions by domain and difficulty.

Model

SAC Agent (TRain.py):
Setup: Actor with LSTM (128 units), Critics with Dense (256 units).
Training: 1500 episodes, replay buffer (10,000), batch size 64.
Hyperparameters: Gamma=0.99, Alpha=0.1, Tau=0.002.
Reward: +2 (correct), -1 (incorrect).


Inference: adaptive_test.py uses SAC actor with fallback rules.

Training
To retrain:
python TRain.py

Models saved in model/.
Future Plans

Add soft skill analytics.
Introduce VR/AR content.
Enable peer collaboration.

Contributors

Bhavya Jappi (60009220004)  
Aniket Waghela (60009220033)  
Kerul Kidecha (60009220064)  
Guide: Prof. Poonam Jadhav

License
MIT License (see LICENSE).
Acknowledgements
Thanks to Dr. Jadhav, Dr. Srivastava, Kriti, Dr. Vasudevan, and the CS Dept. at Dwarkadas J. Sanghvi College.</xai

Issues Identified in the Provided README.md Attempt
The README.md you shared (in the prompt under "make like this for my ml enhance portal") contains several formatting and content errors, which I’ve corrected in the version above. Here are the issues I fixed:

Syntax Errors:
Extra <</ and <xaiArtifactId> tags are malformed. I used proper <xaiArtifact> tags.
</xai and </xaiArtifactId> are incorrect; corrected to `



