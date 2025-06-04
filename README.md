# 🎓 ML-Enhanced Training Portal

An intelligent adaptive learning platform powered by Soft Actor-Critic (SAC) reinforcement learning with LSTM networks to deliver personalized educational experiences.

## 🌟 Overview

The ML-Enhanced Training Portal revolutionizes online learning by using advanced machine learning algorithms to create personalized learning paths. Our system employs a Soft Actor-Critic (SAC) reinforcement learning model with LSTM to dynamically recommend modules based on individual performance metrics, ensuring optimal learning outcomes for technical and soft skills development.

## ✨ Features

- **🔐 Secure Authentication**: Streamlit-based login system with student ID verification
- **📝 Intelligent Assessment**: 5-question skill quiz for personalized recommendations
- **🎯 Adaptive Learning**: SAC-driven difficulty adjustments (Easy, Medium, Hard)
- **📊 Real-time Analytics**: Interactive progress tracking with Altair visualizations
- **📈 Performance Reports**: Exportable CSV reports for detailed analysis
- **🎨 Intuitive Interface**: Clean, responsive UI built with Streamlit
- **🔄 Dynamic Content**: Continuously adapting content based on user performance

## 🏗️ Project Structure

```
adaptive_learning_project/
├── ML-Enhanced-Training-Portal/
│   ├── data/
│   │   ├── adaptive_learning_dataset.csv    # Training dataset (1000+ records)
│   │   └── questions.json                   # Domain-specific quiz questions
│   ├── model/
│   │   ├── sac_adaptive_learning_model_actor.keras      # SAC Actor Network
│   │   ├── sac_adaptive_learning_model_critic_1.keras   # SAC Critic Network 1
│   │   └── sac_adaptive_learning_model_critic_2.keras   # SAC Critic Network 2
│   ├── adaptive_test.py                     # Main Streamlit application
│   ├── TRain.py                            # SAC model training script
│   └── requirements.txt                     # Python dependencies
├── app/                                     # Additional application files
├── train/                                   # Training utilities
├── data/                                    # Additional datasets
├── model/                                   # Model checkpoints
├── requirements.txt                         # Project dependencies
└── README.md                               # Project documentation
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/KerulKidecha234/ML-Enhanced-Training-Portal.git
cd ML-Enhanced-Training-Portal
```

### Step 2: Set up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run adaptive_test.py
```

The application will be available at `http://localhost:8501`

## 💡 Usage Guide

### Getting Started
1. **Login**: Enter your Student ID and select your job preference
2. **Domain Selection**: Choose from Aptitude, Technical, or Soft Skills
3. **Assessment**: Complete the 5-question quiz for personalized recommendations
4. **Learning**: Engage with adaptive modules tailored to your skill level
5. **Progress Tracking**: Monitor your performance through interactive dashboards
6. **Reports**: Export detailed performance analytics as CSV files

### Learning Domains
- **📚 Aptitude**: Logical reasoning, quantitative analysis, verbal ability
- **💻 Technical**: Programming concepts, system design, algorithms
- **🤝 Soft Skills**: Communication, leadership, teamwork, problem-solving

## 🔧 Technology Stack

### Machine Learning
- **TensorFlow/Keras**: Neural network implementation
- **Soft Actor-Critic (SAC)**: Reinforcement learning algorithm
- **LSTM Networks**: Sequential data processing
- **Scikit-learn**: Data preprocessing and evaluation

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **JSON**: Question bank storage
- **CSV**: Dataset management

### Frontend & Visualization
- **Streamlit**: Web application framework
- **Altair**: Interactive data visualizations
- **Python**: Backend logic

## 📊 Dataset Information

### Training Dataset (`adaptive_learning_dataset.csv`)
- **Records**: 1000+ student interaction records
- **Features**:
  - `Student_ID`: Unique identifier
  - `Previous_Score`: Historical performance
  - `Time_Spent`: Learning duration metrics
  - `Attempt_Count`: Number of attempts
  - `Module_Difficulty`: Encoded difficulty levels (0=Easy, 1=Medium, 2=Hard)
  - `Next_Recommended_Module`: Target recommendations

### Question Bank (`questions.json`)
Structured quiz questions organized by:
- Domain categories (Aptitude, Technical, Soft Skills)
- Difficulty levels (Easy, Medium, Hard)
- Multiple choice format with correct answers

## 🤖 Model Architecture

### SAC Agent Configuration
- **Actor Network**: LSTM with 128 units for sequential decision making
- **Critic Networks**: Dual critics with 256 dense units each
- **Training Episodes**: 1500 episodes with experience replay
- **Replay Buffer**: 10,000 experience samples
- **Batch Size**: 64 samples per training step

### Hyperparameters
- **Discount Factor (γ)**: 0.99
- **Soft Update Rate (τ)**: 0.002
- **Entropy Coefficient (α)**: 0.1
- **Learning Rate**: Adaptive based on performance

### Reward System
- **Correct Answer**: +2 points
- **Incorrect Answer**: -1 point
- **Time Bonus**: Additional rewards for efficient learning

## 🎯 Model Training

To retrain the model with new data:

```bash
python TRain.py
```

Trained models are automatically saved in the `model/` directory:
- Actor network for action selection
- Critic networks for value estimation

## 🔮 Future Enhancements

- **📈 Advanced Analytics**: Comprehensive soft skill assessment and analytics
- **🥽 Immersive Learning**: VR/AR content integration for enhanced engagement
- **👥 Collaborative Features**: Peer-to-peer learning and group challenges
- **🌐 Multi-language Support**: Internationalization for global accessibility
- **📱 Mobile Application**: Native mobile app development
- **🎮 Gamification**: Achievement systems and leaderboards

**Faculty Guide**: Prof. Poonam Jadhav

## 🏫 Institution

**Dwarkadas J. Sanghvi College of Engineering**  
Computer Science Department  
Mumbai, Maharashtra, India

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

