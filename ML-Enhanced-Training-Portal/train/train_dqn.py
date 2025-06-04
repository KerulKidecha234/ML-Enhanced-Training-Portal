import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random
from collections import deque
import pandas as pd

# ✅ Load dataset safely
dataset_path = "../data/adaptive_learning_dataset.csv"
try:
    df = pd.read_csv(dataset_path)
except Exception as e:
    print(f"⚠️ Error loading dataset: {str(e)}")
    exit()

# ✅ Encode categorical variables
module_map = {"Easy": 0, "Medium": 1, "Hard": 2}
df["Module_Difficulty"] = df["Module_Difficulty"].map(module_map)
df["Next_Recommended_Module"] = df["Next_Recommended_Module"].map(module_map)

# ✅ Define features and target variable
X = df[["Previous_Score", "Time_Spent", "Attempt_Count", "Module_Difficulty"]].values
y = df["Next_Recommended_Module"].values

# ✅ Fix: Register the model properly for serialization
@tf.keras.utils.register_keras_serializable()
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.90  # ✅ Adjusted discount factor for better learning
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05  # ✅ Prevent getting stuck at exploration
        self.epsilon_decay = 0.995  # ✅ Slow decay for better convergence
        self.learning_rate = 0.001

        # ✅ Build Neural Network
        self.model = Sequential([
            Dense(128, input_dim=self.state_size, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='softmax')  # ✅ Softmax for better stability
        ])
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Exploration
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # Exploitation

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # ✅ Adjusted epsilon decay for better learning

    # ✅ Fix: Ensure serialization works
    def get_config(self):
        return {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "learning_rate": self.learning_rate
        }

# ✅ Train the model efficiently
state_size = 4
action_size = 3
agent = DQNAgent(state_size, action_size)

batch_size = 64
episodes = 200  # ✅ Optimized training time

for e in range(episodes):
    state = np.reshape(X[random.randint(0, len(X) - 1)], [1, state_size])
    for time in range(10):
        action = agent.act(state)
        next_state = np.reshape(X[random.randint(0, len(X) - 1)], [1, state_size])
        
        # ✅ Fix: Strongly penalize incorrect answers
        reward = 1 if action == y[random.randint(0, len(y) - 1)] else -3  
        
        done = time == 9
        agent.remember(state, action, reward, next_state, done)
        state = next_state
    
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

# ✅ Save trained model
model_path = "../model/dqn_adaptive_learning_model.h5"
agent.model.save(model_path)
print(f"✅ Model saved successfully at: {model_path}")
