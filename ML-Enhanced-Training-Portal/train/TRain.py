import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, BatchNormalization
import random
from collections import deque
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset safely
dataset_path = "../data/adaptive_learning_dataset.csv"
try:
    df = pd.read_csv(dataset_path)
except Exception as e:
    print(f"⚠️ Error loading dataset: {str(e)}")
    exit()

# Encode categorical variables
module_map = {"Easy": 0, "Medium": 1, "Hard": 2}
df["Module_Difficulty"] = df["Module_Difficulty"].map(module_map)
df["Next_Recommended_Module"] = df["Next_Recommended_Module"].map(module_map)

# Feature Scaling (Normalization)
scaler = StandardScaler()
X = scaler.fit_transform(df[["Previous_Score", "Time_Spent", "Attempt_Count", "Module_Difficulty"]].values)
y = df["Next_Recommended_Module"].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@tf.keras.utils.register_keras_serializable()
class SACAgent:
    def __init__(self, state_size, action_size, lstm_units=128):
        self.state_size = state_size
        self.action_size = action_size
        self.lstm_units = lstm_units
        self.memory = deque(maxlen=10000)  # Increased Replay Memory
        self.gamma = 0.99  # Discount factor
        self.learning_rate = 0.0005  # Reduced for stability
        self.alpha = 0.1  # Lower entropy regularization for better exploitation
        self.tau = 0.002  # More stable target updates

        # Build models
        self.actor = self.build_actor()
        self.critic_1 = self.build_critic()
        self.critic_2 = self.build_critic()
        self.target_critic_1 = self.build_critic()
        self.target_critic_2 = self.build_critic()
        self.update_target_networks(tau=1.0)  # Initialize target networks

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def build_actor(self):
        model = Sequential([
            Input(shape=(None, self.state_size)),  # LSTM expects variable timesteps
            LSTM(self.lstm_units, activation='relu', return_sequences=False),
            Dropout(0.2),  # Dropout added to prevent overfitting
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dense(self.action_size, activation='softmax')  # Output probability distribution over actions
        ])
        return model

    def build_critic(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dense(1)
        ])
        return model

    def update_target_networks(self, tau=None):
        """ Soft update target networks with tau """
        if tau is None:
            tau = self.tau
        for target_weights, model_weights in zip(self.target_critic_1.weights, self.critic_1.weights):
            target_weights.assign(tau * model_weights + (1 - tau) * target_weights)

        for target_weights, model_weights in zip(self.target_critic_2.weights, self.critic_2.weights):
            target_weights.assign(tau * model_weights + (1 - tau) * target_weights)

    def remember(self, state, action, reward, next_state, done):
        # Store transition in replay memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Add time dimension for LSTM
        state_actor = np.expand_dims(state, axis=1)
        action_prob = self.actor(state_actor, training=False)
        action = np.random.choice(self.action_size, p=action_prob.numpy()[0])
        return action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_actor = np.expand_dims(state, axis=1)

            target_q1 = self.target_critic_1(next_state)
            target_q2 = self.target_critic_2(next_state)
            target_q = tf.minimum(target_q1, target_q2)

            target = reward if done else reward + self.gamma * target_q

            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                q_value_1 = self.critic_1(state)
                q_value_2 = self.critic_2(state)
                critic_loss = tf.reduce_mean(tf.square(target - q_value_1)) + tf.reduce_mean(tf.square(target - q_value_2))

            critic_grads = tape1.gradient(critic_loss,
                                          self.critic_1.trainable_variables + self.critic_2.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_1.trainable_variables + self.critic_2.trainable_variables))

            # Actor update
            with tf.GradientTape() as tape:
                action_prob = self.actor(state_actor)
                actor_loss = -tf.reduce_mean(action_prob[0][action] * (reward + self.gamma * target))
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Update target networks
        self.update_target_networks()

    def save_model(self, path):
        self.actor.save(path + "_actor.keras")
        self.critic_1.save(path + "_critic_1.keras")
        self.critic_2.save(path + "_critic_2.keras")
        print(f"Model saved to {path}")

    def get_config(self):
        return {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "lstm_units": self.lstm_units
        }

# Training Configuration
state_size = 4
action_size = 3  # Three difficulty levels: Easy, Medium, Hard
agent = SACAgent(state_size, action_size)

batch_size = 64
episodes = 1500  # Increased training episodes

# Training Loop
for e in range(episodes):
    idx = random.randint(0, len(X_train) - 1)
    state = np.reshape(X_train[idx], (1, state_size))
    true_label = y_train[idx]

    for t in range(10):
        action = agent.act(state)
        reward = 2 if action == true_label else -1  # Less negative penalty

        idx = random.randint(0, len(X_train) - 1)
        next_state = np.reshape(X_train[idx], (1, state_size))
        true_label = y_train[idx]

        done = (t == 9)
        agent.remember(state, action, reward, next_state, done)
        state = next_state

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

# Save the trained model
model_path = "../model/sac_adaptive_learning_model.h5"
agent.save_model(model_path)
print(f"✅ Model saved successfully at: {model_path}")
