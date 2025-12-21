import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from . import config

class TrafficAgent:
    def __init__(self):
        self.q_table = {}
        self.classifier = self._train_classifier()
        
    def _train_classifier(self):
        """
        Trains the Decision Tree to classify traffic density (Low/Med/High).
        (Logic ported from your final.py)
        """
        print("--- [Agent] Training Traffic Classifier... ---")
        X_train = []
        y_train = []
        # Generate synthetic data
        for _ in range(2000): 
            X_train.append(np.random.randint(0, 3, size=6))
            y_train.append(0) # Low
        for _ in range(2000): 
            X_train.append(np.random.randint(3, 8, size=6))
            y_train.append(1) # Medium
        for _ in range(2000): 
            X_train.append(np.random.randint(8, 30, size=6))
            y_train.append(2) # High
            
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X_train, y_train)
        return clf

    def get_state(self, queue_lengths, current_phase):
        """
        Converts raw data into a state tuple: (Traffic_Level, Phase).
        """
        traffic_level = self.classifier.predict([queue_lengths])[0]
        return (int(traffic_level), current_phase)

    def choose_action(self, state):
        """Epsilon-Greedy Policy."""
        if random.random() < config.EPSILON:
            return random.choice(config.ACTIONS)
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(config.ACTIONS))
            
        return int(np.argmax(self.q_table[state]))

    def learn(self, old_state, action, reward, new_state):
        """Updates Q-Table using Bellman Equation."""
        if old_state not in self.q_table: 
            self.q_table[old_state] = np.zeros(len(config.ACTIONS))
        if new_state not in self.q_table: 
            self.q_table[new_state] = np.zeros(len(config.ACTIONS))
            
        old_q = self.q_table[old_state][action]
        best_future_q = np.max(self.q_table[new_state])
        
        # Bellman Update
        new_q = old_q + config.ALPHA * (reward + config.GAMMA * best_future_q - old_q)
        self.q_table[old_state][action] = new_q