import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from . import config

class TrafficAgent:
    def __init__(self, n_detectors, tl_id=None):
        self.n_detectors = n_detectors
        self.tl_id = tl_id  # Store TL ID
        self.q_table = {}
        self.classifier = self._train_classifier()
        print(f"Agent for {self.tl_id} initialized with {self.n_detectors} detectors")
        
    def _train_classifier(self):
        """
        Trains the Decision Tree to classify traffic density (Low/Med/High).
        Uses variable number of detectors per TL.
        """
        print(f"--- [Agent] Training Traffic Classifier for {self.n_detectors} detectors ---")
        X_train = []
        y_train = []
        
        # Generate synthetic data for the number of detectors
        # Low density (0-2 vehicles per detector)
        for _ in range(2000): 
            X_train.append(np.random.randint(0, 3, size=self.n_detectors))
            y_train.append(0)  # Low
            
        # Medium density (3-7 vehicles per detector)
        for _ in range(2000): 
            X_train.append(np.random.randint(3, 8, size=self.n_detectors))
            y_train.append(1)  # Medium
            
        # High density (8-30 vehicles per detector)
        for _ in range(2000): 
            X_train.append(np.random.randint(8, 30, size=self.n_detectors))
            y_train.append(2)  # High
            
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(X_train, y_train)
        
        print(f"--- [Agent] Classifier trained successfully for {self.n_detectors} features ---")
        return clf

    def get_state(self, queue_lengths, current_phase):
        """
        Converts raw data into a state tuple: (Traffic_Level, Phase).
        """
        # Ensure queue_lengths has the right length
        if len(queue_lengths) != self.n_detectors:
            print(f"Warning: Expected {self.n_detectors} detectors, got {len(queue_lengths)}")
            if len(queue_lengths) < self.n_detectors:
                queue_lengths = queue_lengths + [0] * (self.n_detectors - len(queue_lengths))
            else:
                queue_lengths = queue_lengths[:self.n_detectors]
        
        try:
            # Convert to numpy array and reshape for prediction
            queue_array = np.array(queue_lengths).reshape(1, -1)
            traffic_level = self.classifier.predict(queue_array)[0]
            
            # Normalize phase (assume caller handles % num_phases)
            return (int(traffic_level), current_phase)
        except Exception as e:
            print(f"Error in get_state: {e}")
            # Return default state in case of error
            return (1, current_phase)

    def choose_action(self, state):
        """Epsilon-Greedy Policy."""
        if random.random() < config.EPSILON:
            action = random.choice(config.ACTIONS)
            return action
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(config.ACTIONS))
            
        action = int(np.argmax(self.q_table[state]))
        return action

    def learn(self, old_state, action, reward, new_state):
        """Updates Q-Table using Bellman Equation."""
        # Initialize Q-values for states if they don't exist
        if old_state not in self.q_table:
            self.q_table[old_state] = np.zeros(len(config.ACTIONS))
        if new_state not in self.q_table:
            self.q_table[new_state] = np.zeros(len(config.ACTIONS))
            
        old_q = self.q_table[old_state][action]
        best_future_q = np.max(self.q_table[new_state])
        
        # Bellman Update
        new_q = old_q + config.ALPHA * (reward + config.GAMMA * best_future_q - old_q)
        self.q_table[old_state][action] = new_q
        
    def save_q_table(self, filename="q_table.npy"):
        """Save the Q-table to a file."""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}")
        print(f"States learned: {len(self.q_table)}")
        
    def load_q_table(self, filename="q_table.npy"):
        """Load the Q-table from a file."""
        import pickle
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded from {filename}")
            print(f"States loaded: {len(self.q_table)}")
            return True
        except Exception as e:
            print(f"Could not load Q-table from {filename}: {e}")
            return False
        
    def choose_emergency_action(self, current_phase_state):
        """
        Determines action during an emergency (Ambulance present).
        Logic: If the light is NOT Green, Force Switch (1). If Green, Keep (0).
        """
        # 'r' = red, 'y' = yellow, 'G' = Priority Green, 'g' = Green
        # If we see ANY Red ('r') or Yellow ('y') in the active index, we try to switch to find Green.
        if 'r' in current_phase_state or 'y' in current_phase_state:
            return 1 # Force Switch to try and get Green
        else:
            return 0 # Hold Green