# ==============================================================================
# SYSTÈME COMPLET : SUMO + RL + ML + VISION (AVEC PRIORITÉ URGENCE)
# ==============================================================================

import os
import sys
import random
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------------------
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Erreur : Variable 'SUMO_HOME' non définie.")

import traci

Sumo_config = [
    'sumo-gui',
    '-c', 'RL.sumocfg',
    '--step-length', '0.5', 
    '--lateral-resolution', '0'
]

# ------------------------------------------------------------------------------
# 2. ML (CLASSIFICATION TRAFIC)
# ------------------------------------------------------------------------------
def train_traffic_classifier():
    print("--- [ML] Entraînement du modèle... ---")
    X_train = []
    y_train = []
    for _ in range(2000): X_train.append(np.random.randint(0, 3, size=6)); y_train.append(0)
    for _ in range(2000): X_train.append(np.random.randint(3, 8, size=6)); y_train.append(1)
    for _ in range(2000): X_train.append(np.random.randint(8, 30, size=6)); y_train.append(2)
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    return clf

traffic_classifier = train_traffic_classifier()

# ------------------------------------------------------------------------------
# 3. COMPUTER VISION (PIPELINE COMPLET + URGENCE)
# ------------------------------------------------------------------------------
def get_real_image_analysis_with_debug(save_images=False):
    """
    Détecte : Voitures (Jaune), Poids Lourds (Rouge/Magenta), Urgences (Blanc).
    """
    image_path = "sumo_capture.png"
    
    try:
        traci.gui.screenshot("View #0", image_path)
    except:
        return 0, 0, 0
    time.sleep(0.05)
    
    img = cv2.imread(image_path)
    if img is None: return 0, 0, 0

    if save_images: cv2.imwrite("Rapport_1_Originale.png", img)

    # Prétraitement
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
    
    # --- DÉFINITION DES MASQUES ---

    # 1. VOITURES (Jaune)
    mask_cars = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))

    # 2. POIDS LOURDS (Magenta + Rouge) - SANS CYAN
    mask_mag = cv2.inRange(hsv, np.array([135, 100, 100]), np.array([165, 255, 255]))
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    )
    mask_heavy = cv2.bitwise_or(mask_mag, mask_red)

    # 3. URGENCES (Blanc)
    # Le blanc a une saturation très faible (0-50) et une valeur haute (200-255)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask_emerg = cv2.inRange(hsv, lower_white, upper_white)

    if save_images: cv2.imwrite("Rapport_3_Masque_Urgence.png", mask_emerg)

    # --- MORPHOLOGIE ---
    kernel = np.ones((5, 5), np.uint8)
    
    # Nettoyage
    mask_cars = cv2.dilate(cv2.erode(mask_cars, kernel, iterations=1), kernel, iterations=2)
    mask_heavy = cv2.dilate(cv2.erode(mask_heavy, kernel, iterations=1), kernel, iterations=2)
    mask_emerg = cv2.dilate(cv2.erode(mask_emerg, kernel, iterations=1), kernel, iterations=2)

    # --- COMPTAGE ---
    contours_cars, _ = cv2.findContours(mask_cars, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_heavy, _ = cv2.findContours(mask_heavy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_emerg, _ = cv2.findContours(mask_emerg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_cars = sum(1 for c in contours_cars if cv2.contourArea(c) > 100)
    final_heavy = sum(1 for c in contours_heavy if cv2.contourArea(c) > 100)
    final_emerg = sum(1 for c in contours_emerg if cv2.contourArea(c) > 100)
    
    # --- VISUALISATION ---
    if save_images:
        img_res = img.copy()
        # Dessin Voitures (Vert)
        for c in contours_cars:
            if cv2.contourArea(c) > 100:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(img_res, (x,y), (x+w,y+h), (0, 255, 0), 2)
        # Dessin Poids Lourds (Rouge)
        for c in contours_heavy:
            if cv2.contourArea(c) > 100:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(img_res, (x,y), (x+w,y+h), (0, 0, 255), 2)
                cv2.putText(img_res, "HEAVY", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # Dessin Urgence (Bleu + Texte Gros)
        for c in contours_emerg:
            if cv2.contourArea(c) > 100:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(img_res, (x,y), (x+w,y+h), (255, 255, 0), 3)
                cv2.putText(img_res, "URGENCE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
        cv2.imwrite("Rapport_6_Resultat_Final.png", img_res)
            
    return final_cars, final_heavy, final_emerg

def get_reward_via_ComputerVision(debug=False):
    n_c, n_h, n_e = get_real_image_analysis_with_debug(save_images=debug)
    
    # Pénalités : Voiture=1, Camion=10, Urgence=100 (Priorité Absolue)
    penalty = (n_c * 1.0) + (n_h * 10.0) + (n_e * 100.0)
    return -penalty, n_c, n_h, n_e

# ------------------------------------------------------------------------------
# 4. RL & BOUCLE PRINCIPALE
# ------------------------------------------------------------------------------
TOTAL_STEPS = 200
ALPHA, GAMMA, EPSILON = 0.1, 0.9, 0.1
ACTIONS = [0, 1]
Q_table = {}

# Paramètre de sécurité standard
STANDARD_MIN_GREEN = 10 
last_switch_step = -STANDARD_MIN_GREEN

def get_state_ML():
    detectors = ["Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2", 
                 "Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"]
    queues = [traci.lanearea.getLastStepVehicleNumber(d) for d in detectors]
    trafic_level = traffic_classifier.predict([queues])[0]
    phase = traci.trafficlight.getPhase("Node2")
    return (int(trafic_level), phase)

def get_action(state):
    if random.random() < EPSILON: return random.choice(ACTIONS)
    if state not in Q_table: Q_table[state] = np.zeros(len(ACTIONS))
    return int(np.argmax(Q_table[state]))

def update_Q(old_s, a, r, new_s):
    if old_s not in Q_table: Q_table[old_s] = np.zeros(len(ACTIONS))
    if new_s not in Q_table: Q_table[new_s] = np.zeros(len(ACTIONS))
    old_q = Q_table[old_s][a]
    best_future = np.max(Q_table[new_s])
    Q_table[old_s][a] = old_q + ALPHA * (r + GAMMA * best_future - old_q)

# --- DÉMARRAGE ---
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

history_steps = []
history_rewards = []
cumulative_reward = 0

print("\n=== SYSTEME PRET AVEC PRIORITÉ URGENCE ===")

for step in range(TOTAL_STEPS):
    
    # 1. Perception
    state = get_state_ML()
    action = get_action(state)
    
    # 2. Analyse Urgence (AVANT D'AGIR)
    # On regarde l'image maintenant pour savoir si on doit forcer le feu
    # On ne sauvegarde pas l'image à chaque pas pour la vitesse (debug=False)
    _, _, n_emerg = get_real_image_analysis_with_debug(save_images=False)
    
    # --- LOGIQUE D'URGENCE (OVERRIDE) ---
    current_min_green = STANDARD_MIN_GREEN
    
    if n_emerg > 0:
        # URGENCE DÉTECTÉE !
        # 1. On supprime la contrainte de temps (Switch Instantané autorisé)
        current_min_green = 0 
        print(f"!!! ALERTE URGENCE (Step {step}) !!! Priorité Absolue activée.")
        
        # 2. Optionnel : On peut même FORCER l'action '1' (Switch) si on veut débloquer
        # Mais ici on laisse le RL décider, sachant qu'il a 0 délai.
    
    # 3. Action
    if action == 1 and (step - last_switch_step > current_min_green):
        traci.trafficlight.setPhase("Node2", (traci.trafficlight.getPhase("Node2")+1)%4)
        last_switch_step = step
        
    traci.simulationStep()
    
    # 4. Évaluation et Apprentissage
    new_state = get_state_ML()
    
    # Sauvegarde image à l'étape 50 ou si Urgence détectée (pour le rapport)
    do_save = (step == 50) or (n_emerg > 0 and step % 10 == 0)
    reward, n_c, n_h, n_e = get_reward_via_ComputerVision(debug=do_save)
    
    cumulative_reward += reward
    update_Q(state, action, reward, new_state)
    
    if step % 10 == 0:
        status = "URGENCE" if n_e > 0 else "Normal"
        print(f"Step {step} | Mode: {status} | Vision: {n_c} Cars, {n_h} Heavy, {n_e} Emerg | Rwd: {reward}")
        history_steps.append(step)
        history_rewards.append(cumulative_reward)

traci.close()

# Visualisation
plt.figure(figsize=(10, 6))
plt.plot(history_steps, history_rewards)
plt.title("Performance RL avec Priorité Urgence (Computer Vision)")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid(True)
plt.show()
