# 🚦 Système Intelligent de Gestion du Trafic  
### (IA, Vision par Ordinateur & Simulation SUMO)

Ce projet implémente un **contrôleur adaptatif de feux de signalisation** basé sur :
- 🛣️ **SUMO** pour la simulation du trafic
- 👁️ **Vision par ordinateur** (OpenCV & YOLOv8)
- 🧠 **Algorithmes décisionnels** pour l’optimisation du flux et la priorité aux véhicules d’urgence

---

## 🧱 Architecture Générale

Le projet est structuré en **4 grandes phases**, chacune composée de cellules logiques (Notebook / Script Python).

---

## 🟢 Phase 1 : Infrastructure (SUMO)

### 📌 Cellule 1 : Initialisation de l’Environnement
**Rôle :**  
Configurer l’environnement système et vérifier que SUMO est correctement installé.

**Logique :**
- Définition du chemin vers `sumo-gui.exe`
- Import de la librairie `traci` pour permettre à Python de communiquer avec SUMO

---

### 📌 Cellule 2 : Génération du Réseau et du Trafic
**Rôle :**  
Créer physiquement le réseau routier et les véhicules.

**Logique :**
- Génération des fichiers XML :
  - `nodes.xml` (intersections)
  - `edges.xml` (routes)
  - `routes.xml` (véhicules)
- Création d’une **intersection à 4 voies**
- Compilation du réseau via `netconvert`

---

## 🔵 Phase 2 : Vision par Ordinateur (IA)

### 📌 Cellule 3 : Traitement d’Image Classique (Segment 2)
**Rôle :**  
Prétraiter l’image capturée pour améliorer l’analyse.

**Logique :**
- Application d’un **Flou Gaussien** pour réduire le bruit
- Détection des contours avec l’algorithme **Canny**

📎 *Limite :* Cette méthode ne reconnaît pas la nature des objets (uniquement des contours).

---

### 📌 Cellule 4 : Segmentation par IA (YOLOv8 – Segment 3)
**Rôle :**  
Reconnaissance intelligente des objets.

**Logique :**
- Chargement du modèle **YOLOv8-Seg**
- Détection et classification des objets :
  - 🚗 Voitures
  - 🚶 Piétons
  - 🚑 Véhicules d’urgence
- Création de **masques colorés** autour des objets détectés

---

## 🟡 Phase 3 : Collecte de Données (Monitoring)

### 📌 Cellules 5 à 8 : Tests de Capture & Debug
**Rôle :**  
Valider la communication en temps réel entre Python et SUMO.

**Logique :**
- Captures d’écran continues de la simulation
- Vérification de la stabilité (pas de crash SUMO)
- Ajustement des fréquences de capture

---

### 📌 Cellule 9 : Extraction des Métriques (Segment 4)
**Rôle :**  
Transformer les données visuelles en données numériques.

**Logique :**
Pour chaque voie (ex : `N2C`, `E2C`) :
- 📊 Nombre total de véhicules (**Densité**)
- ⛔ Nombre de véhicules arrêtés (**Longueur de la file d’attente**)

---

## 🔴 Phase 4 : Intelligence Décisionnelle (Le Cerveau)

### 📌 Cellule 10 : Algorithme de Comparaison Simple
**Rôle :**  
Optimisation basique du flux de trafic.

**Logique :**
- Comparaison des files d’attente
- Exemple :
  - Si **Nord > Est** → feu **VERT pour le Nord**

---

### 📌 Cellule 11 : Configuration du Scénario d’Urgence
**Rôle :**  
Tester la priorité des véhicules d’urgence.

**Logique :**
- Injection de véhicules d’urgence (en rouge) aux instants :
  - ⏱️ t = 30
  - ⏱️ t = 80
  - ⏱️ t = 150
- Vérification de leur détection par YOLO

---

### 📌 Cellule 12 : Contrôleur Adaptatif Final (Segment 5)
**Rôle :**  
Pilotage intelligent et prioritaire des feux.

**Logique :**
Application d’une **hiérarchie décisionnelle** :

1. 🚑 **Priorité Urgence**  
   - Si un véhicule d’urgence est détecté → **VERT immédiat**
2. 🚦 **Priorité Flux**  
   - Une voie vide et l’autre encombrée → basculement du feu
3. 📈 **Priorité Densité**  
   - Le vert est donné à la file la plus longue

---

## 🎯 Objectifs du Projet
- Réduire les embouteillages
- Donner la priorité aux urgences
- Tester une approche **IA + Vision + Simulation**
- Base pour des systèmes de **Smart City**

---

## 🛠️ Technologies Utilisées
- **Python**
- **SUMO & TraCI**
- **OpenCV**
- **YOLOv8 (Segmentation)**
- **NumPy**


