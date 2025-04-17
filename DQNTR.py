import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from TheRoom import *
import time  # Importer la bibliothèque pour le chronométrage
import argparse  # Importer argparse pour les arguments de ligne de commande

torch.backends.cudnn.benchmark = True

# Réseau neuronal
class DQN2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN2, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)  # Nouvelle couche ajoutée
        self.fc4 = nn.Linear(256, output_dim)  # Mise à jour de la dernière couche

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))  # Nouvelle couche activée
        return self.fc4(x)
    
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)  # Mise à jour de la dernière couche

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Entraînement RL
def train_rl_dqn(resume_modele=None, device_choice="auto"):
    game = Game(render=False)  # Initialiser le jeu sans rendu

    episodes = 1000 # Nombre total d'épisodes d'entraînement. Plus ce nombre est élevé, plus l'agent a de chances d'apprendre.
    max_moves = 1000 # Nombre maximum de mouvements autorisés par épisode. Cela limite la durée d'un épisode.
    gamma = 0.99 # Facteur de discount (ou taux d'actualisation). Il détermine l'importance des récompenses futures par rapport aux récompenses immédiates. Une valeur proche de 1 favorise les récompenses à long terme.
    epsilon = 1.0 # Valeur initiale de l'exploration epsilon-greedy. Un epsilon élevé signifie que l'agent choisit des actions aléatoires au début de l'entraînement.
    epsilon_min = 0.2 # Valeur minimale d'epsilon. Cela garantit que l'agent continue d'explorer un peu même après un long entraînement.
    epsilon_decay = 0.999 # Facteur de décroissance d'epsilon. À chaque épisode, epsilon est multiplié par ce facteur pour réduire progressivement l'exploration.
    learning_rate = 0.0005 # Taux d'apprentissage pour l'optimiseur Adam. Il contrôle la vitesse à laquelle le modèle ajuste ses poids.
    batch_size = 256 # Taille du batch utilisé pour l'entraînement. Cela détermine combien d'échantillons sont utilisés pour calculer les gradients à chaque étape d'entraînement.
    memory_size = 10000000 # Taille maximale de la mémoire de replay. Cela limite le nombre de transitions stockées pour l'entraînement.
    step_penalty = 1.0 # Pénalité appliquée à chaque mouvement. Cela encourage l'agent à atteindre son objectif en un minimum de mouvements.
    winning_reward = 1000 # Récompense supplémentaire donnée lorsque l'agent atteint l'objectif. Vous pouvez augmenter cette valeur pour encourager l'agent à gagner.
    invalid_move_penalty = 10 # Pénalité appliquée lorsque l'agent tente un mouvement invalide (par exemple, sortir des limites de la grille). Cela décourage les actions inutiles.
    stop_penalty = 1 # Pénalité appliquée lorsque l'agent reste dans le même état. Cela encourage l'agent à explorer de nouveaux états.

    input_dim = GRID_SIZE * GRID_SIZE * len(BLOCK_TYPES) + 2 + 1
    output_dim = 4

    N_STEP = 10  # Par exemple, 5-step

    # Mémoire pour stocker sur n étapes
    n_step_buffer = deque(maxlen=N_STEP)

    def store_n_step_transition(state, action, reward, next_state, done):
        """Ajoute la transition dans un buffer n-step."""
        n_step_buffer.append((state, action, reward, next_state, done))
        if len(n_step_buffer) == N_STEP:
            # Calcul du retour n-step
            R = 0
            for i, (s, a, r, ns, d) in enumerate(n_step_buffer):
                R += (gamma ** i) * r
                if d:
                    # Si l'état est terminal, on arrête le calcul du retour
                    break
            # On utilise le dernier next_state et done pour l’état terminal
            final_state, final_done = n_step_buffer[-1][3], n_step_buffer[-1][4]
            memory.append((n_step_buffer[0][0],  # État initial
                           n_step_buffer[0][1],  # Action initiale
                           R,                    # Retour n‑step
                           final_state,
                           final_done))

    # Déterminer le périphérique en fonction du choix de l'utilisateur
    if device_choice == "cpu":
        device = torch.device("cpu")
    elif device_choice == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    policy_net = DQN(input_dim, output_dim).to(device)
    if(resume_modele is not None):
        policy_net.load_state_dict(torch.load(resume_modele))
    target_net = DQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = deque(maxlen=memory_size)

    reward_history = []
    
    def preprocess_state(state, player_pos):
        state = state.astype(np.float32)
        state /= state.max() if state.max() != 0 else 1

        # Normaliser la position du joueur
        player_pos_normalized = np.array(player_pos, dtype=np.float32) / GRID_SIZE

        # Nouvelle feature : proximité de la sortie (<= 2 cases en distance de Manhattan)
        manhattan_distance = abs(player_pos[0] - game.exit[0]) + abs(player_pos[1] - game.exit[1])
        is_near_exit = np.array([1.0 if manhattan_distance <= 2 else 0.0], dtype=np.float32)

        # Concatenation de l’état : [grille flatten] + [position joueur] + [proche de la sortie]
        state = np.concatenate([state.flatten(), player_pos_normalized, is_near_exit])

        return torch.tensor(state, dtype=torch.float32, device=device)
    



    def select_action(state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 3)
        with torch.no_grad():
            q_values = policy_net(state)
            return torch.argmax(q_values).item()



    def train_step():
        
        if len(memory) < batch_size or len(n_step_buffer) < N_STEP:
                return
        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        q_values = policy_net(states).gather(1, actions)

        #with torch.no_grad():
        #    next_actions = policy_net(next_states).argmax(1, keepdim=True)
        #    max_next_q_values = target_net(next_states).gather(1, next_actions)
        #    target_q_values = rewards + (gamma * max_next_q_values * (1 - dones))

        #loss = nn.MSELoss()(q_values, target_q_values)
        
        # Q(s', a') — Double DQN
        with torch.no_grad():
            next_q_values = target_net(next_states)
            best_actions = policy_net(next_states).argmax(1, keepdim=True)
            target_q_values = rewards + gamma * next_q_values.gather(1, best_actions) * (1 - dones)

        # Perte
        loss = nn.functional.mse_loss(q_values, target_q_values)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        optimizer.step()

    def getSpawnAndExit(n):
        corners = [(0, 0), (0, GRID_SIZE - 1), (GRID_SIZE - 1, 0), (GRID_SIZE - 1, GRID_SIZE - 1)]
        chosen_exit = random.choice(corners)
        exit_x, exit_y = chosen_exit

        # Calcul de spawn en fonction de exit ± n
        if exit_x == 0:
            spawn_x = exit_x + n
        else:
            spawn_x = exit_x - n

        if exit_y == 0:
            spawn_y = exit_y + n
        else:
            spawn_y = exit_y - n

        spawn = (spawn_x, spawn_y)
        return spawn, chosen_exit
            
    # Chronométrage de l'entraînement
    start_time = time.time()
    victories = 0
    for episode in range(episodes):
        if episode < 100:
            spawn, exit = getSpawnAndExit(1)
            state, player_pos = game.reset(spawn=spawn, exit=exit)        
        elif episode < 200:
            spawn, exit = getSpawnAndExit(3)
            state, player_pos = game.reset(spawn=spawn, exit=exit) 
        elif episode < 300:
            spawn, exit = getSpawnAndExit(5)
            state, player_pos = game.reset(spawn=spawn, exit=exit)
        elif episode < 400:
            spawn, exit = getSpawnAndExit(7)
            state, player_pos = game.reset(spawn=spawn, exit=exit)
        else:
            state, player_pos = game.reset()
        state = preprocess_state(state, player_pos)
        total_reward = 0
        done = False

        # Calculer la distance initiale de Manhattan
        exit_pos = game.exit  
        #prev_distance = abs(player_pos[0] - exit_pos[0]) + abs(player_pos[1] - exit_pos[1])

        for i in range(max_moves):
            action = select_action(state, epsilon)
            reward = game.score  # Récompense actuelle du jeu
            reward -= step_penalty  # Appliquer la pénalité de pas
            
            
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
            dx, dy = moves[action]
            new_x = player_pos[0] + dx
            new_y = player_pos[1] + dy
            
            if not (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE):
                reward -= invalid_move_penalty  # Appliquer la pénalité pour mouvement invalide
            
            (next_state, player_pos), done = game.step(action)
            next_state = preprocess_state(next_state, player_pos)

            # Calculer la nouvelle distance de Manhattan
            #current_distance = abs(player_pos[0] - exit_pos[0]) + abs(player_pos[1] - exit_pos[1])

            # Appliquer une pénalité si la distance n'a pas diminué
            #if current_distance >= prev_distance:
                #reward -= 1  # Ajustez la valeur de la pénalité selon vos besoins

            #prev_distance = current_distance  # Mettre à jour la distance précédente

            # Façonner la récompense
            if done:
                reward += winning_reward

            # Pénalité pour rester dans le même état
            if np.array_equal(state.cpu().numpy(), next_state.cpu().numpy()):
                reward -= stop_penalty  # Appliquer la pénalité pour rester dans le même état

            # Ajouter la transition à la mémoire
            store_n_step_transition(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done and player_pos == list(game.exit):
                victories += 1
            
            train_step()

            if done:
                n_step_buffer.clear()
                break

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if (episode + 1) % 50 == 0:
            print(f"Épisode {episode+1}: Victoires cumulées = {victories}")
            victories = 0

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        reward_history.append(total_reward)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {game.moves}, Winning = {done}")

    # Fin du chronométrage
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")

    torch.save(policy_net.state_dict(), "policy_net4.pth")
    torch.save(target_net.state_dict(), "target_net4.pth")
    game.close()

    # Tracer les récompenses
    plt.plot(reward_history)
    plt.xlabel("Épisode")
    plt.ylabel("Récompense totale")
    plt.title("Performance de l’agent DQN")
    plt.grid(True)
    plt.show()



if __name__ == "__main__":

    train_rl_dqn(device_choice="gpu")
    #train_rl_dqn(resume_modele="policy_net2.pth", device_choice="gpu")

