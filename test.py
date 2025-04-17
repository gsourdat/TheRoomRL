from DQNTR import *

def test_saved_model(device_choice="auto"):
    # Déterminer le périphérique en fonction du choix de l'utilisateur
    if device_choice == "cpu":
        device = torch.device("cpu")
    elif device_choice == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    
    
    game = Game()
    input_dim = GRID_SIZE * GRID_SIZE * len(BLOCK_TYPES) + 2 +1
    output_dim = 4
    train_net = DQN(input_dim, output_dim).to(device)
    train_net.load_state_dict(torch.load("models/policy_net4.pth"))
    train_net.eval()

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
            q_values = train_net(state)
            return torch.argmax(q_values).item()
        
    state, player_pos = game.reset()
    state = preprocess_state(state, player_pos)
    total_reward = 0
    done = False
    clock = pygame.time.Clock()

    while not done:
        with torch.no_grad():
            q_values = train_net(state)
            action = select_action(state, 0.2)  # epsilon = 0 pour le test

        (next_state, player_pos), done = game.step(action)
        print(f"Action: {action}, Reward: {game.score}, Done: {done}")
        next_state = preprocess_state(next_state, player_pos)
        state = next_state
        total_reward = game.final_score
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.close()
                return

        game.render()
        pygame.time.wait(50)
        clock.tick(30)

    print(f"Test terminé : Récompense totale = {total_reward}")
    game.close()

if __name__ == "__main__":
    test_saved_model()
