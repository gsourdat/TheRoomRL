import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Process
from DQNTR import *

def worker_process(worker_id, policy_net, target_net, memory, queue, config):
    game = Game()
    device = torch.device("cpu")  # Chaque worker utilise le CPU
    policy_net.to(device)
    target_net.to(device)

    def preprocess_state(state):
        state = state.astype(np.float32)
        state /= state.max() if state.max() != 0 else 1
        return torch.tensor(state.flatten(), dtype=torch.float32).to(device)

    def select_action(state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 3)
        with torch.no_grad():
            q_values = policy_net(state)
            action = torch.argmax(q_values).item()
        return action

    for episode in range(config["episodes_per_worker"]):
        state, _ = game.reset()
        state = preprocess_state(state)
        total_reward = 0
        done = False

        for _ in range(config["max_moves"]):
            action = select_action(state, config["epsilon"])
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            dx, dy = moves[action]
            new_x = game.player_pos[0] + dx
            new_y = game.player_pos[1] + dy

            if not (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE):
                reward = config["invalid_move_penalty"]
                next_state = state
                done = False
            else:
                (next_state, _), reward, done = game.step(action)
                next_state = preprocess_state(next_state)

            shaped_reward = reward - config["step_penalty"]
            if done:
                shaped_reward += config["winning_reward"]

            memory.append((state, action, shaped_reward, next_state, done))
            queue.put((state, action, shaped_reward, next_state, done))

            state = next_state
            total_reward += shaped_reward

            if done:
                break

        if worker_id == 0 and (episode + 1) % 10 == 0:
            print(f"Worker {worker_id}, Episode {episode + 1}: Total Reward = {total_reward}")

    game.close()

def train_rl_dqn_multiprocessing():
    config = {
        "num_workers": 4,
        "episodes_per_worker": 250,
        "max_moves": 500,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.2,
        "epsilon_decay": 0.99,
        "learning_rate": 0.01,
        "batch_size": 128,
        "memory_size": 10000,
        "step_penalty": 1.0,
        "winning_reward": 0,
        "invalid_move_penalty": -100,
    }

    input_dim = GRID_SIZE * GRID_SIZE * len(BLOCK_TYPES)
    output_dim = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=config["learning_rate"])
    memory = deque(maxlen=config["memory_size"])
    queue = Queue()

    processes = []
    for worker_id in range(config["num_workers"]):
        p = Process(target=worker_process, args=(worker_id, policy_net, target_net, memory, queue, config))
        p.start()
        processes.append(p)

    def train_step():
        if len(memory) < config["batch_size"]:
            return
        batch = random.sample(memory, config["batch_size"])
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        q_values = policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = policy_net(next_states).argmax(1, keepdim=True)
            max_next_q_values = target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + (config["gamma"] * max_next_q_values * (1 - dones))

        loss = nn.MSELoss()(q_values, target_q_values)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        optimizer.step()

    reward_history = []
    for episode in range(config["num_workers"] * config["episodes_per_worker"]):
        while not queue.empty():
            memory.append(queue.get())

        train_step()

        if episode % 20 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (episode + 1) % 10 == 0:
            print(f"Global Episode {episode + 1}: Memory Size = {len(memory)}")

    for p in processes:
        p.join()

    torch.save(policy_net.state_dict(), "policy_net.pth")
    torch.save(target_net.state_dict(), "target_net.pth")

    plt.plot(reward_history)
    plt.xlabel("Épisode")
    plt.ylabel("Récompense totale")
    plt.title("Performance de l’agent DQN (Multiprocessing)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_rl_dqn_multiprocessing()