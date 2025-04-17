import pygame
import random
import numpy as np

# Initialize Pygame
pygame.init()

# Grid settings
GRID_SIZE = 10
BLOCK_SIZE = 30
WINDOW_SIZE = GRID_SIZE * BLOCK_SIZE

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GOLD = (255, 215, 0)
GREY = (128, 128, 128)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

# Block types
BLOCK_TYPES = {
    'white': {'color': WHITE, 'base_reward': 0},
    'red': {'color': RED, 'base_reward': -1},
    'blue': {'color': BLUE, 'base_reward': 1},
    'golden': {'color': GOLD, 'base_reward': 5},
    'black': {'color': BLACK, 'base_reward': 10, 'risk': 0.05},  # 10% chance to die
    'spawn': {'color': GREY, 'base_reward': 0},
    'exit': {'color': GREY, 'base_reward': 0}
}

abbr_to_color = {
    'w': 'white',
    'r': 'red',
    'b': 'blue',
    'g': 'golden',
    'B': 'black'
}

def lire_table_et_convertir(nom_fichier):
    table = []
    with open(nom_fichier, 'r') as fichier:
        for ligne in fichier:
            abbrs = ligne.strip().split()  # split sans virgule
            couleurs = [abbr_to_color.get(a, a) for a in abbrs]
            table.append(couleurs)
    return table




class Game:
    def __init__(self, render=True):
        self.rendering = render
        if render:
            print("there is a render")
            self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            pygame.display.set_caption("RL Mini Game")
        self.reset()

    def reset(self, spawn=None, exit=None):
        # Initialize grid
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=object)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                self.grid[x][y] = random.choice(['white', 'red', 'blue', 'golden', 'black'])

        corners = [(0, 0), (0, GRID_SIZE-1), (GRID_SIZE-1, 0), (GRID_SIZE-1, GRID_SIZE-1)]
        self.spawn = random.choice(corners)
        self.grid[self.spawn] = 'spawn'

        # Set exit to opposite corner
        opposite = {
                (0, 0): (GRID_SIZE-1, GRID_SIZE-1),
                (0, GRID_SIZE-1): (GRID_SIZE-1, 0),
                (GRID_SIZE-1, 0): (0, GRID_SIZE-1),
                (GRID_SIZE-1, GRID_SIZE-1): (0, 0)
        }
        self.exit = opposite[self.spawn]
        self.grid[self.exit] = 'exit'
        
        if(spawn is not None):
            self.spawn = spawn
            self.grid[self.spawn] = 'spawn'
        if(exit is not None):
            self.exit = exit
            self.grid[self.exit] = 'exit'


        # Set player position
        self.player_pos = list(self.spawn)
        self.score = 0 #score du mouvement
        self.moves = 0
        self.final_score = 0 #score total
        self.done = False
        return self.get_state()

    def reset2(self):
        # Grille aléatoire fixée directement dans le code
        self.grid = np.array(lire_table_et_convertir("map.txt"), dtype=object); 
        

        # Fixer les positions de spawn et de sortie
        self.spawn = (0, 0)  # Position de départ fixe
        self.exit = (GRID_SIZE-1, GRID_SIZE-1)  # Position de sortie fixe
        self.grid[self.spawn] = 'spawn'
        self.grid[self.exit] = 'exit'

        # Initialiser la position du joueur
        self.player_pos = list(self.spawn)
        self.rewards = []
        self.score = 0
        self.moves = 0
        self.final_score = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        # Return grid and player position as state
        state = np.zeros((GRID_SIZE, GRID_SIZE, len(BLOCK_TYPES)))
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                block_type = self.grid[x][y]
                idx = list(BLOCK_TYPES.keys()).index(block_type)
                state[x, y, idx] = 1
        return state, self.player_pos

    def step(self, action):
        if self.done:
            return self.get_state(), True

        # Actions: 0=up, 1=down, 2=left, 3=right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = moves[action]
        new_x, new_y = self.player_pos[0] + dx, self.player_pos[1] + dy

        # Check boundaries
        if not (0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE):
            # Invalid move, return current state without penalty
            return self.get_state(), self.done

        self.moves += 1

        self.player_pos = [new_x, new_y]
        block_type = self.grid[new_x][new_y]

        # Handle block rewards
        if block_type == 'black':
            if random.random() < BLOCK_TYPES['black']['risk']:
                self.score = -1
                self.player_pos = list(self.spawn)
            else:
                self.score = BLOCK_TYPES['black']['base_reward'] -1
                self.final_score += self.score
                self.grid[new_x][new_y] = 'white'  # Mark as visited
        elif block_type != 'spawn' and block_type != 'exit':
            self.score = BLOCK_TYPES[block_type]['base_reward'] -1
            self.final_score += self.score
            self.grid[new_x][new_y] = 'white'  # Mark as visited
        
        
        # Check if reached exit
        if self.player_pos == list(self.exit):
            self.done = True
        return self.get_state(), self.done

    def render(self):
        self.screen.fill(BLACK)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                block_type = self.grid[x][y]
                color = BLOCK_TYPES[block_type]['color']
                pygame.draw.rect(self.screen, color,
                               (y * BLOCK_SIZE, x * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.screen, BLACK,
                               (y * BLOCK_SIZE, x * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)

        # Draw player
        px, py = self.player_pos
        pygame.draw.circle(self.screen, (0, 255, 0),
                          (py * BLOCK_SIZE + BLOCK_SIZE // 2, px * BLOCK_SIZE + BLOCK_SIZE // 2),
                          BLOCK_SIZE // 3)

        # Display score and moves
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f'Score: {self.final_score}', True, GREEN)
        moves_text = font.render(f'Moves: {self.moves}', True, GREEN)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(moves_text, (10, 50))

        pygame.display.flip()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.step(0)
                elif event.key == pygame.K_DOWN:
                    self.step(1)
                elif event.key == pygame.K_LEFT:
                    self.step(2)
                elif event.key == pygame.K_RIGHT:
                    self.step(3)
                elif event.key == pygame.K_ESCAPE:
                    return False
        return True

    def close(self):
        pygame.quit()

# Human-playable game loop
def play_game():
    game = Game()
    running = True
    clock = pygame.time.Clock()

    while running:
        running = game.handle_input()
        game.render()
        if game.done:
            print(f"Game Over! Final Score: {game.final_score}, Moves: {game.moves}")
            running = False
        clock.tick(60)
    game.close()


if __name__ == "__main__":
    # Uncomment to play manually
    play_game()
    # Uncomment to run RL training
