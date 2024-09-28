import numpy as np
import random
import pygame
import time

class Autobot:
    def __init__(self, id, start, end, grid_size, obstacles):
        self.id = id
        self.start = start
        self.current = start
        self.end = end
        self.grid_size = grid_size
        self.obstacles = obstacles
        self.actions = ["up", "down", "left", "right", "wait"]
        self.q_table = np.zeros((grid_size[0], grid_size[1], len(self.actions)))
        self.waiting = False
        self.total_time = 0
        self.total_steps = 0
        self.waiting_time = 0
        self.previous_states = []
        self.movements = {
            "forward": 0,
            "backward": 0,
            "right": 0,
            "left": 0,
            "wait": 0
        }
        self.total_movements = 0

    def reset(self):
        self.current = self.start
        self.waiting = False
        self.total_time = 0
        self.total_steps = 0
        self.waiting_time = 0
        self.previous_states = []
        self.movements = {key: 0 for key in self.movements}
        self.total_movements = 0

    def get_action(self, epsilon, other_autobots):
        if np.random.rand() < epsilon:
            return random.choice(self.actions)
        else:
            x, y = self.current
            valid_actions = self.get_valid_actions(other_autobots)
            if not valid_actions:
                return "wait"
            return max(
                valid_actions, key=lambda a: self.q_table[x, y, self.actions.index(a)]
            )

    def get_valid_actions(self, other_autobots):
        x, y = self.current
        valid_actions = []
        for action in self.actions:
            new_x, new_y = self.get_new_position(action)
            if self.is_valid_move(new_x, new_y) and not self.will_collide(
                    new_x, new_y, other_autobots
            ):
                valid_actions.append(action)
        return valid_actions

    def get_new_position(self, action):
        x, y = self.current
        if action == "right":
            return x, y + 1
        elif action == "left":
            return x, y - 1
        elif action == "down":
            return x + 1, y
        elif action == "up":
            return x - 1, y
        else:  # wait
            return x, y

    def is_valid_move(self, x, y):
        return (
                0 <= x < self.grid_size[0]
                and 0 <= y < self.grid_size[1]
                and (x, y) not in self.obstacles
        )

    def will_collide(self, x, y, other_autobots):
        for autobot in other_autobots:
            if autobot.id != self.id and autobot.current == (x, y):
                if (x, y) == self.end and (x, y) == autobot.end and autobot.current == autobot.end:
                    continue
                else:
                    return True
        return False

    def move(self, action, other_autobots):
        self.total_time += 1
        self.total_steps += 1
        self.total_movements += 1

        if action == "wait":
            self.waiting = True
            self.waiting_time += 1
            self.movements["wait"] += 1
        else:
            new_x, new_y = self.get_new_position(action)
            if self.is_valid_move(new_x, new_y) and not self.will_collide(new_x, new_y, other_autobots):
                self.previous_states.append(self.current)
                self.current = (new_x, new_y)
                self.waiting = False
                if action == "up":
                    self.movements["forward"] += 1
                elif action == "down":
                    self.movements["backward"] += 1
                elif action == "right":
                    self.movements["right"] += 1
                elif action == "left":
                    self.movements["left"] += 1
            else:
                self.waiting = True
                self.waiting_time += 1
                self.movements["wait"] += 1

    def backtrack(self):
        if self.previous_states:
            self.current = self.previous_states.pop()
            self.movements["backward"] += 1
            self.total_movements += 1

    def learn(self, old_state, action, reward, new_state, alpha, gamma):
        x_old, y_old = old_state
        x_new, y_new = new_state
        action_idx = self.actions.index(action)

        self.q_table[x_old, y_old, action_idx] = self.q_table[
                                                     x_old, y_old, action_idx
                                                 ] + alpha * (
                                                         reward
                                                         + gamma * np.max(self.q_table[x_new, y_new])
                                                         - self.q_table[x_old, y_old, action_idx]
                                                 )

class Simulation:
    def __init__(self, grid_size, autobots, obstacles):
        self.grid_size = grid_size
        self.autobots = autobots
        self.obstacles = obstacles
        self.total_system_time = 0
        self.cell_size = 60  # Reduced cell size for larger grids
        self.width = grid_size[1] * self.cell_size
        self.height = grid_size[0] * self.cell_size
        self.colors = {
            'background': (255, 255, 255),
            'grid': (200, 200, 200),
            'obstacle': (100, 100, 100),
            'autobot': self.generate_distinct_colors(len(autobots)),
            'destination': (255, 255, 0)
        }
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Autobot Simulation")
        self.font = pygame.font.Font(None, 24)  # Reduced font size

    def generate_distinct_colors(self, n):
        colors = []
        for i in range(n):
            hue = i / n
            rgb = tuple(int(x * 255) for x in self.hsv_to_rgb(hue, 0.8, 0.8))
            colors.append(rgb)
        return colors

    def hsv_to_rgb(self, h, s, v):
        if s == 0.0:
            return (v, v, v)
        i = int(h * 6.)
        f = (h * 6.) - i
        p, q, t = v * (1. - s), v * (1. - s * f), v * (1. - s * (1. - f))
        i %= 6
        if i == 0:
            return (v, t, p)
        if i == 1:
            return (q, v, p)
        if i == 2:
            return (p, v, t)
        if i == 3:
            return (p, q, v)
        if i == 4:
            return (t, p, v)
        if i == 5:
            return (v, p, q)

    def run_episode(self, alpha, gamma, epsilon):
        for autobot in self.autobots:
            autobot.reset()

        total_reward = 0
        max_steps = 100
        all_reached = False

        for _ in range(max_steps):
            if all_reached:
                break

            actions = {}
            for autobot in self.autobots:
                if autobot.current != autobot.end:
                    actions[autobot.id] = autobot.get_action(
                        epsilon, [a for a in self.autobots if a.id != autobot.id]
                    )
                else:
                    actions[autobot.id] = "wait"

            all_reached = True
            for autobot in self.autobots:
                if autobot.current != autobot.end:
                    all_reached = False
                    old_state = autobot.current
                    action = actions[autobot.id]
                    autobot.move(
                        action, [a for a in self.autobots if a.id != autobot.id]
                    )
                    new_state = autobot.current

                    reward = self.get_reward(autobot, old_state, new_state)
                    total_reward += reward

                    autobot.learn(old_state, action, reward, new_state, alpha, gamma)

        self.total_system_time = max(autobot.total_time for autobot in self.autobots)
        return total_reward

    def get_reward(self, autobot, old_state, new_state):
        if new_state == autobot.end:
            return 100
        elif new_state in self.obstacles:
            return -100
        elif new_state == old_state:  # Autobot waited
            return -5
        else:
            return -1

    def train(self, episodes, alpha, gamma, epsilon):
        for episode in range(episodes):
            total_reward = self.run_episode(alpha, gamma, epsilon)
            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")

    def print_grid(self, step):
        grid = np.full(self.grid_size, ".", dtype="<U2")

        for obstacle in self.obstacles:
            grid[obstacle] = "#"
        for autobot in self.autobots:
            grid[autobot.end] = f"E{autobot.id}"

        for autobot in self.autobots:
            grid[autobot.current] = autobot.id

        print(f"\n{'=' * 50}")
        print(f"Step {step + 1}:")
        print(f"{'-' * 50}")
        for row in grid:
            print(" ".join(row))
        print(f"{'-' * 50}")

    def display_grid(self, step):
        self.screen.fill(self.colors['background'])

        # Draw grid
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, self.colors['grid'], (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, self.colors['grid'], (0, y), (self.width, y))

        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.colors['obstacle'],
                             (obstacle[1] * self.cell_size, obstacle[0] * self.cell_size,
                              self.cell_size, self.cell_size))

        # Draw Autobots and their destinations
        for i, autobot in enumerate(self.autobots):
            # Draw destination
            pygame.draw.rect(self.screen, self.colors['destination'],
                             (autobot.end[1] * self.cell_size, autobot.end[0] * self.cell_size,
                              self.cell_size, self.cell_size), 3)

            # Draw Autobot
            pygame.draw.circle(self.screen, self.colors['autobot'][i],
                               (autobot.current[1] * self.cell_size + self.cell_size // 2,
                                autobot.current[0] * self.cell_size + self.cell_size // 2),
                               self.cell_size // 3)

            # Draw Autobot ID
            id_text = self.font.render(autobot.id, True, (0, 0, 0))
            self.screen.blit(id_text, (autobot.current[1] * self.cell_size + self.cell_size // 2 - 5,
                                       autobot.current[0] * self.cell_size + self.cell_size // 2 - 5))

        # Display step number
        step_text = self.font.render(f"Step: {step + 1}", True, (0, 0, 0))
        self.screen.blit(step_text, (10, 10))

    def run_inference(self):
        for autobot in self.autobots:
            autobot.reset()

        all_reached = False
        step = 0
        running = True

        while running and not all_reached and step < 100:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            actions = {}
            for autobot in self.autobots:
                if autobot.current != autobot.end:
                    actions[autobot.id] = autobot.get_action(
                        0, [a for a in self.autobots if a.id != autobot.id]
                    )
                else:
                    actions[autobot.id] = "wait"

            all_reached = True
            for autobot in self.autobots:
                if autobot.current != autobot.end:
                    all_reached = False
                    action = actions[autobot.id]
                    autobot.move(
                        action, [a for a in self.autobots if a.id != autobot.id]
                    )
                    if autobot.waiting:
                        autobot.backtrack()

                if autobot.total_movements >= 25 and autobot.current != autobot.end:
                    print(f"\n{'=' * 50}")
                    print(f"Autobot {autobot.id} couldn't reach its destination in 25 movements.")
                    print(f"Terminating the session.")
                    print(f"{'=' * 50}\n")
                    self.print_statistics()
                    pygame.quit()
                    return

            self.display_grid(step)
            self.print_grid(step)
            pygame.display.flip()
            time.sleep(1)

            step += 1

        if all_reached:
            print(f"\n{'=' * 50}")
            print("All Autobots have reached their destinations!")
            print(f"{'=' * 50}\n")
        else:
            print(f"\n{'=' * 50}")
            print("Maximum steps reached. Some Autobots may not have reached their destinations.")
            print(f"{'=' * 50}\n")

        self.print_statistics()
        pygame.quit()

    def print_statistics(self):
        print(f"\n{'=' * 50}")
        print("Simulation Statistics:")
        print(f"{'-' * 50}")
        total_movements = 0
        for autobot in self.autobots:
            print(f"Autobot {autobot.id}:")
            print(f"  Total Time: {autobot.total_time}")
            print(f"  Total Steps: {autobot.total_steps}")
            print(f"  Waiting Time: {autobot.waiting_time}")
            print(f"  Movements:")
            for movement, count in autobot.movements.items():
                print(f"    {movement.capitalize()}: {count}")
            print(f"  Total Movements: {autobot.total_movements}")
            print(f"{'-' * 50}")
            total_movements += autobot.total_movements

        avg_movements = total_movements / len(self.autobots)
        print(f"Average Movements per Autobot: {avg_movements:.2f}")
        print(f"{'=' * 50}\n")

# Example 2 & 3: Gridlock (Traffic Jam) + Backtraking
grid_size = (5, 5)
obstacles = [(1, 0), (1, 2)]  # Narrow path, causing potential gridlock

autobots = [
    Autobot("A", (0, 1), (4, 1), grid_size, obstacles),
    Autobot("B", (4, 1), (0, 1), grid_size, obstacles),
    Autobot("C", (2, 1), (2, 1), grid_size, obstacles),  # This bot is in the middle
]

simulation = Simulation(grid_size, autobots, obstacles)

# Train the Autobots
simulation.train(episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1)

# Run inference with Pygame visualization and terminal output
simulation.run_inference()
