import random
import pygame
import numpy as np
from gym import spaces
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800


class Player(pygame.sprite.Sprite):
    def __init__(self):
        super(Player, self).__init__()
        self.state = np.array([SCREEN_WIDTH/2, SCREEN_HEIGHT/2])
        self.surf = pygame.Surface((10, 10))
        self.rect = self.surf.get_rect(
            center=(
                self.state[0],
                self.state[1],
            )
        )

    def update_rect(self):
        self.rect = self.surf.get_rect(
            center=(
                self.state[0],
                self.state[1],
            )
        )


class Wall(pygame.sprite.Sprite):
    def __init__(self, position_x, position_y, width, height):
        super(Wall, self).__init__()
        self.surf = pygame.Surface((width, height))
        self.rect = self.surf.get_rect(
            center=(
                position_x + width/2,
                position_y + height/2,
            )
        )


class WallPlane:
    def __init__(self):

        # Set up the drawing window
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        high = np.array(
            [
                self.SCREEN_WIDTH / 2 - 20,
                self.SCREEN_HEIGHT / 2 - 20,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        high = np.array(
            [
                1,
                1,
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Box(-high, high, dtype=np.float32)
        self.screen = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
        self.player = Player()
        self.player.state = np.array([self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2])
        self.big_wall1 = Wall(0, 0, 800, 20)
        self.big_wall2 = Wall(0, 780, 800, 20)
        self.wall_3 = Wall(100, 200, 200, 20)
        self.wall_5 = Wall(0, 300, 300, 20)
        self.wall_6 = Wall(500, 300, 300, 20)
        self.wall_9 = Wall(0, 400, 200, 20)
        self.wall_10 = Wall(500, 500, 300, 20)
        self.wall_11 = Wall(0, 600, 300, 20)

        self.big_wall3 = Wall(0, 0, 20, 800)
        self.big_wall4 = Wall(780, 0, 20, 800)
        self.wall_1 = Wall(300, 200, 20, 120)
        self.wall_2 = Wall(500, 100, 20, 200)
        self.wall_4 = Wall(600, 100, 20, 200)
        self.wall_7 = Wall(500, 500, 20, 200)
        self.wall_8 = Wall(600, 600, 20, 200)

        self.walls_1 = pygame.sprite.Group()
        self.walls_2 = pygame.sprite.Group()

        self.walls_2.add(self.wall_1)
        self.walls_2.add(self.wall_2)
        self.walls_1.add(self.wall_3)
        self.walls_2.add(self.wall_4)
        self.walls_1.add(self.wall_5)
        self.walls_1.add(self.wall_6)
        self.walls_2.add(self.wall_7)
        self.walls_2.add(self.wall_8)
        self.walls_1.add(self.wall_9)
        self.walls_1.add(self.wall_10)
        self.walls_1.add(self.wall_11)

        self.walls_1.add(self.big_wall1)
        self.walls_1.add(self.big_wall2)
        self.walls_2.add(self.big_wall3)
        self.walls_2.add(self.big_wall4)

    def reset(self):
        self.player.state = np.array([self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2])
        return self.player.state - np.array([self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2])

    def step(self, act):
        x_pre_state = self.player.state[0]
        y_pre_state = self.player.state[1]
        self.player.state = self.player.state + act
        self.player.update_rect()
        for args in self.walls_1:
            if pygame.sprite.collide_rect(self.player, args):
                y_state = self.player.rect[1]
                self.player.state[1] = self.player.state[1] - act[1]
                if y_pre_state != y_state:
                    act[1] = -act[1]
                self.player.state[1] = self.player.state[1] + act[1]
                break

        for args in self.walls_2:
            if pygame.sprite.collide_rect(self.player, args):
                x_state = self.player.rect[0]
                self.player.state[0] = self.player.state[0] - act[0]
                if x_pre_state != x_state:
                    act[0] = -act[0]
                self.player.state[0] = self.player.state[0] + act[0]
                break
                # collide when this change

        self.player.update_rect()
        x_state = self.player.state[0]
        y_state = self.player.state[1]
        reward = -0.1
        if (self.player.state[0] > 750) | (self.player.state[1] > 750):
            reward = 10
        # print(reward)
        info = {}
        return self.player.state - np.array([self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2]), reward, info

    def render(self):
        pygame.init()
        for event in pygame.event.get():
        # check if the event is the X button
            if event.type == pygame.QUIT:
                # if it is quit the game
                self.close()
                exit(0)
        self.screen.fill((255, 255, 255))
        for args in self.walls_1:
            self.screen.blit(args.surf, args.rect)
        for args in self.walls_2:
            self.screen.blit(args.surf, args.rect)
        self.screen.blit(self.player.surf, self.player.rect)
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

"""
plane = WallPlane()
plane.reset()
# Run until the user asks to quit
running = True

i = 1
state = [0, 0]
while running:

    action = (np.random.rand(2)*10)-5
    print("action", action)
    print(state)
    state, reward, info = plane.step(action)
    print(reward)
    print(state)
    plane.render()

# Done! Time to quit.

plane.close()


"""







