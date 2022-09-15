import random
import pygame
import numpy as np
from gym import spaces
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 200


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


class Narrow:
    def __init__(self):
        pygame.init()
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
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Box(-high, high, dtype=np.float32)
        self.screen = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
        self.player = Player()
        self.player.state = np.array([self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2])
        self.big_wall1 = Wall(0, 0, 1600, 20)
        self.big_wall2 = Wall(0, 180, 1600, 20)
        self.big_wall3 = Wall(0, 0, 20, 200)
        self.big_wall4 = Wall(1580, 0, 20, 200)
        self.walls = pygame.sprite.Group()
        self.walls.add(self.big_wall1)
        self.walls.add(self.big_wall2)
        self.walls.add(self.big_wall3)
        self.walls.add(self.big_wall4)

    def reset(self):
        self.player.state = np.array([20, self.SCREEN_HEIGHT/2])
        return self.player.state - np.array([20, self.SCREEN_HEIGHT/2])

    def step(self, act):
        x_pre_state = self.player.rect[0]
        y_pre_state = self.player.rect[1]
        self.player.state = self.player.state - np.array([-1, act*5])
        self.player.update_rect()
        for args in self.walls:
            if pygame.sprite.collide_rect(self.player, args):

                x_state = self.player.rect[0]
                y_state = self.player.rect[1]
                self.player.state = self.player.state + np.array([-1, act*5])

                if y_pre_state != y_state:
                    act = -act
                else:
                    self.close()
                    exit(0)
                self.player.state = self.player.state - np.array([-1, act*5])
                break
                # collide when this change
        reward = 0
        info = {}
        return self.player.state - np.array([20, self.SCREEN_HEIGHT/2]), reward, info

    def render(self):
        for event in pygame.event.get():
        # check if the event is the X button
            if event.type == pygame.QUIT:
                # if it is quit the game
                self.close()
                exit(0)
        self.screen.fill((255, 255, 255))
        for args in self.walls:
            self.screen.blit(args.surf, args.rect)
        self.screen.blit(self.player.surf, self.player.rect)
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

"""
plane = Narrow()
plane.reset()
# Run until the user asks to quit
running = True

i = 1

while running:

    action = (np.random.rand()*10)-5

    plane.step(action)

    plane.render()

# Done! Time to quit.

plane.close()
"""



