"""Self-learning agents game."""

import math
import time
from itertools import count
import pygame, random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from policy_gradients import Policy

# Predefined colors
BACKGROUND = (round(0.4 * 255), round(0.4 * 255), round(0.4 * 255))
WALLS  = (0, 0, 0)
PLAYER_COLOR = (round(0.85 * 255), round(0.325 * 255), round(0.0980 * 255))
ENEMY_COLOR = (0, round(0.4470 * 255), round(0.7410 * 255))

# Screen dimensions
SCREEN_WIDTH = 320
SCREEN_HEIGHT = 240
WALL_WIDTH = int(SCREEN_WIDTH / 16)
PLAYER_SIZE = int(SCREEN_WIDTH / 16)
BULLET_SIZE = min(8, int(SCREEN_WIDTH / 32))

# Actions
LEFT = 'left'
RIGHT = 'right'
JUMP = 'jump'
SHOOT_LEFT = 'shoot_left'
SHOOT_RIGHT = 'shoot_right'
STOP = 'stop'
ACTIONS = [LEFT, RIGHT, JUMP, SHOOT_LEFT, SHOOT_RIGHT, STOP]


class Bullet(pygame.sprite.Sprite):
    """Generic bullet class."""

    def __init__(self, player, direction):
        """Constructor."""
        super().__init__()

        self.image = pygame.Surface([BULLET_SIZE, BULLET_SIZE])
        self.image.fill((255, 255, 255))
        self.rect = self.image.get_rect()

        # attribute to know which player spawned the bullet
        self.player = player
        # initial bullet direction and position
        self.direction = direction
        self.rect.x, self.rect.y = self.player.rect.x + 10, self.player.rect.y + 10

        # bullet speed
        self.change_x = 10
        self.change_y = 0

    def update(self):
        """Move bullet and update scores."""
        # update bullet position
        self.rect.x += self.direction * self.change_x
        self.rect.y += self.change_y

        # check collision with platforms in the arena
        block_hit_list = pygame.sprite.spritecollide(self,
                                                     self.player.level.platform_list,
                                                     False)
        if len(block_hit_list) > 0:
            self.player.level.bullet_list.remove(self)
            self.player.bullets.remove(self)

        # check collision with other players
        block_hit_list  = pygame.sprite.spritecollide(self,
                                                      self.player.level.player_list,
                                                      False)
        for block in block_hit_list:
            for other_player in self.player.other_players:
                if block is other_player:
                    self.player.level.bullet_list.remove(self)
                    # might have already removed
                    if self in self.player.bullets:
                        self.player.bullets.remove(self)
                    self.player.score += 1
                    other_player.score -= 1


class Player(pygame.sprite.Sprite):
    """Generic player class."""

    def __init__(self, color, position):
        """Constructor."""
        super().__init__()

        # create an image of the block, and fill it with a color
        # this could also be an image loaded from the disk
        width = PLAYER_SIZE
        height = PLAYER_SIZE
        self.image = pygame.Surface([width, height])
        self.image.fill(color)

        # set a referance to the image rect.
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = position

        # specify a random direction when spawning
        self.direction = 1 - 2 * random.randint(0, 1)

        # set speed vector of player
        self.change_x = 0
        self.change_y = 0

        # list of bullet that the player spawns
        self.bullets = []

        # list of sprites we can bump against
        self.level = None
        self.other_players = None

        # score value when hitting a player or getting hit
        self.score = 0

    def update_other_players(self):
        """Update list of other players."""
        self.other_players = [player for player in self.level.players]
        self.other_players.remove(self)

    def collision_check_x(self):
        """Check collisions along x."""
        # check collisions with platforms
        block_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
        for block in block_hit_list:
            # if we are moving right,
            # set our right side to the left side of the item we hit
            if self.change_x > 0:
                self.rect.right = block.rect.left
            elif self.change_x < 0:
                # otherwise if we are moving left, do the opposite
                self.rect.left = block.rect.right

        # check collisions with other players
        block_hit_list = pygame.sprite.spritecollide(self, self.other_players, False)
        for block in block_hit_list:
            # if we are moving right,
            # set our right side to the left side of the item we hit
            if self.change_x > 0:
                self.rect.right = block.rect.left
            elif self.change_x < 0:
                # otherwise if we are moving left, do the opposite
                self.rect.left = block.rect.right

    def collision_check_y(self):
        """Check collisions along y."""
        # check collisions with platforms
        block_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
        for block in block_hit_list:
            # reset our position based on the top/bottom of the object
            if self.change_y > 0:
                self.rect.bottom = block.rect.top
            elif self.change_y < 0:
                self.rect.top = block.rect.bottom

            # stop our vertical movement
            self.change_y = 0

        # check collisions with other players
        block_hit_list = pygame.sprite.spritecollide(self, self.other_players, False)
        for block in block_hit_list:
            # reset our position based on the top/bottom of the object
            if self.change_y > 0:
                self.rect.bottom = block.rect.top
            elif self.change_y < 0:
                self.rect.top = block.rect.bottom

            # stop our vertical movement
            self.change_y = 0

    def update(self):
        """Move the player."""
        # gravity
        self.calc_grav()

        # move left/right
        self.rect.x += self.change_x
        self.collision_check_x()

        # move up/down
        self.rect.y += self.change_y
        self.collision_check_y()

    def calc_grav(self):
        """Calculate effect of gravity."""
        if self.change_y == 0:
            self.change_y = 1
        else:
            self.change_y += .35

        # see if we are on the ground.
        if self.rect.y >= SCREEN_HEIGHT - self.rect.height and self.change_y >= 0:
            self.change_y = 0
            self.rect.y = SCREEN_HEIGHT - self.rect.height

    def calc_rel_states(self, states):
        """Calculate relative states.

        Calculate relative states (position for now) of other players, bullets
        w.r.t the player.

        Args:
            states (list of tuple): each tuple has state of player and its bullet

        Returns:
            rel_states (list of tuple): relative states (position for now)

        """
        rel_states = []
        for state_pair in states:
            player_state = state_pair[0]
            bullet_state = state_pair[1]
            player_state = (player_state[0] - self.rect.x,
                            player_state[1] - self.rect.y,
                            player_state[2],
                            player_state[3])
            if bullet_state is not None:
                bullet_state = (bullet_state[0] - self.rect.x,
                                bullet_state[1] - self.rect.y,
                                bullet_state[2],
                                bullet_state[3])
            rel_states.append((player_state, bullet_state))
        return rel_states

    def jump(self):
        """Jump."""
        # move down a bit and see if there is a platform below us
        # move down 2 pixels because it doesn't work well if we only move down
        # 1 when working with a platform moving down
        self.rect.y += 2
        platform_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
        self.rect.y -= 2

        # if it is ok to jump, set our speed upwards
        if len(platform_hit_list) > 0 or self.rect.bottom >= SCREEN_HEIGHT:
            self.change_y = -10

    def go_left(self):
        """Move left."""
        self.change_x = -6
        self.direction = -1

    def go_right(self):
        """Move right."""
        self.change_x = 6
        self.direction = 1

    def stop(self):
        """Set velocity to 0."""
        self.change_x = 0

    def shoot_left(self):
        """Shoot bullet left."""
        # currently, allow only one bullet at the time
        if  len(self.bullets) == 0 :
            bullet =  Bullet(self, -1)
            self.bullets.append(bullet)
            self.level.bullet_list.add(bullet)

    def shoot_right(self):
        """Shoot bullet right."""
        # currently, allow only one bullet at the time
        if  len(self.bullets) == 0 :
            bullet =  Bullet(self, 1)
            self.bullets.append(bullet)
            self.level.bullet_list.add(bullet)

    def act(self, action):
        """Perform action."""
        if action is not None:
            if action == LEFT:
                self.go_left()
            elif action == RIGHT:
                self.go_right()
            elif action == JUMP:
                self.jump()
            elif action == SHOOT_LEFT:
                self.shoot_left()
            elif action == SHOOT_RIGHT:
                self.shoot_right()
            elif action == STOP:
                self.stop()

    def get_inputs_from_states(self, states):
        """Use states to create input to Q-Network."""
        inputs = []
        for state_pair in states:
            player_state = state_pair[0]
            bullet_state = state_pair[1]
            if bullet_state is None:
                bullet_state = torch.zeros(4)
            inputs.append(torch.from_numpy(np.array(player_state)).float())
            inputs.append(torch.from_numpy(np.array(bullet_state)).float())
        inputs = torch.cat(inputs).unsqueeze(0)
        return inputs


class Platform(pygame.sprite.Sprite):
    """Platform the user can jump on."""

    def __init__(self, width, height):
        """Platform constructor."""
        super().__init__()

        self.image = pygame.Surface([width, height])
        self.image.fill(WALLS)

        self.rect = self.image.get_rect()


class Level(object):
    """This is a generic class used to define a level."""

    def __init__(self, players):
        """Constructor."""
        self.platform_list = pygame.sprite.Group()
        self.players = players
        for p in players:
            p.level = self
            p.update_other_players()

        self.bullet_list =  pygame.sprite.Group()
        self.player_list = pygame.sprite.Group()
        self.player_list.add(players)

        index = 0
        for player in players:
            player.index = index
            index+=1

        # background image
        self.background = None

    def update(self):
        """Update everything in this level."""
        self.platform_list.update()
        self.bullet_list.update()
        self.player_list.update()

    def get_states(self):
        """Get the states of players, bullets.

        Returns:
            (list of tuple): each tuple has state of player and its bullet

        """
        states = []
        for player in self.players:
            player_state = (player.rect.x, player.rect.y,
                            player.change_x, player.change_y)
            if player.bullets:
                bullet_state = (player.bullets[0].rect.x, player.bullets[0].rect.y,
                                player.bullets[0].change_x, player.bullets[0].change_y)
            else:
                bullet_state = None
            states.append((player_state, bullet_state))
        return states

    def get_scores(self):
        """Get the scores of the players.

        Returns:
            (list): list with score of each player

        """
        scores = []
        for player in self.players:
            scores.append(player.score)
        return scores

    def draw(self, screen):
        """Draw everything on this level."""
        # draw the background
        screen.fill(BACKGROUND)

        # draw all the sprite lists that we have
        self.platform_list.draw(screen)
        self.player_list.draw(screen)
        self.bullet_list.draw(screen)


# create platforms for the level
class SimpleLevel(Level):
    """Definition of level."""

    def __init__(self, players):
        """Create level."""
        super().__init__(players)

        # define all walls: width, height, x pos, y pos
        level = [[WALL_WIDTH, SCREEN_HEIGHT, 0, 0],
                 [WALL_WIDTH, SCREEN_HEIGHT, SCREEN_WIDTH - WALL_WIDTH, 0],
                 [SCREEN_WIDTH, WALL_WIDTH, 0, 0],
                 [SCREEN_WIDTH, WALL_WIDTH, 0, SCREEN_HEIGHT - WALL_WIDTH]]

        # go through the array above and add platforms
        for platform in level:
            block = Platform(platform[0], platform[1])
            block.rect.x = platform[2]
            block.rect.y = platform[3]
            self.platform_list.add(block)


class Game:
    """Generic class to wrap around, with interfaces to run and interact with the game."""

    def __init__(self):
        """Constructor."""
        # initiate pygame
        pygame.init()

        # set screen parameters
        self.size = [SCREEN_WIDTH,  SCREEN_HEIGHT]
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Python self-learning project")

        # control screen fps
        self.clock = pygame.time.Clock()
        self.fps = 120

        # create the players
        self.players = [Player(PLAYER_COLOR, [random.randint(40, 240), random.randint(40, 120)]),
                        Player(ENEMY_COLOR, [random.randint(40, 240), random.randint(40, 120)])]

        # create the level
        self.level = SimpleLevel(self.players)

    def step(self, actions):
        """Step through one iteration of the game."""
        for idx, player in enumerate(self.players):
            player.act(actions[idx])

        self.level.update()
        states = self.level.get_states()

        # return score of each player (which can be used as reward)
        scores = self.level.get_scores()

        for score in scores:
            if score != 0:
                done = True
            else:
                done = False

        return states, scores, done

    def render(self, mode=None):
        """Draw the sprites and update.

        Returns:
            img (numpy.ndarray): if mode is 'rgb_array', returns screen as array

        """
        events = pygame.event.get()
        self.level.draw(self.screen)

        # limit the fps
        self.clock.tick(self.fps)

        # update the screen
        pygame.display.flip()

        # return screen as rgb array
        if mode == 'rgb_array':
            img = self.screen
            img = pygame.surfarray.array3d(img).swapaxes(0, 1)
            return img

    def reset(self):
        """Reinitialise game, resetting states, scores."""
        self.__init__()

    def quit(self):
        """Quit pygame gracefully."""
        pygame.quit()


# Display scores
def add_text(screen, text, pos):
    """Add text to game screen."""
    font = pygame.font.SysFont('ptmono', 20)
    screen.blit(font.render(text, True, (255, 255, 255)), pos)


def random_action():
    """Pick random action."""
    # return random.choice(range(len(ACTIONS)))
    return random.choice([3, 2, 4, 5, 5])


def select_action(policy_net, state):
    """Select action from the policy net."""
    probs = policy_net(state)
    m = Categorical(probs)
    action = m.sample()
    policy_net.saved_log_probs.append(m.log_prob(action))
    return action.item()


def plot_scores(pl_a_scores, pl_b_scores):
    """Plot scores over episodes."""
    x = range(len(pl_a_scores))
    plt.figure()
    plt.plot(x, pl_a_scores, 'r--', label="Score-Player A")
    plt.plot(x, pl_b_scores, 'b--', label="Score-Player B")

    plt.title("Self-learning agents scores")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Score")

    plt.legend()
    plt.show()


def run_random_game():
    """Run game with randomly acting players."""
    # initialise game
    gladiator_game = Game()

    # loop over episodes
    n_episodes = 1000

    # --------main loop-----------
    for episode_idx in range(n_episodes):

        # choose action for each player
        actions = []
        for _ in range(len(gladiator_game.players)):
            actions.append(random_action())

        states, scores, done = gladiator_game.step(actions)
        # NOTE: check if relative states are correct
        # print(gladiator_game.players[1].calc_rel_states(states))

        gladiator_game.render()

    # exit gracefully
    gladiator_game.quit()


def run_policy_gradient():
    """Run game with policy-gradient learning."""
    # initialise game
    gladiator_game = Game()
    random.seed(5234)

    # score counts
    pl_a_score, pl_b_score = 0, 0
    pl_a_scores, pl_b_scores = [], []

    # initialise policy network
    # 16 inputs (states), 6 outputs (actions)
    policy_net = Policy(16, 32, len(ACTIONS))

    # batch size
    batch_size = 16

    # optimizer
    optimizer = optim.RMSprop(policy_net.parameters())

    # loop over episodes
    n_episodes = 1000

    # batch History
    state_pool = []
    action_pool = []
    reward_pool = []
    steps = 0

    # ----------episodes-----------
    for episode_idx in range(n_episodes):

        print('Running episode: {}, scores A:{}, B:{}'.format(episode_idx,
                                                              pl_a_score,
                                                              pl_b_score))
        # keep track of scores
        pl_a_scores.append(pl_a_score)
        pl_b_scores.append(pl_b_score)

        # reset the game and scores
        gladiator_game.reset()

        # get current state, relative to player 1
        current_states = gladiator_game.level.get_states()
        current_states = gladiator_game.players[0].calc_rel_states(current_states)

        for t in count():

            # sample action for player 1
            inputs_policy_net = gladiator_game.players[0].get_inputs_from_states(current_states)
            action_idx = select_action(policy_net, inputs_policy_net)
            selected_action = ACTIONS[action_idx]

            actions_idx, actions = [], []
            actions_idx.append(action_idx) # player 1
            actions.append(selected_action) # player 1

            # player 2 acts randomly
            r_action_idx = random_action()
            r_action = ACTIONS[r_action_idx]
            actions_idx.append(r_action_idx) # player 2
            actions.append(r_action) # player 2

            # step through game, get relative states
            next_states, reward, done = gladiator_game.step(actions)
            next_states = gladiator_game.players[0].calc_rel_states(next_states)

            # TODO: Implement the policy gradient stuff here

            gladiator_game.render()

            # termination condition.
            if done:
                pl_a_score += gladiator_game.players[0].score
                pl_b_score += gladiator_game.players[1].score
                break

    # exit gracefully
    gladiator_game.quit()

    # plot scores
    plot_scores(pl_a_scores, pl_b_scores)


if __name__ == "__main__":
    run_policy_gradient()
