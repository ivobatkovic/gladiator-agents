import pygame, random

# Global constants

# Predefined colors
BACKGROUND = (round(0.4*255),round(0.4*255),round(0.4*255))
WALLS  = (0,0,0)
PLAYER_COLOR = (round(0.85*255),round(0.325*255),round(0.0980*255))
ENEMY_COLOR = (0, round(0.4470*255), round(0.7410*255))

# Screen dimensions
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

# Actions
LEFT = 'left'
RIGHT = 'right'
JUMP = 'jump'
SHOOT = 'shoot'
ACTIONS = [LEFT, RIGHT, JUMP, SHOOT]


class Bullet(pygame.sprite.Sprite):

    def __init__(self, player):
        super().__init__()

        self.image = pygame.Surface([10,5])
        self.image.fill((255,255,255))
        self.rect = self.image.get_rect()

        # Attribute to know which player spawned the bullet
        self.player = player
        # Initial bullet direction  and position
        self.direction = player.direction
        self.rect.x, self.rect.y = self.player.rect.x+20, self.player.rect.y+20

        # Bullet speed
        self.change_x = 10
        self.change_y = 0

    def update(self):

        # Update bullet position
        self.rect.x += self.direction * self.change_x
        self.rect.y += self.change_y

        # Check collision with platforms in the arena
        block_hit_list = pygame.sprite.spritecollide(self, self.player.level.platform_list, False)
        if len(block_hit_list) > 0:
            self.player.level.bullet_list.remove(self)
            self.player.bullets.remove(self)

        # Check collision with other players
        block_hit_list  = pygame.sprite.spritecollide(self, self.player.level.player_list, False)
        for block in block_hit_list:
            for other_player in self.player.other_players:
                if block is other_player:
                    self.player.level.bullet_list.remove(self)
                    self.player.bullets.remove(self)
                    self.player.score += 1
                    other_player.score -= 1


class Player(pygame.sprite.Sprite):

    def __init__(self,color,position):
        """ Constructor function """

        # Call the parent's constructor
        super().__init__()

        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        width = 40
        height = 40
        self.image = pygame.Surface([width, height])
        self.image.fill(color)

        # Set a referance to the image rect.
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = position

        # Specify a random direction when spawning
        self.direction = 1 - 2*random.randint(0,1)

        # Set speed vector of player
        self.change_x = 0
        self.change_y = 0

        # List of bullet that the player spawns
        self.bullets = []

        # List of sprites we can bump against
        self.level = None
        self.other_players = None

        # Score value when hitting a player or getting hit
        self.score = 0

    def update_other_players(self):
        # Update list of other players excluding yourself
        self.other_players = [player for player in self.level.players]
        self.other_players.remove(self)

    def collision_check_x(self):
        # Check collisions with the platforms
        block_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
        for block in block_hit_list:
            # If we are moving right,
            # set our right side to the left side of the item we hit
            if self.change_x > 0:
                self.rect.right = block.rect.left
            elif self.change_x < 0:
                # Otherwise if we are moving left, do the opposite.
                self.rect.left = block.rect.right

        # Check collisions with other players
        block_hit_list = pygame.sprite.spritecollide(self, self.other_players, False)
        for block in block_hit_list:
            # If we are moving right,
            # set our right side to the left side of the item we hit
            if self.change_x > 0:
                self.rect.right = block.rect.left
            elif self.change_x < 0:
                # Otherwise if we are moving left, do the opposite.
                self.rect.left = block.rect.right

    def collision_check_y(self):
        # Check collisions with the platforms
        block_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
        for block in block_hit_list:
            # Reset our position based on the top/bottom of the object.
            if self.change_y > 0:
                self.rect.bottom = block.rect.top
            elif self.change_y < 0:
                self.rect.top = block.rect.bottom

            # Stop our vertical movement
            self.change_y = 0

        # Check collisions with other players
        block_hit_list = pygame.sprite.spritecollide(self, self.other_players, False)
        for block in block_hit_list:
            # Reset our position based on the top/bottom of the object.
            if self.change_y > 0:
                self.rect.bottom = block.rect.top
            elif self.change_y < 0:
                self.rect.top = block.rect.bottom

            # Stop our vertical movement
            self.change_y = 0

    def update(self):
        """ Move the player. """

        # Gravity
        self.calc_grav()

        # Move left/right
        self.rect.x += self.change_x
        self.collision_check_x()

        # Move up/down
        self.rect.y += self.change_y
        self.collision_check_y()

    def calc_grav(self):
        """ Calculate effect of gravity. """
        if self.change_y == 0:
            self.change_y = 1
        else:
            self.change_y += .35

        # See if we are on the ground.
        if self.rect.y >= SCREEN_HEIGHT - self.rect.height and self.change_y >= 0:
            self.change_y = 0
            self.rect.y = SCREEN_HEIGHT - self.rect.height

    def jump(self):
        """ Called when user hits 'jump' button. """
        # move down a bit and see if there is a platform below us.
        # Move down 2 pixels because it doesn't work well if we only move down
        # 1 when working with a platform moving down.
        self.rect.y += 2
        platform_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
        self.rect.y -= 2

        # If it is ok to jump, set our speed upwards
        if len(platform_hit_list) > 0 or self.rect.bottom >= SCREEN_HEIGHT:
            self.change_y = -10

    # Player-controlled movement:
    def go_left(self):
        """ Called when the user hits the left arrow. """
        self.change_x = -6
        self.direction = -1

    def go_right(self):
        """ Called when the user hits the right arrow. """
        self.change_x = 6
        self.direction = 1

    def stop(self):
        """ Called when the user lets off the keyboard. """
        self.change_x = 0

    def shoot(self):
        # Currently, allow only one bullet at the time
        if  len(self.bullets) == 0 :
            bullet =  Bullet(self)
            self.bullets.append(bullet)
            self.level.bullet_list.add(bullet)

    def choose_action(self):
        # Select a random action
        self.action  = random_action()

        # Apply the random action
        if self.action is not None:
            if self.action == LEFT:
                self.go_left()
            elif self.action == RIGHT:
                self.go_right()
            elif self.action == JUMP:
                self.jump()
            elif self.action == SHOOT:
                self.shoot()
        else:
            self.stop()

    def act(self, action):
        """Perform action."""
        if action is not None:
            if action == LEFT:
                self.go_left()
            elif action == RIGHT:
                self.go_right()
            elif action == JUMP:
                self.jump()
            elif action == SHOOT:
                self.shoot()
        else:
            self.stop()


class Platform(pygame.sprite.Sprite):
    """ Platform the user can jump on """

    def __init__(self, width, height):
        """ Platform constructor. Assumes constructed with user passing in
            an array of 5 numbers like what's defined at the top of this
            code. """
        super().__init__()

        self.image = pygame.Surface([width, height])
        self.image.fill(WALLS)

        self.rect = self.image.get_rect()


class Level(object):
    """ This is a generic super-class used to define a level.
        Create a child class for each level with level-specific
        info. """

    def __init__(self, players):
        """ Constructor. Pass in a handle to player. Needed for when moving platforms
            collide with the player. """
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


        # Background image
        self.background = None

    # Update everythign on this level
    def update(self):
        """ Update everything in this level."""
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
        """ Draw everything on this level. """

        # Draw the background
        screen.fill(BACKGROUND)

        # Draw all the sprite lists that we have
        self.platform_list.draw(screen)
        self.player_list.draw(screen)
        self.bullet_list.draw(screen)


# Create platforms for the level
class SimpleLevel(Level):
    """ Definition for level 1. """

    def __init__(self, players):
        """ Create level 1. """

        # Call the parent constructor
        Level.__init__(self, players)


        # Define all walls: width,height,x pos,y pos
        level = [[40, 480, 0, 0],
                 [40, 480, 600, 0],
                 [640, 40, 0, 0],
                 [640, 40, 0, 440],
                 ]

        # Go through the array above and add platforms
        for platform in level:
            block = Platform(platform[0], platform[1])
            block.rect.x = platform[2]
            block.rect.y = platform[3]
            self.platform_list.add(block)


class Game:
    """Generic class to wrap around, with interfaces to run and interact with the game."""

    def __init__(self):
        """Constructor."""

        # Initiate pygame
        pygame.init()

        # Set screen parameters
        self.size = [SCREEN_WIDTH,  SCREEN_HEIGHT]
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Python self-learning project")

        # Used to manage how fast the screen updates
        self.clock = pygame.time.Clock()
        self.fps = 60

        
        # Create the players
        self.players = [Player(PLAYER_COLOR,[300, 80]), Player(ENEMY_COLOR,[500, 80])]

        # Create the level
        self.level = SimpleLevel(self.players)

    def step(self, actions):
        """Step through one iteration of the game."""
        for idx, player in enumerate(self.players):
            player.act(actions[idx])

        self.level.update()
        states = self.level.get_states()

        # reward is the score of each player
        rewards = self.level.get_scores()
        return states, rewards

    def render(self):
        """Draw the sprites and update."""
        events = pygame.event.get()
        self.level.draw(self.screen)

        # Limit the fps
        self.clock.tick(self.fps)

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

    def reset(self):
        """Reinitialise game, resetting states."""
        # TODO: Test this!
        self.__init__()

    def quit(self):
        pygame.quit()


# Display scores
def add_text(screen, text, pos):
    font = pygame.font.SysFont('ptmono', 20)
    screen.blit(font.render(text, True, (255,255,255)), pos)


def random_action():
    """Pick random action."""
    if random.random() < 0.5:
        return random.choice(ACTIONS)
    else:
        return None


def main():
    """ Main Program """
    

    # Initialise game
    gladiator_game = Game()

    # Loop over episodes
    n_episodes = 1000

    # -------- Main Program Loop -----------
    for episode_idx in range(n_episodes):

        # Choose action for each player
        actions = []
        for _ in range(len(gladiator_game.players)):
            actions.append(random_action())

        states, rewards = gladiator_game.step(actions)
        print(rewards)

        gladiator_game.render()

        

        

    # Be IDLE friendly. If you forget this line, the program will 'hang'
    # on exit.
    pygame.quit()


if __name__ == "__main__":
    main()
