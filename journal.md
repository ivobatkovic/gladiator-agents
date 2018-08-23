# Journal

## Goal

- [ ] Implement a self-learning agent that learns to play a simple 2 player shooting game.

## Tasks

- [x] Setup the basic game with 2 players, actions (left, right, jump, shoot).
  - [x] Add basic game skeleton, with empty level and 2 players.
  - [x] Add bullets, check collision with bullets. Update score if bullet hits other player.
  - [x] Add wrapper class for the game.
  - [x] Add `step` method to get states (x_pos, y_pos, x_change, y_change) for player and bullet.
  - [x] Add `render` method to update screen.
  - [x] Add option to get screen as an rgb image, which can be used as a state.
  - [x] Add method to player class to calculate relative position of other players,
  bullets w.r.t to that player.
  
<p>
    <img src="/plots/progress_1.gif" width="200">
    <em>game setup</em>
</p>

- [ ] Implement q-learning.
