"""Minesweeper suite: a self-contained Gymnasium env, sized as an image-like observation so
it goes through the same RetinaEncoder as Atari frames - nothing elsewhere needs to change."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .base import GameSuite, register

GAMES = {"beginner": (9, 9, 10), "intermediate": (16, 16, 40), "expert": (16, 30, 99)}


class MinesweeperEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, height: int = 9, width: int = 9, mines: int = 10, obs_scale: int = 8,
                render_mode: str | None = None):
        super().__init__()
        self.h, self.w, self.n_mines, self.obs_scale = height, width, mines, obs_scale
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(0.0, 1.0, (height * obs_scale, width * obs_scale), np.float32)
        self.action_space = gym.spaces.Discrete(height * width)
        self._mines = self._counts = self._revealed = self._flagged = None
        self._done = False

    def _neighbors(self, r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.h and 0 <= cc < self.w:
                        yield rr, cc

    def _place_mines(self, safe_r, safe_c, rng):
        safe = {(safe_r, safe_c)} | set(self._neighbors(safe_r, safe_c))
        cells = [(r, c) for r in range(self.h) for c in range(self.w) if (r, c) not in safe]
        chosen = rng.choice(len(cells), size=min(self.n_mines, len(cells)), replace=False)
        self._mines = np.zeros((self.h, self.w), dtype=bool)
        for i in chosen:
            self._mines[cells[i]] = True
        self._counts = np.array([[sum(self._mines[rr, cc] for rr, cc in self._neighbors(r, c))
                                  for c in range(self.w)] for r in range(self.h)], dtype=np.int8)

    def _flood_reveal(self, r, c) -> int:
        stack, n = [(r, c)], 0
        while stack:
            r, c = stack.pop()
            if self._revealed[r, c] or self._flagged[r, c]:
                continue
            self._revealed[r, c] = True; n += 1
            if self._counts[r, c] == 0:
                stack.extend(self._neighbors(r, c))
        return n

    def _obs(self) -> np.ndarray:
        out = np.full((self.h, self.w), 0.75, np.float32)              # unrevealed
        out[self._flagged] = 1.0
        m = self._revealed
        out[m] = self._counts[m].astype(np.float32) / 8.0 * 0.5        # revealed: [0, 0.5]
        return np.kron(out, np.ones((self.obs_scale, self.obs_scale), np.float32))
    def _mask(self) -> np.ndarray:
        return (~(self._revealed | self._flagged)).reshape(-1)

    def _won(self) -> bool:
        return bool(np.all(self._revealed | self._mines))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._mines = None
        self._counts = np.zeros((self.h, self.w), dtype=np.int8)
        self._revealed = np.zeros((self.h, self.w), dtype=bool)
        self._flagged = np.zeros((self.h, self.w), dtype=bool)
        self._done = False
        return self._obs(), {"action_mask": self._mask()}

    def step(self, action: int):
        if self._done:
            raise RuntimeError("step() after episode end; call reset()")
        r, c = divmod(int(action), self.w)
        if self._mines is None:
            self._place_mines(r, c, self.np_random)
        if self._revealed[r, c]:
            return self._obs(), -0.05, False, False, {"action_mask": self._mask()}
        if self._mines[r, c]:
            self._revealed |= self._mines
            self._done = True
            return self._obs(), -1.0, True, False, {"action_mask": self._mask()}
        n = self._flood_reveal(r, c)
        won = self._won()
        self._done = won
        return self._obs(), n / (self.h * self.w) + (10.0 if won else 0.0), won, False, {"action_mask": self._mask()}
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        cell = 20
        img = np.full((self.h * cell, self.w * cell, 3), 190, np.uint8)
        palette = [(190,190,190),(30,60,200),(30,130,30),(200,30,30),(10,10,130),
                  (110,20,20),(20,130,130),(0,0,0),(130,130,130)]
        for r in range(self.h):
            for c in range(self.w):
                y, x = r * cell, c * cell
                if self._flagged[r, c]:
                    img[y:y+cell, x:x+cell] = (230, 200, 40)
                elif not self._revealed[r, c]:
                    img[y:y+cell, x:x+cell] = (150, 150, 150)
                elif self._mines[r, c]:
                    img[y:y+cell, x:x+cell] = (220, 30, 30)
                else:
                    img[y+1:y+cell-1, x+1:x+cell-1] = (225, 225, 225)
                    k = int(self._counts[r, c])
                    if k:
                        img[y+6:y+cell-6, x+6:x+cell-6] = palette[k]
        return img


@register("minesweeper")
class MinesweeperSuite(GameSuite):
    def games(self) -> list[str]:
        return list(GAMES)

    def make(self, game, seed=None, render_mode=None, max_episode_steps=200, **kw):
        h, w, mines = GAMES.get(game, GAMES["beginner"])
        env = MinesweeperEnv(h, w, mines, render_mode=render_mode, **kw)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        return self.finish(env, seed, normalize_obs=False)