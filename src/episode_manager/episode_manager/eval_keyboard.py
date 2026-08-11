"""Pygame control window for policy evaluation (mirrors teleop UX)."""

from __future__ import annotations

from dataclasses import dataclass

import pygame


_WIN_W, _WIN_H = 640, 480
_FONT_SIZE = 16
_BG = (24, 24, 28)
_FG = (220, 220, 220)
_ACCENT = (120, 200, 255)
_DIM = (140, 140, 150)


class EvalQuit(Exception):
    """User pressed Esc or closed the control window."""


class EvalAbortEpisode(Exception):
    """User pressed S — stop current episode, mark failure, go to next."""


class EvalMarkOutcome(Exception):
    """User pressed Y or N mid-episode to immediately mark success/failure."""
    def __init__(self, success: bool) -> None:
        self.success = success
        super().__init__('success' if success else 'failure')


@dataclass
class EvalKeyEvent:
    confirm: bool = False
    yes: bool = False
    no: bool = False
    stop_episode: bool = False
    quit: bool = False


class EvalKeyboard:
    """HUD window: Enter/Space, Y/N, S (abort episode), Esc (quit)."""

    def __init__(self) -> None:
        pygame.init()
        self._surface = pygame.display.set_mode((_WIN_W, _WIN_H))
        pygame.display.set_caption('CloudGripper Eval')
        self._font = pygame.font.SysFont('monospace', _FONT_SIZE)
        self._status = 'Starting...'
        self._hint = ''
        self._quit = False
        self._draw()

    def close(self) -> None:
        pygame.quit()

    def set_prompt(self, status: str, hint: str = '') -> None:
        self._status = status
        self._hint = hint
        self._draw()

    def poll(self) -> EvalKeyEvent:
        """Drain pygame events once; update HUD; return edge-triggered keys."""
        ev = EvalKeyEvent()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                ev.quit = True
                self._quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    ev.quit = True
                    self._quit = True
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                    ev.confirm = True
                elif event.key == pygame.K_y:
                    ev.yes = True
                elif event.key == pygame.K_n:
                    ev.no = True
                elif event.key == pygame.K_s:
                    ev.stop_episode = True
        self._draw()
        return ev

    @property
    def quit_requested(self) -> bool:
        return self._quit

    def _draw(self) -> None:
        self._surface.fill(_BG)
        y = 24
        for line, color in (
            ('CloudGripper Policy Eval — keep this window focused', _ACCENT),
            ('', _FG),
            (self._status, _FG),
            (self._hint, _DIM),
            ('', _FG),
            ('Enter / Space  confirm (start eval / start episode)', _DIM),
            ('Y               success → stop now → home → next', _DIM),
            ('N               failure → stop now → home → next', _DIM),
            ('S               abort (fail) → stop now → home → next', _DIM),
            ('Esc             quit evaluation', _DIM),
        ):
            if line:
                surf = self._font.render(line, True, color)
                self._surface.blit(surf, (24, y))
            y += _FONT_SIZE + 6
        pygame.display.flip()
