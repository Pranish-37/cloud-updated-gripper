"""
input_handler.py — pure Python (no ROS dependency) input reading for teleop.

Provides:
  TeleopConfig     — dataclass of all control parameters
  KeyboardInput    — pygame keyboard state reader + HUD renderer
  SpaceMouseInput  — pyspacemouse background thread reader

Usage in the ROS node:
    config  = TeleopConfig()
    kb      = KeyboardInput(config)
    sm      = SpaceMouseInput(config)   # optional

    kb.start()
    sm.start()                          # optional

    while running:
        stop, vx, vy, vz, v_roll, v_pitch, v_yaw, gripper_cmd = kb.step(sm)
        if stop:
            break
        publish(vx, vy, vz, v_roll, v_pitch, v_yaw, gripper_cmd)

    sm.stop()
    kb.stop()
"""

import math
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

try:
    import pyspacemouse
    _SPACEMOUSE_AVAILABLE = True
except ImportError:
    _SPACEMOUSE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TeleopConfig:
    # Translation / rotation speed
    translation_speed: float = 50.0      # mm/s
    rotation_speed: float = 10.0         # deg/s (keyboard rotation)

    # Axis inversion
    invert_x: bool = True
    invert_y: bool = True

    # Control loop rate (Hz) — must match teleop_node publish_rate
    control_hz: float = 50.0

    # Gripper values (xArm raw, 0–850)
    gripper_open_val: float = 850.0
    gripper_close_val: float = 0.0

    # Workspace limits (mm) — enforced in the node, kept here for HUD display
    x_limits: Tuple[float, float] = (200.0, 700.0)
    y_limits: Tuple[float, float] = (-400.0, 400.0)
    z_limits: Tuple[float, float] = (6.0, 600.0)

    # SpaceMouse
    spacemouse_rotation_speed: float = 30.0   # deg/s at full deflection
    spacemouse_deadzone: float = 0.05
    invert_roll: bool = False
    invert_pitch: bool = False
    invert_yaw: bool = False
    roll_scale: float = 1.0
    pitch_scale: float = 1.0
    yaw_scale: float = 1.0


# ---------------------------------------------------------------------------
# SpaceMouse reader (background thread)
# ---------------------------------------------------------------------------

class SpaceMouseInput:
    """Polls the SpaceMouse HID device in a daemon thread; get_state() is non-blocking."""

    def __init__(self, config: TeleopConfig) -> None:
        self._cfg = config
        self._device = None
        self._state = None
        self._lock = threading.Lock() # lock for the state
        self._stop_event = threading.Event() # event for the stop
        self._thread: Optional[threading.Thread] = None # thread for the polling

    def start(self) -> None:
        if not _SPACEMOUSE_AVAILABLE:
            raise ImportError('pyspacemouse not installed. Run: pip install pyspacemouse')
        self._device = pyspacemouse.open() # open the SpaceMouse HID device
        if self._device is None:
            raise RuntimeError('Could not open SpaceMouse — check USB connection.')
        self._stop_event.clear()
        # start the polling thread with in a daemon thread, it will run until the node is stopped
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name='spacemouse')
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set() # set the stop event
        if self._thread:
            self._thread.join(timeout=1.0) # join the thread, it will wait for the thread to finish
            self._thread = None
        if self._device:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None
        with self._lock:
            self._state = None

    def _poll_loop(self) -> None: # this loop keeps reading the SpaceMouse HID device in a daemon thread
        while not self._stop_event.is_set():
            state = self._device.read() # read the state from the SpaceMouse HID device
            if state is not None:
                with self._lock:
                    self._state = state

    def get_rotation_velocity(self) -> Tuple[float, float, float]:
        """Return (v_roll, v_pitch, v_yaw) in deg/s. Non-blocking."""
        with self._lock:
            state = self._state

        if state is None:
            return 0.0, 0.0, 0.0

        cfg = self._cfg
        rs = cfg.spacemouse_rotation_speed
        dz = cfg.spacemouse_deadzone

        def _axis(raw: float, invert: bool, scale: float) -> float:
            # scale the raw value by the scale factor and invert the sign if the invert flag is True
            v = raw * scale * (-1.0 if invert else 1.0)
            # if the absolute value of the scaled value is less than the deadzone, return 0.0
            if abs(v) < dz:
                return 0.0
            # otherwise, return the sign of the scaled value multiplied by the sign of the scaled value minus the deadzone divided by 1 minus the deadzone multiplied by the rotation speed
            sign = 1.0 if v > 0 else -1.0 # sign of the scaled value
            return sign * ((abs(v) - dz) / (1.0 - dz)) * rs

        return (
            _axis(state.roll,  cfg.invert_roll,  cfg.roll_scale),
            _axis(state.pitch, cfg.invert_pitch, cfg.pitch_scale),
            _axis(state.yaw,   cfg.invert_yaw,   cfg.yaw_scale),
        )


# ---------------------------------------------------------------------------
# pygame window + keyboard input
# ---------------------------------------------------------------------------

_FONT_SIZE = 16
_WIN_W, _WIN_H = 640, 460 # window width and height
_BG     = (30,  30,  30) # background color
_FG     = (220, 220, 220) # foreground color
_ACCENT = (80,  200, 120) # accent color
_WARN   = (220, 80,  80) # warning color


class KeyboardInput:
    """
    Manages the pygame window, reads keyboard events, renders the HUD.

    step() returns a StepResult namedtuple:
      stop         bool    — True when Esc / window close requested
      vx,vy,vz     float   — translation velocity (mm/s)
      v_roll,...   float   — rotation velocity (deg/s)  from keyboard OR spacemouse
      gripper_cmd  float | None — new gripper value if Q/E pressed, else None
    """

    def __init__(self, config: TeleopConfig) -> None:
        if not _PYGAME_AVAILABLE:
            raise ImportError('pygame not installed. Run: pip install pygame')
        self._cfg = config
        self._surface = None
        self._font = None
        self._stop = False
        self._prev_g_open = False
        self._prev_g_close = False
        self._recording_status = ''

    def start(self) -> None:
        pygame.init()
        self._surface = pygame.display.set_mode((_WIN_W, _WIN_H))
        pygame.display.set_caption('xArm Teleop (ROS 2)')
        self._font = pygame.font.SysFont('monospace', _FONT_SIZE)
        pygame.mouse.get_rel()  # drain accumulator

    def stop(self) -> None:
        try:
            pygame.quit()
        except Exception:
            pass
        self._surface = None
        self._font = None

    @property
    def stop_requested(self) -> bool:
        return self._stop

    def step(
        self,
        spacemouse: Optional[SpaceMouseInput],
        target: list,
        gripper: float,
        sent: bool = True,
        extra_keydown_callbacks: Optional[dict] = None,
        recording_status: str = '',
    ) -> Tuple[bool, float, float, float, float, float, float, Optional[float]]:
        """
        Process one input tick.

        Args:
          extra_keydown_callbacks: optional dict mapping pygame key constants to zero-arg
            callables, invoked once on KEYDOWN. Use for recording control (R/N/D/ESC).
          recording_status: short status string rendered in the HUD (e.g. 'REC', 'PAUSED').

        Returns:
          (stop, vx, vy, vz, v_roll, v_pitch, v_yaw, gripper_cmd)
          gripper_cmd is None if no gripper change requested this tick.
        """
        cfg = self._cfg
        dt = 1.0 / cfg.control_hz

        # drain pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._stop = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._stop = True
                if extra_keydown_callbacks:
                    cb = extra_keydown_callbacks.get(event.key)
                    if cb is not None:
                        cb()

        self._recording_status = recording_status

        keys = pygame.key.get_pressed()
        pygame.mouse.get_rel()

        # --- translation velocity (mm/s) ---
        ts = cfg.translation_speed
        sx = -1.0 if cfg.invert_x else 1.0
        sy = -1.0 if cfg.invert_y else 1.0
        vx = sx * ((ts if keys[pygame.K_w] else 0.0) - (ts if keys[pygame.K_s] else 0.0))
        vy = sy * ((ts if keys[pygame.K_a] else 0.0) - (ts if keys[pygame.K_d] else 0.0))
        ctrl = keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]
        vz = (ts if keys[pygame.K_SPACE] else 0.0) - (ts if ctrl else 0.0)

        # --- rotation velocity (deg/s) ---
        if spacemouse is not None:
            v_roll, v_pitch, v_yaw = spacemouse.get_rotation_velocity()
            rot_hint = ('SpaceMouse', 'SpaceMouse', 'SpaceMouse')
        else:
            rs = cfg.rotation_speed
            v_roll  = (rs if keys[pygame.K_j] else 0.0) - (rs if keys[pygame.K_l] else 0.0)
            v_pitch = (rs if keys[pygame.K_i] else 0.0) - (rs if keys[pygame.K_k] else 0.0)
            v_yaw   = (rs if keys[pygame.K_u] else 0.0) - (rs if keys[pygame.K_o] else 0.0)
            rot_hint = ('J / L', 'I / K', 'U / O')

        # --- gripper (edge detect) ---
        g_open  = bool(keys[pygame.K_q])
        g_close = bool(keys[pygame.K_e])
        gripper_cmd: Optional[float] = None
        if g_open and not self._prev_g_open:
            gripper_cmd = cfg.gripper_open_val
        elif g_close and not self._prev_g_close:
            gripper_cmd = cfg.gripper_close_val
        self._prev_g_open  = g_open
        self._prev_g_close = g_close

        # --- HUD ---
        if self._surface and self._font:
            self._render_hud(target, gripper, sent, rot_hint, self._recording_status)

        return self._stop, vx, vy, vz, v_roll, v_pitch, v_yaw, gripper_cmd

    def _render_hud(self, target, gripper, sent, rot_hint, recording_status: str = '') -> None:
        roll_h, pitch_h, yaw_h = rot_hint
        self._surface.fill(_BG)

        rec_color = _WARN if recording_status == 'REC' else _ACCENT if recording_status else _FG
        lines = [
            ('xArm Teleop (ROS 2) — keep this window focused', _FG),
            ('', _FG),
            (f'  x    ={target[0]:7.1f} mm   W / S',         _FG),
            (f'  y    ={target[1]:7.1f} mm   A / D',         _FG),
            (f'  z    ={target[2]:7.1f} mm   Space / Ctrl',  _FG),
            (f'  roll ={target[3]:7.1f} deg  {roll_h}',      _FG),
            (f'  pitch={target[4]:7.1f} deg  {pitch_h}',     _FG),
            (f'  yaw  ={target[5]:7.1f} deg  {yaw_h}',       _FG),
            (f'  gripper={gripper:5.0f}   Q=open  E=close',  _FG),
            ('', _FG),
            ('  [sent]' if sent else '  [idle]', _ACCENT if sent else _FG),
            (f'  [{recording_status}]' if recording_status else '', rec_color),
            ('  [STOPPED — Esc]' if self._stop else '', _WARN),
        ]
        y = 12
        for text, color in lines:
            surf = self._font.render(text, True, color)
            self._surface.blit(surf, (10, y))
            y += _FONT_SIZE + 3
        pygame.display.flip()
