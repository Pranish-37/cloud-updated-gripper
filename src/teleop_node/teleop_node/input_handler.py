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
    translation_speed: float = 50.0      # mm/s (keyboard)
    rotation_speed: float = 10.0         # deg/s (keyboard rotation)

    # Axis inversion — keyboard translation
    invert_x: bool = True
    invert_y: bool = True

    # Control loop rate (Hz) — must match teleop_node publish_rate
    control_hz: float = 50.0

    # DoF mode: 3 = xyz + gripper (RPY locked), 6 = full pose + gripper
    dof_mode: int = 3

    # Gripper values (xArm raw, 0–850)
    gripper_open_val: float = 850.0
    gripper_close_val: float = 0.0

    # Workspace / attitude safety limits (enforced every teleop tick)
    # XYZ in mm; RPY in degrees (roll uses [0,360) convention near home≈180°)
    x_limits: Tuple[float, float] = (200.0, 700.0)
    y_limits: Tuple[float, float] = (-400.0, 400.0)
    z_limits: Tuple[float, float] = (6.0, 600.0)
    roll_limits: Tuple[float, float] = (90.0, 270.0)
    pitch_limits: Tuple[float, float] = (-45.0, 45.0)
    yaw_limits: Tuple[float, float] = (-90.0, 90.0)

    # Per-tick absolute step caps (anti-爆冲); mm / deg per control tick
    max_step_x_mm: float = 5.0
    max_step_y_mm: float = 5.0
    max_step_z_mm: float = 5.0
    max_step_roll_deg: float = 2.0
    max_step_pitch_deg: float = 2.0
    max_step_yaw_deg: float = 2.0

    # SpaceMouse
    spacemouse_translation_speed: float = 50.0  # mm/s at full deflection
    spacemouse_rotation_speed: float = 30.0     # deg/s at full deflection
    spacemouse_translation_deadzone: float = 0.05
    spacemouse_rotation_deadzone: float = 0.05
    spacemouse_invert_x: bool = False
    spacemouse_invert_y: bool = False
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
    """Polls the SpaceMouse HID device in a daemon thread; getters are non-blocking."""

    def __init__(self, config: TeleopConfig) -> None:
        self._cfg = config
        self._device = None
        self._state = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._prev_btn_left = False
        self._prev_btn_right = False
        # Until armed, all motion axes report zero — prevents startup jerk from
        # residual deflection / first noisy HID samples.
        self._armed = False

    def start(self) -> None:
        if not _SPACEMOUSE_AVAILABLE:
            raise ImportError('pyspacemouse not installed. Run: pip install pyspacemouse')
        self._device = pyspacemouse.open()
        if self._device is None:
            raise RuntimeError('Could not open SpaceMouse — check USB connection.')
        self._armed = False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name='spacemouse')
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._device:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None
        with self._lock:
            self._state = None
        self._armed = False

    def arm(self) -> None:
        """Enable motion output (call after is_centered())."""
        self._armed = True

    def is_centered(self) -> bool:
        """True when a sample exists and all DOF axes are inside their deadzones."""
        with self._lock:
            state = self._state
        if state is None:
            return False
        tdz = self._cfg.spacemouse_translation_deadzone
        rdz = self._cfg.spacemouse_rotation_deadzone
        return (
            abs(state.x) <= tdz and abs(state.y) <= tdz and abs(state.z) <= tdz
            and abs(state.roll) <= rdz and abs(state.pitch) <= rdz and abs(state.yaw) <= rdz
        )

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            state = self._device.read()
            if state is not None:
                with self._lock:
                    self._state = state

    @staticmethod
    def _scaled_axis(raw: float, invert: bool, scale: float, deadzone: float, speed: float) -> float:
        v = raw * scale * (-1.0 if invert else 1.0)
        if abs(v) < deadzone:
            return 0.0
        sign = 1.0 if v > 0 else -1.0
        return sign * ((abs(v) - deadzone) / (1.0 - deadzone)) * speed

    def get_translation_velocity(self) -> Tuple[float, float, float]:
        """Return (vx, vy, vz) in mm/s. Non-blocking.

        SpaceMouse HID: x = lateral (left/right), y = longitudinal (forward/back).
        xArm base:      x = forward,               y = lateral.
        Invert flags apply to the *device* axes (before the SM↔robot swap).
        """
        if not self._armed:
            return 0.0, 0.0, 0.0
        with self._lock:
            state = self._state
        if state is None:
            return 0.0, 0.0, 0.0

        cfg = self._cfg
        ts = cfg.spacemouse_translation_speed
        dz = cfg.spacemouse_translation_deadzone
        # Invert on device axes, then swap into robot base frame.
        sm_x = self._scaled_axis(state.x, cfg.spacemouse_invert_x, 1.0, dz, ts)
        sm_y = self._scaled_axis(state.y, cfg.spacemouse_invert_y, 1.0, dz, ts)
        sm_z = self._scaled_axis(state.z, False, 1.0, dz, ts)
        return (
            sm_y,  # SM y (fwd/back) → robot x
            sm_x,  # SM x (left/right) → robot y
            sm_z,
        )

    def get_rotation_velocity(self) -> Tuple[float, float, float]:
        """Return (v_roll, v_pitch, v_yaw) in deg/s. Non-blocking."""
        if not self._armed:
            return 0.0, 0.0, 0.0
        with self._lock:
            state = self._state
        if state is None:
            return 0.0, 0.0, 0.0

        cfg = self._cfg
        rs = cfg.spacemouse_rotation_speed
        dz = cfg.spacemouse_rotation_deadzone
        return (
            self._scaled_axis(state.roll,  cfg.invert_roll,  cfg.roll_scale,  dz, rs),
            self._scaled_axis(state.pitch, cfg.invert_pitch, cfg.pitch_scale, dz, rs),
            self._scaled_axis(state.yaw,   cfg.invert_yaw,   cfg.yaw_scale,   dz, rs),
        )

    def get_gripper_edge(self) -> Optional[float]:
        """
        Edge-detect SpaceMouse buttons.
        buttons[0] = left  → open
        buttons[1] = right → close
        Returns gripper value on rising edge, else None.
        """
        with self._lock:
            state = self._state
        if state is None:
            return None

        buttons = getattr(state, 'buttons', None) or []
        btn_left = bool(buttons[0]) if len(buttons) > 0 else False
        btn_right = bool(buttons[1]) if len(buttons) > 1 else False

        cfg = self._cfg
        gripper_cmd: Optional[float] = None
        if btn_left and not self._prev_btn_left:
            gripper_cmd = cfg.gripper_open_val
        elif btn_right and not self._prev_btn_right:
            gripper_cmd = cfg.gripper_close_val

        self._prev_btn_left = btn_left
        self._prev_btn_right = btn_right
        return gripper_cmd


# ---------------------------------------------------------------------------
# pygame window + keyboard input
# ---------------------------------------------------------------------------

_FONT_SIZE = 16
_WIN_W, _WIN_H = 640, 500
_BG     = (30,  30,  30)
_FG     = (220, 220, 220)
_ACCENT = (80,  200, 120)
_WARN   = (220, 80,  80)


class KeyboardInput:
    """
    Manages the pygame window, reads keyboard events, renders the HUD.

    When spacemouse is provided, EE motion and SM gripper buttons come from
    SpaceMouse; keyboard keeps Q/E gripper, recording keys, and Esc.
    When spacemouse is None, falls back to the full keyboard control scheme.
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
        pygame.mouse.get_rel()

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

        Returns:
          (stop, vx, vy, vz, v_roll, v_pitch, v_yaw, gripper_cmd)
          gripper_cmd is None if no gripper change requested this tick.
        """
        cfg = self._cfg
        allow_rotation = cfg.dof_mode == 6

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

        if spacemouse is not None:
            vx, vy, vz = spacemouse.get_translation_velocity()
            if allow_rotation:
                v_roll, v_pitch, v_yaw = spacemouse.get_rotation_velocity()
                rot_hint = ('SpaceMouse', 'SpaceMouse', 'SpaceMouse')
            else:
                v_roll = v_pitch = v_yaw = 0.0
                rot_hint = ('locked', 'locked', 'locked')
            xyz_hint = ('SpaceMouse', 'SpaceMouse', 'SpaceMouse')
            sm_gripper = spacemouse.get_gripper_edge()
        else:
            ts = cfg.translation_speed
            sx = -1.0 if cfg.invert_x else 1.0
            sy = -1.0 if cfg.invert_y else 1.0
            vx = sx * ((ts if keys[pygame.K_w] else 0.0) - (ts if keys[pygame.K_s] else 0.0))
            vy = sy * ((ts if keys[pygame.K_a] else 0.0) - (ts if keys[pygame.K_d] else 0.0))
            ctrl = keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]
            vz = (ts if keys[pygame.K_SPACE] else 0.0) - (ts if ctrl else 0.0)
            xyz_hint = ('W / S', 'A / D', 'Space / Ctrl')

            if allow_rotation:
                rs = cfg.rotation_speed
                v_roll  = (rs if keys[pygame.K_j] else 0.0) - (rs if keys[pygame.K_l] else 0.0)
                v_pitch = (rs if keys[pygame.K_i] else 0.0) - (rs if keys[pygame.K_k] else 0.0)
                v_yaw   = (rs if keys[pygame.K_u] else 0.0) - (rs if keys[pygame.K_o] else 0.0)
                rot_hint = ('J / L', 'I / K', 'U / O')
            else:
                v_roll = v_pitch = v_yaw = 0.0
                rot_hint = ('locked', 'locked', 'locked')
            sm_gripper = None

        # Keyboard Q/E always available (backup when SpaceMouse is active)
        g_open  = bool(keys[pygame.K_q])
        g_close = bool(keys[pygame.K_e])
        gripper_cmd: Optional[float] = sm_gripper
        if g_open and not self._prev_g_open:
            gripper_cmd = cfg.gripper_open_val
        elif g_close and not self._prev_g_close:
            gripper_cmd = cfg.gripper_close_val
        self._prev_g_open  = g_open
        self._prev_g_close = g_close

        if self._surface and self._font:
            self._render_hud(
                target, gripper, sent, xyz_hint, rot_hint,
                spacemouse is not None, self._recording_status,
            )

        return self._stop, vx, vy, vz, v_roll, v_pitch, v_yaw, gripper_cmd

    def _render_hud(
        self,
        target,
        gripper,
        sent,
        xyz_hint,
        rot_hint,
        sm_active: bool,
        recording_status: str = '',
    ) -> None:
        x_h, y_h, z_h = xyz_hint
        roll_h, pitch_h, yaw_h = rot_hint
        self._surface.fill(_BG)

        rec_color = _WARN if recording_status == 'REC' else _ACCENT if recording_status else _FG
        dof = self._cfg.dof_mode
        grip_hint = (
            'SM L/R + Q/E'
            if sm_active else
            'Q=open  E=close'
        )
        lines = [
            (f'xArm Teleop (ROS 2) — {dof}-DoF — keep this window focused', _FG),
            ('', _FG),
            (f'  x    ={target[0]:7.1f} mm   {x_h}',         _FG),
            (f'  y    ={target[1]:7.1f} mm   {y_h}',         _FG),
            (f'  z    ={target[2]:7.1f} mm   {z_h}',         _FG),
            (f'  roll ={target[3]:7.1f} deg  {roll_h}',      _FG),
            (f'  pitch={target[4]:7.1f} deg  {pitch_h}',     _FG),
            (f'  yaw  ={target[5]:7.1f} deg  {yaw_h}',       _FG),
            (f'  gripper={gripper:5.0f}   {grip_hint}',      _FG),
            ('', _FG),
            ('  Recording: R=toggle  N=save  Bksp=discard  Esc=finalize', _FG),
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
