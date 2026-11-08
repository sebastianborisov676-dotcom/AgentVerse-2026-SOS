"""Draw what the robot perceives and decides onto camera images (for the demo, video and report)."""

from __future__ import annotations

import cv2
import numpy as np

STATE_COLOURS = {  # BGR
    "searching": (200, 200, 200), "approach": (239, 182, 134), "align": (229, 135, 57), "insert": (171, 92, 28),
    "seated": (12, 163, 12), "retract": (90, 131, 236), "aborted": (59, 59, 208), "safe_stop": (59, 59, 208),
    "estop": (59, 59, 208), "success": (12, 163, 12), "failed": (59, 59, 208),
}


def robot_camera_overlay(agent, data, scale: float = 0.5) -> np.ndarray:
    """Robot camera image (BGR) with the detected marker, estimated socket axis and status."""
    per = agent.perception
    img = per.last_image if per.last_image is not None else np.zeros((per.h, per.w, 3), np.uint8)
    out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    est = agent.last_obs
    if est is not None and est.corners_px is not None:
        pts = est.corners_px.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 220, 0) if est.detected else (0, 0, 255), 3)
    if agent.filter.pos is not None:
        uv = per.project_socket(data, agent.filter.pos, agent.filter.rot)
        if uv is not None:
            c, tip, up = (tuple(int(v) for v in p) for p in uv)
            cv2.circle(out, c, 10, (0, 255, 255), 3)
            cv2.arrowedLine(out, c, tip, (0, 255, 255), 3, tipLength=0.25)
    out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    label = "inlet detected" if (est and est.detected) else ("no inlet in view" if est else "starting")
    conf = f"conf {est.confidence:.2f}  reproj {est.reproj_px:.2f}px  {est.latency_ms:.0f}ms" if est and est.detected else (
        est.reason if est else "")
    _banner(out, [f"ROBOT CAMERA  |  {label}", conf])
    return out


def status_panel(agent, data, width: int, height: int, extra: list[str] | None = None) -> np.ndarray:
    panel = np.full((height, width, 3), (30, 24, 20), np.uint8)
    phase = agent.phase
    col = STATE_COLOURS.get(phase, (200, 200, 200))
    cv2.putText(panel, "AUTOCHARGE AI - AUTONOMOUS", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 120), 2)
    cv2.circle(panel, (24, 64), 9, col, -1)
    cv2.putText(panel, phase.upper(), (42, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    lines = [f"t = {data.time:5.1f} s"]
    c = agent.ctrl
    if c is not None:
        al = c._estimated_alignment(data)
        lines += [
            f"offset from axis  {al.lateral * 1e3:6.2f} mm (estimated)",
            f"distance to seat  {-al.axial * 1e3:6.1f} mm",
            f"angle error       {np.degrees(al.angle):6.2f} deg",
            f"contact force     {c.force_filt:6.1f} N",
            f"re-plans {c.replans}   retries {c.retries}",
        ]
    lines += extra or []
    y = 104
    for ln in lines:
        cv2.putText(panel, ln, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        y += 22
    cv2.putText(panel, "DECISIONS", (14, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 120), 1)
    y += 30
    for t, txt in agent.events[-8:]:
        txt = txt if len(txt) < 62 else txt[:59] + "..."
        cv2.putText(panel, f"{t:5.1f}s {txt}", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)
        y += 19
        if y > height - 8:
            break
    return panel


def _banner(img, lines):
    h = 20 * len(lines) + 8
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (8, 18 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
