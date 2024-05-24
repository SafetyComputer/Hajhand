import math

import pyautogui

## get screen size
screenWidth, screenHeight = pyautogui.size()


def track_mouse(norm_x, norm_y, duration=0.1):
    norm_x = max(0, min(1, norm_x))
    norm_y = max(0, min(1, norm_y))

    pyautogui.moveTo(
        norm_x * screenWidth,
        norm_y * screenHeight,
        duration=duration,
        _pause=False
    )


def circle_coordinate(cx, cy, r, step=100):
    ## A iterator to generate the coordinate of a circle
    for i in range(step):
        x = cx + r * math.cos(2 * math.pi * i / step)
        y = cy + r * math.sin(2 * math.pi * i / step)
        yield x, y


import time

t = time.time()

# move cursor in a circle

for x, y in circle_coordinate(0.5, 0.5, 0.3, 100):
    track_mouse(x, y, 0)

print(f"Time: {time.time() - t}")
