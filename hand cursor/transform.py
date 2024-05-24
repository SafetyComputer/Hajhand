A = (0, 0)
B = (1, 0)
C = (1, 1)
D = (0, 1)

A1 = (0.1, 0.2)
B1 = (0.9, 0.2)
C1 = (0.9, 0.9)
D1 = (0.1, 0.9)

import cv2


class line:
    def __init__(self, point1: tuple[float, float], point2: tuple[float, float]) -> None:
        t = (point1[0] - point2[0]) / (point1[1] - point2[1])
        m = point1[0] - t * point1[1]
        self.t = t
        self.m = m

    def __call__(self, y: float) -> float:
        return self.t * y + self.m


def transform(point: tuple[float, float]) -> tuple[float, float]:
    y = (point[1] - B1[1]) * (C[1] - B[1]) / (C1[1] - B1[1])
    left_side = line(A1, D1)
    right_side = line(B1, C1)
    lm = left_side(point[1])
    rm = right_side(point[1])
    x = (point[0] - lm) / (rm - lm) * (C[0] - D[0])
    return (x, y)


def draw_quadrangle(frame: cv2.VideoCapture):
    global A1, B1, C1, D1

    # draw the quadrangle on frame
    h, w, _ = frame.shape

    a = (int(A1[0] * w), int(A1[1] * h))
    b = (int(B1[0] * w), int(B1[1] * h))
    c = (int(C1[0] * w), int(C1[1] * h))
    d = (int(D1[0] * w), int(D1[1] * h))

    cv2.line(frame, a, b, (0, 255, 0), 2)
    cv2.line(frame, b, c, (0, 255, 0), 2)
    cv2.line(frame, c, d, (0, 255, 0), 2)
    cv2.line(frame, d, a, (0, 255, 0), 2)


