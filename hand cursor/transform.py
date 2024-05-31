import cv2

class Line:
    def __init__(self, point1: tuple[float, float], point2: tuple[float, float]) -> None:
        t = (point1[0] - point2[0]) / (point1[1] - point2[1])
        m = point1[0] - t * point1[1]
        self.t = t
        self.m = m

    def __call__(self, y: float) -> float:
        return self.t * y + self.m


class PerspectiveTransformer:
    def __init__(self,
                 A1: tuple[float, float] = (0.1, 0.2),
                 B1: tuple[float, float] = (0.9, 0.2),
                 C1: tuple[float, float] = (0.9, 0.9),
                 D1: tuple[float, float] = (0.1, 0.9)):
        self.A = (0, 0)
        self.B = (1, 0)
        self.C = (1, 1)
        self.D = (0, 1)

        self.A1 = A1
        self.B1 = B1
        self.C1 = C1
        self.D1 = D1

    def transform(self, point: tuple[float, float]) -> tuple[float, float]:
        y = (point[1] - self.B1[1]) * (self.C[1] - self.B[1]) / (self.C1[1] - self.B1[1])
        left_side = Line(self.A1, self.D1)
        right_side = Line(self.B1, self.C1)
        lm = left_side(point[1])
        rm = right_side(point[1])
        x = (point[0] - lm) / (rm - lm) * (self.C[0] - self.D[0])
        return x, y

    def draw_quadrangle(self, frame: cv2.VideoCapture):
        # draw the quadrangle on frame
        h, w, _ = frame.shape

        a = (int(self.A1[0] * w), int(self.A1[1] * h))
        b = (int(self.B1[0] * w), int(self.B1[1] * h))
        c = (int(self.C1[0] * w), int(self.C1[1] * h))
        d = (int(self.D1[0] * w), int(self.D1[1] * h))

        cv2.line(frame, a, b, (0, 255, 0), 2)
        cv2.line(frame, b, c, (0, 255, 0), 2)
        cv2.line(frame, c, d, (0, 255, 0), 2)
        cv2.line(frame, d, a, (0, 255, 0), 2)
