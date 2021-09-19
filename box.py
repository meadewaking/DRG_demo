import cv2


class Box:
    def __init__(self, box):
        self.x1, self.y1, self.x2, self.y2 = box

    def get_state(self, frame):
        self.img = frame[self.y1:self.y2, self.x1:self.x2]
        return cv2.resize(self.img, (227, 227))

    def next(self, next_box):
        self.x1, self.y1, self.x2, self.y2 = next_box

    def draw(self, frame):
        p1 = (int(self.x1), int(self.y1))
        p2 = (int(self.x2), int(self.y2))
        cv2.rectangle(frame, p1, p2, (255, 255, 255), 3, 2)
