import cv2
from loader import Loader
from goturn import Goturn
from box import Box

loader = Loader()
global_goturn = Goturn()


def test_ma(k=1.0):
    video_path = 'basketball.mp4'
    video = cv2.VideoCapture(video_path)

    boxes = []
    _, frame = video.read()
    gt = loader.get_gt()
    limit = loader.get_shape()
    ground_zero = [gt[0][0], gt[0][1]]
    total_samples = loader.test_sample(ground_zero, limit, [gt[0][2], gt[0][3]],k)
    bbox = Box([0, 0, 10, 10])
    while True:
        target_imgs = []
        for p in total_samples:
            box = [p[0], p[1], p[0] + gt[0][2], p[1] + gt[0][3]]
            bbox.next(box)
            target_img = bbox.get_state(frame)
            target_imgs.append(target_img)
        IOUs = global_goturn.get_output(target_imgs)
        max_iou_point = total_samples[IOUs.argmax()]
        predict_box = [max_iou_point[0], max_iou_point[1], max_iou_point[0] + gt[0][2],
                       max_iou_point[1] + gt[0][3]]
        boxes.append(predict_box)
        bbox.next(predict_box)
        bbox.draw(frame)
        cv2.imshow('DRG', frame)
        _, frame = video.read()
        if not _:
            break
        ground_zero = [predict_box[0], predict_box[1]]
        total_samples = loader.test_sample(ground_zero, limit, k=k)
        if cv2.waitKey(1) & 0xff == 27:
            break


test_ma()
