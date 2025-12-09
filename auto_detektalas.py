from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

model = YOLO("yolov8n.pt")

tracker = DeepSort(max_age=15, n_init=3, max_iou_distance=0.7)

cap = cv2.VideoCapture("video3.mp4")

line_y = 700

count = 0

prev_positions = {}

cv2.namedWindow("YOLO+DeepSort - Autószámlálás", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    dets = []
    
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            # car=2, bus=5, truck=7 (COCO osztályok)
            if cls in [2, 5, 7]:
                # tensor -> float lista
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                conf = float(box.conf[0])
                # DeepSort formátum: [[x, y, w, h], conf, class_id]
                dets.append([[x1, y1, w, h], conf, cls])

    tracks = tracker.update_tracks(dets, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        x, y, w, h = track.to_ltwh()
        cx = x + w / 2
        cy = y
        track_id = track.track_id

        if track_id not in prev_positions:
            prev_positions[track_id] = cy

        prev_y = prev_positions[track_id]

        #vonalátlépés
        if prev_y > line_y >= cy:
            count += 1
            print(f">>> AUTÓ SZÁMLÁLVA, ID={track_id}, összesen={count}")

        prev_positions[track_id] = cy

        cv2.rectangle(frame,
                      (int(x), int(y)),
                      (int(x + w), int(y + h)),
                      (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}",
                    (int(x), int(y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

    # Számlálóvonal kirajzolása
    cv2.line(frame, (0, line_y),
             (frame.shape[1], line_y),
             (255, 0, 0), 2)

    # Számláló kiírása
    cv2.putText(frame, f"Count: {count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)
    resized = cv2.resize(frame, (580,750), interpolation=cv2.INTER_AREA)
    cv2.imshow("YOLO+DeepSort - Autószámlálás",resized)

    # ESC-re kilép
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
