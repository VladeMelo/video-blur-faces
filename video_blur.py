import cv2
from ultralytics import YOLO
from utils import handle_face_blur

model = YOLO('yolov8n-face.pt')

video = cv2.VideoCapture('video.mp4')

fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('video_blurred.mp4', fourcc, fps, (width, height))

while True:
  success, frame = video.read()

  if not success:
    break

  face_results = model(frame, conf=0.3, stream=True)

  for face_result in face_results:
    for box in face_result.boxes:
      x1, y1, x2, y2 = box.xyxy[0]
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

      top_left_corner = (x1, y1)
      bottom_right_corner = (x2, y2)

      # cv2.rectangle(frame, top_left_corner, bottom_right_corner, (0, 255, 0), 2)

      frame = handle_face_blur(frame, top_left_corner, bottom_right_corner)

  output_video.write(frame)

  cv2.imshow('Result', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

output_video.release()
cv2.destroyAllWindows()