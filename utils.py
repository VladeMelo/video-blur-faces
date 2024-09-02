import cv2

def handle_face_blur(image, top_left_corner, bottom_right_corner):
  face_blurred = image[top_left_corner[1]:bottom_right_corner[1], top_left_corner[0]:bottom_right_corner[0]]

  image[top_left_corner[1]:bottom_right_corner[1], top_left_corner[0]:bottom_right_corner[0]] = cv2.blur(face_blurred, (20, 20))

  return image