import time
import cv2

# print(cv2.__version__)
cap = cv2.VideoCapture('sample.mp4')
l_count = r_count = tmp_x = tmp_y = count = 0
flag = None

while True:
    ret, img = cap.read()

    if not ret: break

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (21, 21), 0)
    ret, thresh = cv2.threshold(grey, 100, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours):
        c = max(contours, key=cv2.contourArea)
        coor_x, coor_y, width, height = cv2.boundingRect(c)
        count += 1; tmp_x += coor_x; tmp_y += coor_y
        if coor_x + width // 2 > 320:
            r_count += 0 if flag else 1
            color = (255, 0, 0)
            flag = True
        else:
            l_count += 1 if flag else 0
            color = (0, 0, 255)
            flag = False

        cv2.circle(img, (coor_x + width // 2, coor_y + height // 2), width // 2, color, 2)
        cv2.line(img, (0, coor_y + height // 2), (640, coor_y + height // 2), color, 1)
        cv2.line(img, (coor_x + width // 2, 0), (coor_x + width // 2, 480), color, 1)
        cv2.putText(img, "x: " + str(coor_x) + " y: " + str(coor_y) + " w: " + str(width) + " h: " + str(height),
                    (0, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))
        cv2.putText(img, "left: " + str(l_count) + " right: " + str(r_count), (0, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))
        cv2.putText(img,
                    "Distance: " + str(((coor_x + width // 2 - 320) ** 2 + (coor_y + height // 2 - 240) ** 2) ** 0.5),
                    (0, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))
        cv2.putText(img, "Area percentage: " + str(100 * height * width / 640 / 480) + " %", (0, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0))

        cv2.rectangle(img, (220, 140), (420, 340), (0, 0, 0), 2)

    cv2.imshow('DVB 2 laba', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.05)

print("x:", tmp_x / count, "y:", tmp_y / count)

cap.release()
cv2.destroyAllWindows()
