
# IMPORTS
import cv2
import csv
import can

# LOADING THE CASCADE CLASSIFIER
stop_sign_cascade = cv2.CascadeClassifier('input_files/stop_data.xml')

# CAPTURING THE INPUT VIDEO
cap = cv2.VideoCapture('input_files/StopSign.mp4')

# INPUT VIDEO DETAILS
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_fps = cap.get(cv2.CAP_PROP_FPS)

# CREATE A VIDEO WRITER TO SAVE THE OUTPUT VIDEO
video_writer_cascade = cv2.VideoWriter('output_files/CascadeOutput.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                       cap_fps, (cap_width, cap_height))

# CREATE A CSV FILE TO SAVE BOUNDING BOXES
file_to_output = open('output_files/bounding_boxes.csv', mode='w', newline='')
csv_writer = csv.writer(file_to_output, delimiter=',')
csv_writer.writerows([['bounding boxes'], ['x1', 'y1', 'width', 'height']])


# PREPARE CAN FOR SENDING MESSAGES
can_msg = can.Message(arbitration_id=0x018, dlc=8, data=[0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])


def send_bounding_boxes_as_can_msg(x1, y1, width, height):
    if x1 < 256:
        can_msg.data[1] = x1
    else:
        can_msg.data[1] = x1 & 0x00ff
    can_msg.data[2] = y1
    can_msg.data[3] = width
    can_msg.data[4] = height


if not cap.isOpened():
    print('File not found or wrong codec used!')

# OPERATIONS
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # USING CASCADE CLASSIFIER
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stop_signs = stop_sign_cascade.detectMultiScale(gray, 1.1, 6)

        for (x, y, w, h) in stop_signs:
            # DRAWING BOUNDING BOXES
            cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=2)
            send_bounding_boxes_as_can_msg(x, y, w, h)
            print(can_msg)
            csv_writer.writerow([x, y, w, h])

        # SAVE THE PREDICTED FRAMES TO THE OUTPUT VIDEO
        video_writer_cascade.write(frame)
        cv2.imshow('Stop Sign Detection', frame)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    else:
        break

cap.release()
video_writer_cascade.release()
file_to_output.close()
cv2.destroyAllWindows()


# END OF CODE
