
# IMPORTS
from keras.models import load_model
from keras.preprocessing import image
import cv2

# LOAD THE SELF TRAINED MODEL
my_model = load_model('self_trained_model/model/stop_cnn_model.h5')

# CAPTURE THE INPUT VIDEO
cap = cv2.VideoCapture('input_files/StopSign.mp4')

# INPUT VIDEO DETAILS
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_fps = cap.get(cv2.CAP_PROP_FPS)

# CREATE A VIDEO WRITER TO SAVE THE OUTPUT VIDEO
video_writer_self_model = cv2.VideoWriter('output_files/SelfModelOutput.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
                                          cap_fps, (cap_width, cap_height))

if not cap.isOpened():
    print('File not found or wrong codec used!')

# OPERATIONS
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # USING SELF TRAINED MODEL
        input_frame = image.img_to_array(frame)
        input_frame.resize(1, 150, 150, 3)
        prediction = my_model.predict(input_frame)
        if prediction == [[1.]]:
            cv2.putText(frame, 'Stop Sign Detected!', org=(10, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 0, 255),
                        thickness=1, lineType=cv2.LINE_AA)

        video_writer_self_model.write(frame)
        cv2.imshow('Self Model Detection', frame)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    else:
        break

cap.release()
video_writer_self_model.release()
cv2.destroyAllWindows()


# END OF CODE
