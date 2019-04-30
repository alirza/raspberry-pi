import face_recognition
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

camera = PiCamera()
camera.resolution= (540,380)
camera.framerate=32
rawCapture = PiRGBArray(camera,size =(540,380))

time.sleep(0.1)

# Load a sample picture and learn how to recognize it.
me_image = face_recognition.load_image_file("your_pic.jpg")
me_face_encoding = face_recognition.face_encodings(me_image)[0]

# Load a second sample picture and learn how to recognize it.
some_one_else_image = face_recognition.load_image_file("some_one_else.jpg")
some_one_else_face_encoding = face_recognition.face_encodings(some_one_else_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    me_face_encoding,
    some_one_else_face_encoding
]
known_face_names = [
    "me",
    "someone else"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    frame=image.array
    
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame
    
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)



        
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF
   
    rawCapture.truncate(0)
    
    if key == ord("q"):
        break

cv2.destroyAllWindows()
