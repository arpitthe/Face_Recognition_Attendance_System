import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Use CAP_DSHOW on Windows for more reliable camera access
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# --- Load and encode known faces ---
rishika_image = face_recognition.load_image_file("IMG/Rishika(1).jpg")
rishika_encoding = face_recognition.face_encodings(rishika_image)[0]

vansh_image = face_recognition.load_image_file("IMG/vansh(1).jpg")
vansh_encoding = face_recognition.face_encodings(vansh_image)[0]

sarthak_image = face_recognition.load_image_file("IMG/Sarthak(1).jpg")
sarthak_encoding = face_recognition.face_encodings(sarthak_image)[0]

known_face_encodings = [
    rishika_encoding,
    vansh_encoding,
    sarthak_encoding
]

known_face_names = [
    "Rishika",
    "vansh",
    "Sarthak"
]

students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# CSV setup
current_date = datetime.now().strftime("%Y-%m-%d")
f = open(current_date + ".csv", "w+", newline='')
lnwriter = csv.writer(f)

while True:
    ret, frame = video_capture.read()
    if not ret or frame is None:
        print("⚠️ Could not access the webcam")
        break

    # Resize frame to 1/4 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert BGR to RGB (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        # Find all faces in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Get face encodings for all detected faces
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face matches any known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            face_names.append(name)

            # If recognized and in students list, mark attendance
            if name != "Unknown" and name in students:
                students.remove(name)
                print(f"Attendance marked for {name}")
                current_time = datetime.now().strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])
                f.flush()  # Ensure data is written immediately

    # Toggle processing to save resources
    process_this_frame = not process_this_frame

    # Display results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Attendance System', frame)

    # Hit 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()