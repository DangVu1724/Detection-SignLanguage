import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3  
import threading

# Khởi tạo pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 150)

is_muted = False

# Tải mô hình đã huấn luyện
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Tải hình ảnh biểu tượng speaker và mute
volume_on_image = cv2.imread('speaker.jpg')  
mute_image = cv2.imread('mute.jpg')  

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

labels_dict = {0: 'V', 1: 'U', 2: 'B', 3: 'I', 4: 'E', 5: 'W', 6: 'D',7: 'Clear',8:'Love',9: 'Tha tim' ,10: 'Yeh' , 11: 'Phenikaa',12:'L',13:'Sorry',14:'Like',15:'Hello',16:'Stop',17:'Dislike',18:'Fighting',19:'Sleep',20:'Pray',21:'Eat',22:'H',23:'G'}

text_output = ""
previous_character_left = ""
previous_character_right = ""

white_color = (255, 255, 255)

last_added_time_left = time.time()
last_added_time_right = time.time()
delay_time = 1  # Thời gian trì hoãn giữa các lần thêm ký tự

confirmation_counter_left = 0
confirmation_counter_right = 0
required_confirmations = 10  # Số lần xác nhận cần thiết trước khi thêm ký tự

# Hàm chuyển văn bản thành giọng nói
def speak_text(text):
    global is_muted
    if not is_muted and text.strip() != "":
        engine.say(text)
        engine.runAndWait()

# Hàm bật/tắt mute khi nhấp chuột
def toggle_mute(event, x, y, flags, param):
    global is_muted
    if event == cv2.EVENT_LBUTTONDOWN:
        if x >= param[0] and x <= param[0] + 50 and y >= param[1] and y <= param[1] + 50:
            is_muted = not is_muted  # Chuyển đổi trạng thái mute/unmute

# Thiết lập callback cho cửa sổ OpenCV
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', toggle_mute, param=(10, 10))  # Vị trí góc trên bên trái

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    # Thêm hình biểu tượng speaker/mute vào góc trên bên trái
    if is_muted:
        if mute_image is not None:
            frame[10:60, 10:60] = cv2.resize(mute_image, (50, 50))  # Hiển thị hình mute
    else:
        if volume_on_image is not None:
            frame[10:60, 10:60] = cv2.resize(volume_on_image, (50, 50))  # Hiển thị hình volume on

    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 1:
            for hand_landmarks, hand_type in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Vẽ các điểm mốc bàn tay lên khung hình
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=white_color, thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=white_color, thickness=2)
                )

                label = hand_type.classification[0].label
                data_aux = []
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)
                if data_aux:
                    x_coords = data_aux[::2]
                    y_coords = data_aux[1::2]
                    min_x = min(x_coords)
                    min_y = min(y_coords)
                    normalized_data = []
                    for i in range(len(x_coords)):
                        normalized_data.append(x_coords[i] - min_x)
                        normalized_data.append(y_coords[i] - min_y)

                    if len(normalized_data) == 42:
                        normalized_data += [0] * 42  # Bổ sung thêm 42 giá trị 0 nếu thiếu

                    normalized_data = np.asarray(normalized_data).reshape(1, -1)

                    # Dự đoán và xác định độ tin cậy
                    try:
                        y_proba = model.predict_proba(normalized_data)[0]
                        prediction = model.predict(normalized_data)[0]
                        confidence = max(y_proba)
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        confidence = 0

                    predicted_character = labels_dict.get(int(prediction), "Unknown")

                    current_time = time.time()  # Thời gian hiện tại

                    if confidence >= 0.6:  # Ngưỡng độ tin cậy
                        if label == "Left":
                            if predicted_character == "Clear":
                                threading.Thread(target=speak_text, args=(text_output,)).start()
                                text_output = ""  # Xóa văn bản sau khi phát xong
                            elif predicted_character != previous_character_left:
                                confirmation_counter_left += 1
                            else:
                                confirmation_counter_left = 0

                            if confirmation_counter_left >= required_confirmations and current_time - last_added_time_left >= delay_time:
                                text_output += predicted_character + " "
                                previous_character_left = predicted_character
                                last_added_time_left = current_time
                                confirmation_counter_left = 0  # Reset bộ đếm

                                threading.Thread(target=speak_text, args=(predicted_character,)).start()

                            x_coords_pixels = [int(lm.x * W) for lm in hand_landmarks.landmark]
                            y_coords_pixels = [int(lm.y * H) for lm in hand_landmarks.landmark]
                            x1 = min(x_coords_pixels) - 10
                            y1 = min(y_coords_pixels) - 10
                            x2 = max(x_coords_pixels) + 10
                            y2 = max(y_coords_pixels) + 10

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

                        elif label == "Right":
                            if predicted_character == "Clear":
                                threading.Thread(target=speak_text, args=(text_output,)).start()
                                text_output = ""  
                            elif predicted_character != previous_character_right:
                                confirmation_counter_right += 1
                            else:
                                confirmation_counter_right = 0

                            if confirmation_counter_right >= required_confirmations and current_time - last_added_time_right >= delay_time:
                                text_output += predicted_character + " "
                                previous_character_right = predicted_character
                                last_added_time_right = current_time
                                confirmation_counter_right = 0  # Reset bộ đếm
                                threading.Thread(target=speak_text, args=(predicted_character,)).start()

                            x_coords_pixels = [int(lm.x * W) for lm in hand_landmarks.landmark]
                            y_coords_pixels = [int(lm.y * H) for lm in hand_landmarks.landmark]
                            x1 = min(x_coords_pixels) - 10
                            y1 = min(y_coords_pixels) - 10
                            x2 = max(x_coords_pixels) + 10
                            y2 = max(y_coords_pixels) + 10

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
                    else:
                        # Nếu độ tin cậy thấp, không hiển thị gì
                        predicted_character = "Unknown"

        elif len(results.multi_hand_landmarks) == 2:
            all_x_coords = []
            all_y_coords = []
            data_aux = []
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=white_color, thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=white_color, thickness=2)
                )
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]

                all_x_coords.extend(x_coords)
                all_y_coords.extend(y_coords)

                min_x = min(x_coords)
                min_y = min(y_coords)
                for i in range(len(x_coords)):
                    data_aux.append(x_coords[i] - min_x)
                    data_aux.append(y_coords[i] - min_y)

            normalized_data = np.asarray(data_aux).reshape(1, -1)

            try:
                y_proba = model.predict_proba(normalized_data)[0]
                prediction = model.predict(normalized_data)[0]
                confidence = max(y_proba)
            except Exception as e:
                print(f"Prediction error: {e}")
                confidence = 0

            predicted_character = labels_dict.get(int(prediction), "Unknown")

            current_time = time.time()  # Thời gian hiện tại

            if confidence >= 0.6:  # Ngưỡng độ tin cậy
                if predicted_character == "Clear":
                    threading.Thread(target=speak_text, args=(text_output,)).start()
                    text_output = ""  # Xóa văn bản sau khi phát xong
                elif predicted_character != previous_character_left and predicted_character != previous_character_right:
                    confirmation_counter_left += 1
                    confirmation_counter_right += 1

                    if (confirmation_counter_left >= required_confirmations and confirmation_counter_right >= required_confirmations
                            and current_time - last_added_time_left >= delay_time and current_time - last_added_time_right >= delay_time):
                        text_output += predicted_character + " "
                        previous_character_left = predicted_character
                        previous_character_right = predicted_character
                        last_added_time_left = current_time
                        last_added_time_right = current_time
                        confirmation_counter_left = 0
                        confirmation_counter_right = 0
                        threading.Thread(target=speak_text, args=(predicted_character,)).start()

                x1 = int(min(all_x_coords) * W) - 10
                y1 = int(min(all_y_coords) * H) - 10
                x2 = int(max(all_x_coords) * W) + 10
                y2 = int(max(all_y_coords) * H) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                # Nếu độ tin cậy thấp, không hiển thị gì
                predicted_character = "Unknown"

    max_text_length = 40
    if len(text_output) > max_text_length:
        text_output = ""
    cv2.rectangle(frame, (0, H - 50), (W, H), (255, 255, 255), -1)
    cv2.putText(frame, text_output, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
