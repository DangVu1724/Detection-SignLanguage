import pyttsx3

# Khởi tạo pyttsx3
engine = pyttsx3.init()

# Lấy danh sách các giọng có sẵn
voices = engine.getProperty('voices')

# In ra các thông tin về giọng nói
for index, voice in enumerate(voices):
    print(f"Voice {index}:")
    print(f" - ID: {voice.id}")
    print(f" - Name: {voice.name}")
    print(f" - Gender: {voice.gender}")
    print(f" - Language: {voice.languages}\n")

# Ví dụ: Chọn giọng thứ 1 trong danh sách (có thể là giọng nữ)
engine.setProperty('voice', voices[1].id)

# Bạn có thể chọn giọng khác bằng cách đổi chỉ số [1] thành chỉ số mong muốn

# Phát âm thử với giọng đã chọn
engine.say("Hello, this is a test with the new voice.")
engine.runAndWait()
