import sqlite3
import hashlib
import PIL
import streamlit as st
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from pathlib import Path  # Add this import

# Setting page layout
st.set_page_config(
    page_title="Object Detection menggunakan YOLOv11",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Local Modules
import settings
import helper

# --- USER AUTHENTICATION ---

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def verify_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT name, password FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    if user and user[1] == hash_password(password):
        return user[0]
    return None


if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
    st.session_state['username'] = None
    st.session_state['name'] = None

if st.session_state['authentication_status'] != True:
    st.header("Login 🍎")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        name = verify_user(username, password)
        if name:
            st.session_state['authentication_status'] = True
            st.session_state['username'] = username
            st.session_state['name'] = name
            st.success("Berhasil login")
            st.experimental_rerun()
        else:
            st.error("Username atau password salah")
else:

    def main():
        # Initialize dark mode session state
        if 'dark_mode' not in st.session_state:
            st.session_state.dark_mode = False

        # Main page heading
        st.title("Apple Detection🍎")

        # Sidebar logout
        if st.sidebar.button("Logout"):
            st.session_state['authentication_status'] = None
            st.session_state['name'] = None
            st.session_state['username'] = None
            st.experimental_rerun()

        st.sidebar.title(f"Welcome, {st.session_state['name']}")
        st.sidebar.header("🍎Apel Indonesia")

        # Menu Options
        menu_options = {
            "Home": "🏠 Beranda",
            "Detection": "🔍 Deteksi",
            "History": "📝 Riwayat"
        }
        selected_menu = st.sidebar.radio(
            "Select Menu", list(menu_options.keys()),
            format_func=lambda x: menu_options[x]
        )

        # --- HOME SECTION ---
        if selected_menu == "Home":
            st.header("Selamat datang di aplikasi penerapan algoritma YOLOv11 untuk deteksi penyakit pada buah apel berbasis web!")
            st.write(
                """
                Aplikasi ini membantu deteksi dan segmentasi penyakit pada buah apel menggunakan model YOLOv11.

                Gunakan sidebar untuk navigasi:
                - *Home*: Deskripsi aplikasi.
                - *Detection*: Pilih model dan sumber data untuk deteksi/segmentasi.
                - *History*: Riwayat hasil deteksi.
                """
            )

            st.markdown("---")
            st.subheader("Pengertian Penyakit Buah Apel dan Cara Penanggulangannya")
            st.write(
                """
**1. Apple Scab**
- *Pengertian*: Penyakit oleh *Venturia inaequalis*. Bercak zaitun hingga coklat pada kulit buah dan daun.
- *Penanggulangan*: 
  - Pemangkasan cabang terinfeksi.
  - Penyemprotan fungisida sulfur atau chlorothalonil.
  - Pengumpulan daun dan buah gugur.

**2. Black Spot**
- *Pengertian*: Disebabkan *Alternaria alternata*, bercak hitam berlubang.
- *Penanggulangan*:
  - Pemangkasan untuk sirkulasi udara.
  - Aplikasi fungisida captan.
  - Sanitasi kebun.

**3. Black Rot**
- *Pengertian*: Jamur *Botryosphaeria obtusa*, busuk hitam pada buah dan lesi pada ranting.
- *Penanggulangan*:
  - Pemangkasan ranting terserang.
  - Fungisida mancozeb.
  - Sterilisasi alat potong.

**4. Fly Speck**
- *Pengertian*: Jamur *Schizothyrium pomi*, bintik hitam kecil pada kulit buah.
- *Penanggulangan*:
  - Kontrol tanaman inang alternatif.
  - Fungisida tembaga.
  - Penyemprotan saat cuaca kering.
                """
            )

            # Menampilkan gambar setelah pengertian penyakit
            col1, col2 = st.columns(2)
            with col1:
                st.image("images/yak apple.jpg", caption="Overview Image", use_column_width=True)
            with col2:
                st.image("images/yak apple hasil.png", caption="Overview Webcam", use_column_width=True)

        # --- DETECTION SECTION ---
        elif selected_menu == "Detection":
            st.sidebar.header("ML Model Config")
            model_type = st.sidebar.radio("Select Task", ['Detection'])
            confidence = float(st.sidebar.slider(
                "Select Model Confidence", 25, 100, 40)) / 100
            model_path = Path(settings.DETECTION_MODEL)
            try:
                model = helper.load_model(model_path)
            except Exception as ex:
                st.error(f"Unable to load model: {model_path}")
                st.error(ex)

            st.sidebar.header("Image/Video Config")
            source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

            if source_radio == settings.IMAGE:
                source_img = st.sidebar.file_uploader(
                    "Choose an image...", type=("jpg","jpeg","png","bmp","webp"))
                col1, col2 = st.columns(2)
                with col1:
                    try:
                        if source_img:
                            img = PIL.Image.open(source_img)
                            st.image(img, caption="Uploaded Image", use_column_width=True)
                        else:
                            default_img = PIL.Image.open(settings.DEFAULT_IMAGE)
                            st.image(default_img, caption="Default Image", use_column_width=True)
                    except Exception as ex:
                        st.error("Error membuka gambar.")
                        st.error(ex)
                with col2:
                    if source_img and st.sidebar.button('Detect Objects'):
                        res = model.predict(img, conf=confidence)
                        boxes = res[0].boxes
                        plotted = res[0].plot()[:,:,::-1]
                        st.image(plotted, caption="Detected Image", use_column_width=True)
                        if 'history' not in st.session_state:
                            st.session_state.history = []
                        st.session_state.history.append({
                            "image": img,
                            "result": plotted,
                            "boxes": boxes
                        })
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)

            elif source_radio == settings.WEBCAM:
                st.subheader("📷 Deteksi Webcam Real-time")

                is_display_tracker, tracker = helper.display_tracker_options()

                class VideoProcessor(VideoProcessorBase):
                    def __init__(self):
                        self.model = model
                        self.conf = confidence
                        self.tracker = tracker
                        self.is_display_tracker = is_display_tracker
                        self.last_frame = None

                    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                        try:
                            img = frame.to_ndarray(format="bgr24")
                            img = cv2.resize(img, (720, int(720 * (9 / 16))))
                            if self.is_display_tracker:
                                res = self.model.track(img, conf=self.conf, persist=True, tracker=self.tracker)
                            else:
                                res = self.model.predict(img, conf=self.conf)
                            result_img = res[0].plot()
                            self.last_frame = result_img  # Save latest frame
                            return av.VideoFrame.from_ndarray(result_img, format="bgr24")
                        except Exception as e:
                            print("Error in recv:", e)
                            return frame

                webrtc_ctx = webrtc_streamer(
                    key="apel-webcam",
                    video_processor_factory=VideoProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                    rtc_configuration=None
                )

                if webrtc_ctx.video_processor:
                    webrtc_ctx.video_processor.conf = confidence

                    st.markdown("---")
                    if st.button("📸 Simpan Deteksi Sekarang"):
                        frame = webrtc_ctx.video_processor.last_frame
                        if frame is not None:
                            image = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            st.image(image, caption="📸 Snapshot disimpan", use_column_width=True)

                            if 'history' not in st.session_state:
                                st.session_state.history = []

                            st.session_state.history.append({
                                "image": image,
                                "result": frame,
                                "boxes": []
                            })
                            st.success("Berhasil menyimpan snapshot dari webcam.")
                        else:
                            st.warning("Belum ada frame ditangkap.")


        # --- HISTORY SECTION ---
        elif selected_menu == "History":
            st.header("Detection History")
            if st.session_state.get('history'):
                for idx, rec in enumerate(st.session_state.history):
                    st.subheader(f"Detection {idx+1}")
                    st.image(rec['image'], caption=f"Image {idx+1}", use_column_width=True)
                    st.image(rec['result'], caption=f"Result {idx+1}", use_column_width=True)
                    with st.expander(f"Results {idx+1}"):
                        for box in rec['boxes']:
                            st.write(box.data)
            else:
                st.write("Belum ada riwayat deteksi.")

        # --- DARK MODE TOGGLE ---
        st.sidebar.markdown("---")
        st.session_state.dark_mode = st.sidebar.checkbox(
            'Dark Mode', value=st.session_state.dark_mode)
        if st.session_state.dark_mode:
            st.sidebar.markdown(
                "<p style=\"color:white; font-size:12px;\">❗ Gunakan saat Streamlit dalam mode gelap ❗</p>",
                unsafe_allow_html=True
            )
            st.markdown(
                """
                <style>
                [data-testid="stAppViewContainer"] {background-color:#1E1E1E; color:#FFF;}
                [data-testid="stSidebar"] {background-color:#333; color:#FFF;}
                [data-testid="stExpander"] {background-color:#2E2E2E;}
                </style>
                """, unsafe_allow_html=True)
        else:
            st.sidebar.markdown(
                "<p style=\"color:black; font-size:12px;\">❗ Gunakan saat Streamlit dalam mode terang ❗</p>",
                unsafe_allow_html=True
            )
            st.markdown(
                """
                <style>
                [data-testid="stAppViewContainer"] {background-color:#ffffe0; color:#000;}
                [data-testid="stSidebar"] {background-color:#FFA62F; color:#000;}
                </style>
                """, unsafe_allow_html=True)

        st.sidebar.image("images/poon.png", use_column_width=True)

    if __name__ == "__main__":
        main()
