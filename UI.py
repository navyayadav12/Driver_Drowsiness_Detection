import streamlit as st
import time

# Create a splash screen
st.set_page_config(page_title="Driver Drowsiness Detection", page_icon=":guardsman:", layout="wide")
splash_img = "./splash_screen.jpg"
with open(splash_img, "rb") as f:
	splash_bytes = f.read()
st.image(splash_bytes, use_column_width=True)
time.sleep(3)

# Define the main UI
# st.title("Driver Drowsiness Detection")
st.markdown("<h1 style='text-align: center; color: #654E92;'>Driver Drowsiness Detection</h1>", unsafe_allow_html=True)

# Sidebar menu
# menu = ["Home", "View Detected Frames"]
# st.sidebar.title("Options")
# choice = st.sidebar.radio("Select an option", menu)

# Sidebar menu
menu = ["Home", "View Detected Frames"]
with st.sidebar:
	st.title("Options")
	choice = st.radio("", menu, index=0)

# Styling for sidebar
st.markdown(
	"""
	<style>
	.sidebar .sidebar-content {
		font-size: 18px;
		color: #444444;
	}
	</style>
	""",
	unsafe_allow_html=True
)

# Home page
if choice == "Home":
	st.subheader("Select an input source")
	st.write("")

	# Add input options
	col1, col2, col3 = st.columns(3)
	with col1:
		if st.button("Webcam"):
			st.subheader("Webcam Settings")
			webcam_fps = st.slider("Select FPS", min_value=1, max_value=30, value=15, step=1)
			webcam_res = st.selectbox("Select Resolution", ["320x240", "640x480", "1280x720"])
			if st.button("Start"):
				st.write("Driver drowsiness detection started")

	with col2:
		if st.button("Video File"):
			st.subheader("Upload video file")
			video_file = st.file_uploader("Select a video file", type=["mp4", "avi"])
			if st.button("Start"):
				st.write("Driver drowsiness detection started")

	with col3:
		if st.button("Image File"):
			st.subheader("Upload image file")
			image_file = st.file_uploader("Select an image file", type=["jpg", "png"])
			if st.button("Start"):
				st.write("Driver drowsiness detection started")

# View detected frames page
elif choice == "View Detected Frames":
	st.subheader("View Detected Frames")
	st.write("No detected frames available")

# Add CSS style to reduce margin above the image
st.markdown("""
    <style>
        div.stDocument iframe {
            margin-top: 0px;
        }
    </style>
""", unsafe_allow_html=True)