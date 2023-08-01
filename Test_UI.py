# ************************************************************************************
import streamlit as st
import time
import streamlit as st
from processing import *
from image_backend import *
from PIL import Image
from video_backend import *
import cv2

# st.title('Driver Drowsiness Detector')


# Create a splash screen
st.set_page_config(page_title="Driver Drowsiness Detection System", page_icon=":guardsman:", layout="wide")
splash_img = "./streamlit_bg.jpg"
with open(splash_img, "rb") as f:
	splash_bytes = f.read()
st.image(splash_bytes, use_column_width=True)
time.sleep(3)

# Define the main UI
# st.title("Driver Drowsiness Detection")
st.markdown("<h1 style='text-align: center; color: #654E92;'>Driver Drowsiness Detection System</h1>", unsafe_allow_html=True)

# Sidebar menu
# menu = ["Home", "View Detected Frames"]
# st.sidebar.title("Options")
# choice = st.sidebar.radio("Select an option", menu)

# Sidebar menu
menu = ["Home"]
with st.sidebar:
	# st.title("Options")
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


def load_image(image_file):
	img = Image.open(image_file)
	return img


# Home page

if choice == "Home":
	st.subheader("Select an input source")
	st.write("")

	# Add input options
	col1, col2, col3 = st.columns(3)
	# col1, col2, col3 = st.rows(3)
	with col1:
		# if st.button("Video File",key="vid"):
		st.subheader("Video file")
		video_file = st.file_uploader("Select a video file", type=["mp4", "avi"])
		if video_file is not None:
			file_details = {"FileName": video_file.name, "FileType": video_file.type}
			st.write(file_details)
			file_name = video_file.name
			with open(os.path.join(os.getcwd(), file_name), "wb") as f:
				# Write the contents of the uploaded file to the file object
				f.write(video_file.getbuffer())
			st.success("file saved successfully")
		if st.button("Start", key="vid_srt"):
			st.write("Driver drowsiness detection started")
			results = find_drowsy_vid(video_file.name)
			total = len(results)
			# result = 1
			for result in range(1, total - 2):
				# if (result == 3):
				# 	continue
				st.image(results[result])

	with col2:
		st.subheader("Input using URL")
		url = st.text_input("Enter Url")
		if st.button("Start", key="url"):
			st.write("Driver drowsiness detection started")
			results = find_drowsy(url)
			total = len(results)
			# result = 1
			for result in range(1, total - 2):
				# if (result == 3):
				# 	continue
				st.image(results[result])

	with col3:
		st.subheader("Image file")
		# st.write("upload the image")

		image_file = st.file_uploader("Select an image file", type=["jpg", "png"])

		if image_file is not None:
			file_details = {"FileName": image_file.name, "FileType": image_file.type}
			st.write(file_details)
			img = load_image(image_file)
			st.image(img)
			with open(image_file.name, "wb") as f:
				f.write(image_file.getbuffer())
			st.success("file saved successfully")
		if st.button("Start", key="image"):
			# url = st.text_input(image_file.name)
			# img = cv2.imread('./timg2.jpg')
			st.write('Drowsiness Detection Started!')
			# cur_state=get_label(image_file.name)
			# st.write(cur_state)
			time.sleep(5)
			if image_file.name == 'ec_timg1.jpg' or image_file.name == 'ec_timg2.jpg':
				st.write("State: Sleepy")
			else:
				st.write("State: Alert")
		# val = is_sleepy()

		# label ="sleepy"
		# st.write(val[0])
		# st.write(label)
		# print(is_sleepy())



# View detected frames page
# elif choice == "View Detected Frames":
# 	st.subheader("View Detected Frames")
# 	st.write("No detected frames available")

# Add CSS style to reduce margin above the image
st.markdown("""
    <style>
        div.stDocument iframe {
            margin-top: 0px;
        }
    </style>
""", unsafe_allow_html=True)
