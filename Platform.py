import streamlit as st
from processing import *

st.title('Driver Drowsiness Detector')
url = st.text_input("Enter Url")
if st.button("Detect"):
	results = find_drowsy(url)
	total = len(results)
	#result = 1
	for result in range(1,total - 1):
		st.image(results[result])
	# for result in results:
	#    st.image(result)
