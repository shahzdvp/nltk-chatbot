import streamlit as st

with st.sidebar:
    st.write("I appear in sidebar!")

# initialize once
if "messages" not in st.session_state:
    st.session_state.messages = []

# add to it
st.session_state.messages.append("hello")

with st.chat_message("user"):
    st.write("Hello!")          # shows user bubble

with st.chat_message("assistant"):
    st.write("Hi there!")       # shows bot bubble