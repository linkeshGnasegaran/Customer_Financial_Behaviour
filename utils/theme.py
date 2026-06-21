import streamlit as st


def load_css(path):
    with open(path, "r", encoding="utf-8") as file:
        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )


def load_theme(theme):
    if theme == "Dark Mode":
        load_css("css/dark.css")
    elif theme == "Light Mode":
        load_css("css/light.css")
    else:
        load_css("css/system.css")