import streamlit as st
from pathlib import Path


def _image_path(name: str) -> Path:
    """Return absolute Path to static/images/<name> relative to this file."""
    return Path(__file__).resolve().parent.joinpath("static", "images", name)


def _show_image(name: str, caption: str | None = None, width: int | None = None) -> None:
    """Safely show an image; warn if missing or fails to load."""
    path = _image_path(name)
    if not path.exists():
        st.warning(f"Image not found: {path.name}")
        return
    try:
        st.image(str(path), caption=caption, width=width)
    except Exception as e:
        st.warning(f"Unable to load image '{path.name}': {e}")


def show():
    """Render the About / People page."""
    st.title("About the MacFawn Lab")
    st.markdown("""
    The MacFawn Lab is a research lab at Grove City College that focuses on ___. 
    Led by Dr. Ian MacFawn, the lab specializes in [specific areas of research] and is dedicated to innovation, 
    collaboration, and mentoring the next generation of researchers.
    """)

    # Team member 1 - Dr. Ian MacFawn
    st.write("---")  # Separator
    col1, col2 = st.columns([1, 3])
    with col1:
        # Use resolved path and safe loader
        _show_image("macfawn.png", caption="Dr. Ian MacFawn", width=150)
    with col2:
        st.subheader("Dr. Ian MacFawn, Ph.D.")
        st.markdown("""
        *Principal Investigator*  
        Dr. Ian MacFawn is a dedicated researcher and professor at Grove City College. 
        His work focuses on [specific research focus].  
        He is passionate about advancing knowledge, mentoring students, and driving innovative projects within the lab.
        """)

    # Team member 2 - Joseph Shin
    st.write("---")  # Separator
    col1, col2 = st.columns([1, 3])
    with col1:
        _show_image("shin.png", caption="Joseph Shin", width=150)
    with col2:
        st.subheader("Joseph Shin")
        st.markdown("""
        *Undergraduate Researcher - Grove City College*  
        Joseph focuses on [specific research focus or role in the lab]. 
        His interests include [areas of interest or contribution to the lab].
        """)

    # Team member 3 - Luca Wilkins
    st.write("---")  # Separator
    col1, col2 = st.columns([1, 3])
    with col1:
        _show_image("wilkins.png", caption="Luka Wilkins", width=150)
    with col2:
        st.subheader("Luka Wilkins")
        st.markdown("""
        *Undergraduate Researcher - Grove City College*  
        Luka specializes in [specific research focus or role in the lab]. 
        He is enthusiastic about [areas of interest or contribution to the lab].
        """)

    # Team member 4 - Sarah Zhou
    st.write("---")  # Separator
    col1, col2 = st.columns([1, 3])
    with col1:
        _show_image("macfawn.png", caption="Sarah Zhou", width=150)
    with col2:
        st.subheader("Sarah Zhou")
        st.markdown("""
        *Undergraduate Researcher - Grove City College*  
        Sarah's research involves [specific research focus]. 
        She contributes to [specific areas of interest or key tasks within the lab].
        """)

    # Team member 5 - Mike Christensen
    st.write("---")  # Separator
    col1, col2 = st.columns([1, 3])
    with col1:
        _show_image("christensen.png", caption="Mike Christensen", width=150)
    with col2:
        st.subheader("Mike Christensen")
        st.markdown("""
        *Undergraduate Researcher - Grove City College*  
        Mike focuses on [specific research focus or role in the lab]. 
        His contributions include [specific areas of interest or lab projects].
        """)

    # Team member 6 - Asher Alley
    st.write("---")  # Separator
    col1, col2 = st.columns([1, 3])
    with col1:
        _show_image("alley.png", caption="Asher Alley", width=150)
    with col2:
        st.subheader("Asher Alley")
        st.markdown("""
        *Undergraduate Researcher - Grove City College*  
        Asher focuses on [specific research focus or role in the lab]. 
        His contributions include [specific areas of interest or lab projects].
        """)

    st.write("---")  # Final separator
    st.markdown("More team members will be added soon!")

    # Contact Us Page (rendered here as part of About if needed)
    # You can add a separate function for contact if desired.
