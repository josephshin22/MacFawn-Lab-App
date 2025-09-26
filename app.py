import time
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from plotnine import ggplot, aes, geom_boxplot
import plotly.graph_objects as go
from plotnine import ggplot, aes, geom_boxplot, geom_jitter, labs, theme_minimal, theme
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind


# Set page configuration
st.set_page_config(
    page_title="Bio Lab",
    page_icon="🧬",
    layout="wide",
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page", ["Home", "Troubleshoot", "(1) Submit image data", "(2) Run Preliminary QuPath Analysis", "(3) Run Full QuPath Analysis", "(4) Create graphs / visualizations", "About", "Contact Us"])

# Home Page
if page == "Home":
    st.title("Welcome to the MacFawn Lab")
    st.markdown("""
    This website is a hub for all ongoing research and experiments in our lab.
    Explore our research, utilize our proprietary software, or get in touch with us.
    """)

    st.image("./static/images/qupath2.png",
             caption="from gcc.edu", use_column_width=True)

# Troubleshoot
elif page == "Troubleshoot":
    st.title("Troubleshoot")
    st.markdown("Fill out the form below to submit new data to the lab.")

    with st.form("experiment_form"):
        name = st.text_input("Researcher Name")
        email = st.text_input("Email")
        experiment_title = st.text_input("Experiment Title")
        description = st.text_area("Experiment Description")
        file = st.file_uploader("Upload Experiment Data (CSV)", type=["csv"])
        submitted = st.form_submit_button("Submit")

        if submitted:
            if name and email and experiment_title and description:
                st.success("Experiment submitted successfully!")
                if file:
                    st.write("Uploaded file preview:")
                    df = pd.read_csv(file)
                    st.dataframe(df.head())
            else:
                st.error("Please fill out all required fields.")

elif page == "(1) Submit image data":

    st.title("Submit Image Data")
    st.markdown("Enter image data for analysis in QuPath")

    # File uploader for training images
    training_file = st.file_uploader("Upload training images", type=[
                                     "tif", "png", "jpg", "jpeg", "svs"], key="training")

    # File uploader for actual data images
    actual_file = st.file_uploader("Upload actual images", type=[
                                   "tif", "png", "jpg", "jpeg", "svs"], key="actual")

    # Handle training data
    if training_file is not None:
        st.image(training_file, caption="Training Image",
                 use_column_width=True)
        st.success("Training image uploaded successfully!")

        # Save training file
        with open(f"temp/training_{training_file.name}", "wb") as f:
            f.write(training_file.getbuffer())

        st.write(f"Saved as temp/training_{training_file.name}")

    # Handle actual data
    if actual_file is not None:
        st.image(actual_file, caption="Actual Data Image",
                 use_column_width=True)
        st.success("Actual data image uploaded successfully!")

        # Save actual data file
        with open(f"temp/actual_{actual_file.name}", "wb") as f:
            f.write(actual_file.getbuffer())

        st.write(f"Saved as temp/actual_{actual_file.name}")

    # Placeholder for QuPath API integration
    # You can send these images to QuPath for processing after upload
    # Example:
    # response = requests.post("YOUR_QUATH_API_URL", files={"file": actual_file})
    # st.json(response.json())


elif page == "(2) Run Preliminary QuPath Analysis":
    st.title("Enter csv / data analysis")
    if st.button("Run QuPath Analysis"):
        with st.spinner("Running analysis..."):
            time.sleep(3)  # Simulate processing time

            # Example results data
            data = {
                "Image": ["Image1", "Image2", "Image3"],
                "Cell Count": [120, 95, 110],
                "Area": [300.5, 280.3, 290.1]
            }
            df = pd.DataFrame(data)

            # Save results to CSV
            csv_filename = "qupath_results.csv"
            df.to_csv(csv_filename, index=False)

            st.success("Analysis completed! Download results below.")

            # Provide a download button
            with open(csv_filename, "rb") as f:
                st.download_button(
                    label="Download Results",
                    data=f,
                    file_name=csv_filename,
                    mime="text/csv"
                )

elif page == "(3) Run Full QuPath Analysis":
    st.title("Enter csv / data analysis")
    st.markdown("Run QuPath via api call to analyze images")
    if st.button("Run QuPath Analysis"):
        with st.spinner("Running analysis..."):
            time.sleep(3)  # Simulate processing time

            # Example results data
            data = {
                "Image": ["Image1", "Image2", "Image3"],
                "Cell Count": [120, 95, 110],
                "Area": [300.5, 280.3, 290.1]
            }
            df = pd.DataFrame(data)

            # Save results to CSV
            csv_filename = "qupath_results.csv"
            df.to_csv(csv_filename, index=False)

            st.success("Analysis completed! Download results below.")

            # Provide a download button
            with open(csv_filename, "rb") as f:
                st.download_button(
                    label="Download Results",
                    data=f,
                    file_name=csv_filename,
                    mime="text/csv"
                )

elif page == "(4) Create graphs / visualizations":
    st.title("Enter csv / data analysis")
    st.markdown("Add your dataset for analysis")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        # Read the data
        df = pd.read_csv(uploaded_file)

        # Check if the required columns exist
        required_columns = ["ROI", "Num CD20+", "Num Ki67+", "Classification"]
        if all(col in df.columns for col in required_columns):
            # df = df.drop(columns=['Num AID+: CD20+: CD4+', 'Num AID+: Ki67+: CD20+: CD4+', 'Num CD20+: CD4+', 'Num CD21+: AID+: CD20+: CD4+', 'Num CD21+: AID+: Ki67+: CD20+: CD4+', 'Num CD21+: CD20+: CD4+',
            #  'Num Ki67+: CD20+: CD4+', 'Num PNAd+: AID+: CD20+: CD4+', 'Num PNAd+: CD20+: CD4+', 'Num PNAd+: CD21+: AID+: CD20+: CD4+', 'Num PNAd+: CD21+: AID+: Ki67+: CD20+: CD4+'])

            st.write(df)
            # Replace 0 values in "Num CD20+" with NaN to avoid division by zero
            df["Num CD20+"] = df["Num CD20+"].replace(0, float("nan"))
            col1, col2 = st.columns(2)

            # TASK: Define Lymphoid Aggregate (LA), Tertiary Lymphoid Structure (TLS), and TLS + Germinal Center (GC)

            with col1:
                threshold = st.slider(
                    "Select Minimum Number of Detections", min_value=1, max_value=100, value=50, key=1)
                # Number of Lymphoid Aggregate (LA) positive annotations by patient
                # filter by num detections (cells) - 50 or more
                df_la = df[df["Num Detections"] >= threshold]
                count_la_per_patient = df_la[df_la["Classification"] == "LA"].groupby(
                    "Image").size().reset_index(name="LA Count")
                # Create a bar chart
                fig = px.bar(count_la_per_patient, x="Image", y="LA Count",
                             title="Count of Lymphoid Aggregate (LA) Classifications per Patient", labels={"Image": "Patient (Image ID)", "LA Count": "# of LA Classifications"})
                st.plotly_chart(fig)
            with col2:
                # % GC+ per patient
                # Add Streamlit slider for adjusting the threshold for "Num Detections"
                threshold_la = st.slider(
                    "Select Minimum Number of Detections (LA)", min_value=1, max_value=100, value=50, key=2)
                threshold_gc = st.slider(
                    "Select Minimum Number of Detections (CD21+)", min_value=1, max_value=100, value=50, key=3)
                # Filter "LA" rows based on the selected threshold for "Num Detections"

                # Filter separately for LA and CD21+
                la_only_filtered = df[(df["Classification"] == "LA") & (
                    df["Num Detections"] > threshold_la)]
                gc_only_filtered = df[(
                    df["Classification"] == "CD21+") & (df["Num Detections"] > threshold_gc)]

                # Group by "Image" and count occurrences for each classification
                count_la_per_patient = la_only_filtered.groupby(
                    "Image").size().reset_index(name="LA Count")
                count_gc_per_patient = gc_only_filtered.groupby(
                    "Image").size().reset_index(name="CD21+ Count")

                # Merge the two DataFrames
                percentage_df = count_gc_per_patient.merge(
                    count_la_per_patient, on="Image", how="inner")

                # Calculate the percentage of GC+ relative to LA
                percentage_df["Percentage GC+"] = (
                    percentage_df["CD21+ Count"] / percentage_df["LA Count"]) * 100

                # Create the bar chart
                fig = px.bar(percentage_df,
                             x="Image",
                             y="Percentage GC+",
                             title="Percentage of GC+ (CD21+) Classifications Relative to LA per Patient",
                             labels={"Image": "Patient (Image ID)", "% GC+": "Percentage of GC+ Classifications (%)"})

                # Display the plot in Streamlit
                st.plotly_chart(fig)

            col3, col4 = st.columns(2)
            with col3:
                plot_type = st.radio("Select Plot Type", [
                                     "Interactive (Plotly)", "Static (ggplot)"])
                # Numerators for graph
                df["numerator"] = df[[
                    "Num AID+: Ki67+: CD20+", "Num AID+: Ki67+: CD20+: CD4+", "Num CD21+: AID+: Ki67+: CD20+",
                    "Num CD21+: AID+: Ki67+: CD20+: CD4+", "Num CD21+: Ki67+: CD20+",
                    "Num Ki67+: CD20+", "Num Ki67+: CD20+: CD4+",
                ]].sum(axis=1)

                # Denominators for graph

                df["denominator"] = df[[
                    "Num AID+: CD20+", "Num AID+: CD20+: CD4+", "Num CD20+",
                    "Num CD20+: CD4+", "Num CD21+: AID+: CD20+", "Num CD21+: AID+: CD20+: CD4+",
                    "Num CD21+: CD20+", "Num CD21+: CD20+: CD4+"]].sum(axis=1)

                df["Ki67&CD20/CD20"] = (df["numerator"] /
                                        (df["denominator"] + df["numerator"])) * 100
                df["Ki67&CD20/CD20"].replace([np.inf, -np.inf],
                                             np.nan, inplace=True)

                df["Classification"] = df["Classification"].astype("str")
                df["Classification"] = df["Classification"].replace(
                    "CD21+", "GC")
                df["Classification"] = df["Classification"].replace(
                    "LA", "non-GC")

                # Compute Q1 (25th percentile) and Q3 (75th percentile)
                Q1 = df["Ki67&CD20/CD20"].quantile(0.25)
                Q3 = df["Ki67&CD20/CD20"].quantile(0.75)
                IQR = Q3 - Q1

                # Define outlier boundaries
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Filter out outliers - NOT USING THIS RIGHT NOW
                df_filtered = df[(df["Ki67&CD20/CD20"] >= lower_bound)
                                 & (df["Ki67&CD20/CD20"] <= upper_bound)]

                if plot_type == "Interactive (Plotly)":

                    fig = px.box(df,
                                 x="Classification",
                                 y="Ki67&CD20/CD20",
                                 title="B Cell Proliferation",
                                 points=False,  # Hide outliers
                                 color="Classification",
                                 color_discrete_sequence=["#ff8000"])  # Hex code for orange

                    jitter = px.strip(df,
                                      x="Classification",
                                      y="Ki67&CD20/CD20",
                                      color="Classification",
                                      color_discrete_sequence=["#ff8000"])  # Hex code for orange

                    # Get the traces from the jitter plot and add them to the box plot
                    for trace in jitter.data:
                        trace.marker.opacity = 0.7  # Make points semi-transparent
                        trace.marker.size = 4       # Set point size
                        fig.add_trace(trace)

                    # Update layout for better visualization
                    fig.update_layout(
                        # Adjust left and right margins
                        margin=dict(l=10, r=10),

                        boxmode='group',
                        boxgap=0.1,
                        legend_title_text='Classification',
                        xaxis_title='Classification',
                        yaxis_title='% Ki67+/CD20+ Cells',
                        showlegend=False  # Hide legend if duplicated
                    )

                    st.plotly_chart(fig)

                else:
                    # Create the plot
                    plot = (
                        ggplot(df, aes(x="Classification",
                                       y="Ki67&CD20/CD20", color="Classification"))
                        # Boxplot without outliers
                        + geom_boxplot(outlier_shape=None,
                                       fill="#ff8000", alpha=0.5)
                        # Jittered points
                        + geom_jitter(size=2, width=0.2,
                                      alpha=0.7, color="#ff8000")
                        + labs(title="B Cell Proliferation",
                               x="Classification", y="% Ki67+/CD20+ Cells", )
                        + theme(legend_position="none")  # Hide legend
                    )

                    # Show plot
                    fig = plot.draw()
                    st.pyplot(fig)

                # Drop NaNs to avoid issues in t-test -- not necessary but do anyways for now
                group_gc = df[df["Classification"] ==
                              "GC"]["Ki67&CD20/CD20"].dropna()
                group_non_gc = df[df["Classification"] ==
                                  "non-GC"]["Ki67&CD20/CD20"].dropna()

                t_stat, p_value = ttest_ind(
                    group_gc, group_non_gc, equal_var=False)

                # Display the result in Streamlit
                st.markdown(
                    f"**Two-tailed t-test**: t = {t_stat:.3f}, p = {p_value:.3e}")
                if p_value < 0.05:
                    st.markdown(
                        "**Result:** Significant difference (p < 0.05)")
                else:
                    st.markdown(
                        "**Result:** No significant difference (p ≥ 0.05)")

            with col4:
                plot_type_2 = st.radio(
                    "Select Plot Type",
                    ["Interactive (Plotly)", "Static (ggplot)"],
                    key="plot_toggle_2"
                )
                # Numerators for graph
                df["numerator"] = df[[
                    "Num AID+: CD20+", "Num AID+: Ki67+: CD20+", "Num CD21+: AID+: CD20+", "Num CD21+: AID+: Ki67+: CD20+"
                ]].sum(axis=1)

                # Denominators for graph

                df["denominator"] = df[[
                    "Num CD20+", "Num CD21+: CD20+", "Num CD21+: Ki67+: CD20+", "Num Ki67+: CD20+"]].sum(axis=1)

                df["Ki67&CD20/CD20"] = (df["numerator"] /
                                        (df["denominator"] + df["numerator"])) * 100
                df["Ki67&CD20/CD20"].replace([np.inf, -np.inf],
                                             np.nan, inplace=True)

                df["Classification"] = df["Classification"].astype("str")
                df["Classification"] = df["Classification"].replace(
                    "CD21+", "GC")
                df["Classification"] = df["Classification"].replace(
                    "LA", "non-GC")

                # Compute Q1 (25th percentile) and Q3 (75th percentile)
                Q1 = df["Ki67&CD20/CD20"].quantile(0.25)
                Q3 = df["Ki67&CD20/CD20"].quantile(0.75)
                IQR = Q3 - Q1

                # Define outlier boundaries
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Filter out outliers - NOT USING THIS RIGHT NOW
                df_filtered = df[(df["Ki67&CD20/CD20"] >= lower_bound)
                                 & (df["Ki67&CD20/CD20"] <= upper_bound)]

                if plot_type_2 == "Interactive (Plotly)":

                    fig = px.box(df,
                                 x="Classification",
                                 y="Ki67&CD20/CD20",
                                 title="B Cell Proliferation",
                                 points=False,  # Hide outliers
                                 color="Classification",
                                 color_discrete_sequence=["#00ff00"])  # Hex code for lime

                    jitter = px.strip(df,
                                      x="Classification",
                                      y="Ki67&CD20/CD20",
                                      color="Classification",
                                      color_discrete_sequence=["#00ff00"])  # Hex code for lime

                    # Get the traces from the jitter plot and add them to the box plot
                    for trace in jitter.data:
                        trace.marker.opacity = 0.7  # Make points semi-transparent
                        trace.marker.size = 4       # Set point size
                        fig.add_trace(trace)

                    # Update layout for better visualization
                    fig.update_layout(
                        # Adjust left and right margins
                        margin=dict(l=10, r=10),

                        boxmode='group',
                        boxgap=0.1,
                        legend_title_text='Classification',
                        xaxis_title='Classification',
                        yaxis_title='% AID+/CD20+ Cells',
                        showlegend=False  # Hide legend if duplicated
                    )

                    st.plotly_chart(fig)
                else:
                    # Create the plot
                    plot = (
                        ggplot(df, aes(x="Classification",
                                       y="Ki67&CD20/CD20", color="Classification"))
                        # Boxplot without outliers
                        + geom_boxplot(outlier_shape=None,
                                       fill="#00ff00", alpha=0.5)
                        # Jittered points
                        + geom_jitter(size=2, width=0.2,
                                      alpha=0.7, color="#00ff00")
                        + labs(title="B Cell Hypermutation",
                               x="Classification", y="% AID+/CD20+ Cells")
                        + theme(legend_position="none")  # Hide legend
                    )

                    # Show plot
                    fig = plot.draw()
                    st.pyplot(fig)

                # Drop NaN values to avoid issues in the t-test
                group1 = df[df["Classification"] ==
                            "GC"]["Ki67&CD20/CD20"].dropna()
                group2 = df[df["Classification"] ==
                            "non-GC"]["Ki67&CD20/CD20"].dropna()

                # Two-tailed independent t-test
                t_stat, p_val = ttest_ind(
                    group1, group2, equal_var=False)  # Welch's t-test

                # Display the result in Streamlit
                st.markdown(
                    f"**Two-tailed t-test**: t = {t_stat:.3f}, p = {p_val:.3e}")
                if p_val < 0.05:
                    st.markdown(
                        "**Result:** Significant difference (p < 0.05)")
                else:
                    st.markdown(
                        "**Result:** No significant difference (p ≥ 0.05)")
            
            col5, col6 = st.columns(2)

        else:
            st.error(
                "The uploaded file must contain the following columns: ROI, Num CD20+, Num Ki67+, and Classification."
            )

# About Page# About Page
if page == "About":
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
        # Replace with actual image path
        st.image("./static/images/macfawn.png",
                 caption="Dr. Ian MacFawn", width=150)
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
        st.image("./static/images/shin.png", caption="Joseph Shin",
                 width=150)  # Replace with actual image path
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
        # Replace with actual image path
        st.image("./static/images/wilkins.png",
                 caption="Luka Wilkins", width=150)
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
        st.image("./static/images/macfawn.png", caption="Sarah Zhou",
                 width=150)  # Replace with actual image path
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
        # Replace with actual image path
        st.image("./static/images/christensen.png",
                 caption="Mike Christensen", width=150)
    with col2:
        st.subheader("Mike Christensen")
        st.markdown("""
        *Undergraduate Researcher - Grove City College*  
        Mike focuses on [specific research focus or role in the lab]. 
        His contributions include [specific areas of interest or lab projects].
        """)

    st.write("---")  # Final separator
    st.markdown("More team members will be added soon!")

# Contact Us Page
elif page == "Contact Us":
    st.title("Contact Us")
    st.markdown("""
    Have questions or need assistance? Reach out to us!
    
    **Email:** shinjj22@gcc.edu  
    **Phone:** (123) 456-7890  
    **Address:** 200 Campus Drive, Grove City, PA 16127  
    """)

    # Add an image in the future
    # st.image("https://via.placeholder.com/1024x300",
    #          caption="Connect with us", use_column_width=True) update thiz to include a publications page, projects page and switch about to people
