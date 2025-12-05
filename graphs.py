import time
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from plotnine import ggplot, aes, geom_boxplot
import plotly.graph_objects as go
from plotnine import ggplot, aes, geom_boxplot, geom_jitter, labs, theme_minimal, theme
from scipy.stats import ttest_ind


def show():
    """Render the Graphs / analysis page where users can upload a CSV and create plots."""
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
                # Slider thresholds (if you still want to filter detections, keep this;
                # otherwise, remove threshold logic entirely)
                threshold_la = st.slider(
                    "Select Minimum Number of Detections (LA)", min_value=1, max_value=100, value=50, key="la_slider"
                )
                threshold_gc = st.slider(
                    "Select Minimum Number of Detections (CD21+)", min_value=1, max_value=100, value=50, key="gc_slider"
                )

                # Filter rows based on threshold
                la_only_filtered = df[(df["Classification"] == "LA") & (df["Num Detections"] >= threshold_la)]
                gc_only_filtered = df[(df["Classification"] == "CD21+") & (df["Num Detections"] >= threshold_gc)]

                # Count **rows per patient**, not detections
                count_la_per_patient = la_only_filtered.groupby("Image").size().reset_index(name="LA Rows")
                count_gc_per_patient = gc_only_filtered.groupby("Image").size().reset_index(name="CD21+ Rows")

                # Merge
                percentage_df = count_gc_per_patient.merge(count_la_per_patient, on="Image", how="outer").fillna(0)

                # Calculate percentage (row counts)
                percentage_df["Percentage GC+"] = (
                    percentage_df["CD21+ Rows"] / percentage_df["LA Rows"]
                ) * 100

                # Plot
                fig = px.bar(
                    percentage_df.sort_values("Percentage GC+"),
                    x="Image",
                    y="Percentage GC+",
                    title="Percent Germinal Center Positive (CD21+) Relative to LA (Row Counts)",
                    labels={"Image": "Patient (Image ID)", "Percentage GC+": "Percent GC+ (%)"}
                )

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
            # metadata_cols = [c for c in df.columns if str(c).lower().startswith("metadata")]
            # if metadata_cols:
            #     # select which metadata column to group by
            #     meta_col = st.selectbox("Select metadata column to group by", metadata_cols, key="meta_select")
            #     with col5:
            #         if "Num Detections" not in df.columns:
            #             st.warning("Column 'Num Detections' not found — metadata grouping requires it.")
            #         else:
            #             meta_threshold = st.slider(
            #                 "Select Minimum Number of Detections (metadata)",
            #                 min_value=1, max_value=100, value=50, key="meta_thr_1"
            #             )

            #             # filter rows
            #             df_meta = df[df["Num Detections"] >= meta_threshold]

            #             # group by metadata value + classification
            #             counts_meta = df_meta.groupby(meta_col).size().reset_index(name="Count")

            #             # plot grouped bars (one for each classification under each metadata value)
            #             fig_meta = px.bar(
            #                 counts_meta,
            #                 x=meta_col,
            #                 y="Count",
            #                 barmode="group",
            #                 title=f"Counts of Classifications by {meta_col}",
            #                 labels={meta_col: meta_col, "Count": "# of Detections"}
            #             )
            #             st.plotly_chart(fig_meta)


            #     with col6:
            #         if "Num Detections" not in df.columns:
            #             st.warning("Column 'Num Detections' not found — metadata grouping requires it.")
            #         else:
            #             threshold_la_meta = st.slider(
            #                 "Select Minimum Number of Detections (LA) by metadata",
            #                 min_value=1, max_value=100, value=50, key="meta_thr_2"
            #             )
            #             threshold_gc_meta = st.slider(
            #                 "Select Minimum Number of Detections (CD21+) by metadata",
            #                 min_value=1, max_value=100, value=50, key="meta_thr_3"
            #             )

            #             la_only_meta = df[(df["Classification"] == "LA") & (df["Num Detections"] > threshold_la_meta)]
            #             gc_only_meta = df[(df["Classification"] == "CD21+") & (df["Num Detections"] > threshold_gc_meta)]

            #             count_la_per_meta = la_only_meta.groupby(meta_col).size().reset_index(name="LA Count")
            #             count_gc_per_meta = gc_only_meta.groupby(meta_col).size().reset_index(name="CD21+ Count")

            #             percentage_meta = count_gc_per_meta.merge(count_la_per_meta, on=meta_col, how="inner")
            #             if not percentage_meta.empty:
            #                 percentage_meta["Percentage GC+"] = (percentage_meta["CD21+ Count"] / percentage_meta["LA Count"]) * 100
            #                 fig_meta_pct = px.bar(
            #                     percentage_meta,
            #                     x=meta_col,
            #                     y="Percentage GC+",
            #                     title=f"Percentage of GC+ (CD21+) Classifications Relative to LA by {meta_col}",
            #                     labels={meta_col: meta_col, "Percentage GC+": "Percentage of GC+ Classifications (%)"}
            #                 )
            #                 st.plotly_chart(fig_meta_pct)
            #             else:
            #                 st.info("No overlapping metadata groups found with the selected thresholds.")
            # else:
            #     with col5:
            #         st.info("No columns starting with 'metadata' were found. Add columns whose names start with 'metadata' to enable grouping.")
            #     with col6:
            #         st.empty()
        else:
            st.error(
                "The uploaded file must contain the following columns: ROI, Num CD20+, Num Ki67+, and Classification."
            )