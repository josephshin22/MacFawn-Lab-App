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
        required_columns = ["ROI", "Num CD20+", "Num Ki67+", "Classification", "Num Detections"]
        
        # Check if the required columns exist, including "Num Detections" for the threshold sliders
        if all(col in df.columns for col in required_columns):
            
            # --- METADATA SELECTION ---
            metadata_cols = [c for c in df.columns if str(c).lower().startswith("metadata")]
            meta_col_options = ["Image (Default/Patient ID)"] + metadata_cols
            
            selected_meta_col = st.selectbox(
                "Select a metadata column for grouping in plots 1 & 2 (and x-axis in plots 3 & 4)",
                meta_col_options,
                key="meta_select_all"
            )
            
            # Determine the column to group by. This will be 'Image' by default or the selected metadata column.
            group_by_col = "Image" if selected_meta_col == "Image (Default/Patient ID)" else selected_meta_col
            x_axis_title_grouping = "Patient (Image ID)" if group_by_col == "Image" else selected_meta_col
            # --------------------------

            st.write(df)
            # Replace 0 values in "Num CD20+" with NaN to avoid division by zero
            df["Num CD20+"] = df["Num CD20+"].replace(0, float("nan"))
            col1, col2 = st.columns(2)

            # TASK: Define Lymphoid Aggregate (LA), Tertiary Lymphoid Structure (TLS), and TLS + Germinal Center (GC)

            with col1:
                threshold = st.slider(
                    "Select Minimum Number of Detections", min_value=1, max_value=100, value=50, key=1)
                # Number of Lymphoid Aggregate (LA) positive annotations by patient/group
                df_la = df[df["Num Detections"] >= threshold]
                
                # Group by selected column
                count_la = df_la[df_la["Classification"] == "LA"].groupby(
                    group_by_col).size().reset_index(name="LA Count")

                # Create a bar chart
                fig = px.bar(count_la, x=group_by_col, y="LA Count",
                             title=f"Count of Lymphoid Aggregate (LA) Classifications per {x_axis_title_grouping}", 
                             labels={group_by_col: x_axis_title_grouping, "LA Count": "# of LA Classifications"})
                st.plotly_chart(fig)
                
            with col2:
                # Slider thresholds 
                threshold_la = st.slider(
                    "Select Minimum Number of Detections (LA)", min_value=1, max_value=100, value=50, key="la_slider"
                )
                threshold_gc = st.slider(
                    "Select Minimum Number of Detections (CD21+)", min_value=1, max_value=100, value=50, key="gc_slider"
                )

                # Filter rows based on threshold
                la_only_filtered = df[(df["Classification"] == "LA") & (df["Num Detections"] >= threshold_la)]
                gc_only_filtered = df[(df["Classification"] == "CD21+") & (df["Num Detections"] >= threshold_gc)]

                # Count rows per group
                count_la = la_only_filtered.groupby(group_by_col).size().reset_index(name="LA Rows")
                count_gc = gc_only_filtered.groupby(group_by_col).size().reset_index(name="CD21+ Rows")

                # Merge
                percentage_df = count_gc.merge(count_la, on=group_by_col, how="outer").fillna(0)
                
                # Filter out groups where LA Rows is 0 before calculating percentage to avoid division by zero issues
                percentage_df = percentage_df[percentage_df["LA Rows"] > 0]
                
                # Calculate percentage (row counts)
                percentage_df["Percentage GC+"] = (
                    percentage_df["CD21+ Rows"] / percentage_df["LA Rows"]
                ) * 100

                # Plot
                fig = px.bar(
                    percentage_df.sort_values("Percentage GC+"),
                    x=group_by_col,
                    y="Percentage GC+",
                    title=f"Percent Germinal Center Positive (CD21+) Relative to LA by {x_axis_title_grouping} (Row Counts)",
                    labels={group_by_col: x_axis_title_grouping, "Percentage GC+": "Percent GC+ (%)"}
                )

                st.plotly_chart(fig)

            col3, col4 = st.columns(2)
            
            # --- Common Data Prep for Col 3 and 4 ---
            
            # Recalculate Ki67&CD20/CD20 for B Cell Proliferation (Col 3)
            df["numerator_prolif"] = df[[
                "Num AID+: Ki67+: CD20+", "Num AID+: Ki67+: CD20+: CD4+", "Num CD21+: AID+: Ki67+: CD20+",
                "Num CD21+: AID+: Ki67+: CD20+: CD4+", "Num CD21+: Ki67+: CD20+",
                "Num Ki67+: CD20+", "Num Ki67+: CD20+: CD4+",
            ]].sum(axis=1)

            df["denominator_prolif"] = df[[
                "Num AID+: CD20+", "Num AID+: CD20+: CD4+", "Num CD20+",
                "Num CD20+: CD4+", "Num CD21+: AID+: CD20+", "Num CD21+: AID+: CD20+: CD4+",
                "Num CD21+: CD20+", "Num CD21+: CD20+: CD4+"]].sum(axis=1)

            df["Ki67&CD20/CD20_prolif"] = (df["numerator_prolif"] /
                                    (df["denominator_prolif"] + df["numerator_prolif"])) * 100
            df["Ki67&CD20/CD20_prolif"].replace([np.inf, -np.inf],
                                         np.nan, inplace=True)
            
            # Recalculate AID+/CD20+ for B Cell Hypermutation (Col 4)
            df["numerator_hyper"] = df[[
                "Num AID+: CD20+", "Num AID+: Ki67+: CD20+", "Num CD21+: AID+: CD20+", "Num CD21+: AID+: Ki67+: CD20+"
            ]].sum(axis=1)

            df["denominator_hyper"] = df[[
                "Num CD20+", "Num CD21+: CD20+", "Num CD21+: Ki67+: CD20+", "Num Ki67+: CD20+"]].sum(axis=1)

            df["AID+/CD20+_hyper"] = (df["numerator_hyper"] /
                                    (df["denominator_hyper"] + df["numerator_hyper"])) * 100
            df["AID+/CD20+_hyper"].replace([np.inf, -np.inf],
                                         np.nan, inplace=True)

            # Re-map Classification column for plotting
            df["Classification"] = df["Classification"].astype("str")
            df["Classification"] = df["Classification"].replace("CD21+", "GC")
            df["Classification"] = df["Classification"].replace("LA", "non-GC")

            # --- Column 3: B Cell Proliferation ---
            with col3:
                plot_type = st.radio("Select Plot Type (Proliferation)", [
                                     "Interactive (Plotly)", "Static (ggplot)"], key="plot_toggle_3")
                
                df_plot = df.dropna(subset=["Ki67&CD20/CD20_prolif"])
                
                # Check if we should use Classification on the x-axis or the grouping column
                x_col_3 = "Classification" if selected_meta_col == "Image (Default/Patient ID)" else group_by_col
                color_col_3 = "Classification" if selected_meta_col != "Image (Default/Patient ID)" else None # Use Classification for color if grouping by metadata

                if plot_type == "Interactive (Plotly)":
                    
                    fig = px.box(df_plot,
                                 x=x_col_3,
                                 y="Ki67&CD20/CD20_prolif",
                                 title="B Cell Proliferation",
                                 points=False,
                                 color=color_col_3, # Color by Classification if grouping by metadata
                                 color_discrete_sequence=["#ff8000"]) # Hex code for orange (default if no color_col)

                    jitter = px.strip(df_plot,
                                      x=x_col_3,
                                      y="Ki67&CD20/CD20_prolif",
                                      color=color_col_3,
                                      color_discrete_sequence=["#ff8000"])

                    # Get the traces from the jitter plot and add them to the box plot
                    for trace in jitter.data:
                        trace.marker.opacity = 0.7
                        trace.marker.size = 4
                        fig.add_trace(trace)

                    # Update layout for better visualization
                    x_axis_title = "Classification" if x_col_3 == "Classification" else x_axis_title_grouping
                    
                    fig.update_layout(
                        margin=dict(l=10, r=10),
                        boxmode='group',
                        boxgap=0.1,
                        legend_title_text='Classification' if color_col_3 else None,
                        xaxis_title=x_axis_title,
                        yaxis_title='% Ki67+/CD20+ Cells',
                        showlegend=bool(color_col_3) # Show legend if coloring by classification
                    )

                    st.plotly_chart(fig)

                else: # Static (ggplot)
                    x_axis_title = "Classification" if x_col_3 == "Classification" else x_axis_title_grouping
                    fill_aes = aes(fill=color_col_3) if color_col_3 else aes(fill="#ff8000")
                    
                    # Create the plot
                    plot = (
                        ggplot(df_plot, aes(x=x_col_3,
                                       y="Ki67&CD20/CD20_prolif")) + fill_aes
                        # Boxplot without outliers
                        + geom_boxplot(outlier_shape=None,
                                       alpha=0.5, 
                                       position='dodge' if color_col_3 else 'identity')
                        # Jittered points
                        + geom_jitter(size=2, width=0.2,
                                      alpha=0.7, 
                                      position='dodge' if color_col_3 else 'identity')
                        + labs(title="B Cell Proliferation",
                               x=x_axis_title, y="% Ki67+/CD20+ Cells")
                        + theme(legend_position="right" if color_col_3 else "none")
                    )

                    # Show plot
                    fig = plot.draw()
                    st.pyplot(fig)

                # T-test
                if selected_meta_col == "Image (Default/Patient ID)":
                    group_gc = df[df["Classification"] ==
                                  "GC"]["Ki67&CD20/CD20_prolif"].dropna()
                    group_non_gc = df[df["Classification"] ==
                                      "non-GC"]["Ki67&CD20/CD20_prolif"].dropna()

                    t_stat, p_value = ttest_ind(
                        group_gc, group_non_gc, equal_var=False)

                    st.markdown(
                        f"**Two-tailed t-test (GC vs. non-GC overall)**: t = {t_stat:.3f}, p = {p_value:.3e}")
                    if p_value < 0.05:
                        st.markdown(
                            "**Result:** Significant difference (p < 0.05)")
                    else:
                        st.markdown(
                            "**Result:** No significant difference (p ≥ 0.05)")
                else:
                    st.info(f"T-test is only calculated for the overall GC vs. non-GC difference. Select 'Image (Default/Patient ID)' to run the t-test.")
            
            # --- Column 4: B Cell Hypermutation ---
            with col4:
                plot_type_2 = st.radio(
                    "Select Plot Type (Hypermutation)",
                    ["Interactive (Plotly)", "Static (ggplot)"],
                    key="plot_toggle_2"
                )
                
                df_plot_2 = df.dropna(subset=["AID+/CD20+_hyper"])

                # Check if we should use Classification on the x-axis or the grouping column
                x_col_4 = "Classification" if selected_meta_col == "Image (Default/Patient ID)" else group_by_col
                color_col_4 = "Classification" if selected_meta_col != "Image (Default/Patient ID)" else None
                
                if plot_type_2 == "Interactive (Plotly)":

                    fig = px.box(df_plot_2,
                                 x=x_col_4,
                                 y="AID+/CD20+_hyper",
                                 title="B Cell Hypermutation",
                                 points=False,
                                 color=color_col_4,
                                 color_discrete_sequence=["#00ff00"]) # Hex code for lime (default if no color_col)

                    jitter = px.strip(df_plot_2,
                                      x=x_col_4,
                                      y="AID+/CD20+_hyper",
                                      color=color_col_4,
                                      color_discrete_sequence=["#00ff00"])

                    # Get the traces from the jitter plot and add them to the box plot
                    for trace in jitter.data:
                        trace.marker.opacity = 0.7
                        trace.marker.size = 4
                        fig.add_trace(trace)

                    # Update layout for better visualization
                    x_axis_title = "Classification" if x_col_4 == "Classification" else x_axis_title_grouping
                    
                    fig.update_layout(
                        margin=dict(l=10, r=10),
                        boxmode='group',
                        boxgap=0.1,
                        legend_title_text='Classification' if color_col_4 else None,
                        xaxis_title=x_axis_title,
                        yaxis_title='% AID+/CD20+ Cells',
                        showlegend=bool(color_col_4)
                    )

                    st.plotly_chart(fig)
                else: # Static (ggplot)
                    x_axis_title = "Classification" if x_col_4 == "Classification" else x_axis_title_grouping
                    fill_aes = aes(fill=color_col_4) if color_col_4 else aes(fill="#00ff00")
                    
                    # Create the plot
                    plot = (
                        ggplot(df_plot_2, aes(x=x_col_4,
                                       y="AID+/CD20+_hyper")) + fill_aes
                        # Boxplot without outliers
                        + geom_boxplot(outlier_shape=None,
                                       alpha=0.5, 
                                       position='dodge' if color_col_4 else 'identity')
                        # Jittered points
                        + geom_jitter(size=2, width=0.2,
                                      alpha=0.7, 
                                      position='dodge' if color_col_4 else 'identity')
                        + labs(title="B Cell Hypermutation",
                               x=x_axis_title, y="% AID+/CD20+ Cells")
                        + theme(legend_position="right" if color_col_4 else "none")
                    )

                    # Show plot
                    fig = plot.draw()
                    st.pyplot(fig)

                # T-test
                if selected_meta_col == "Image (Default/Patient ID)":
                    group1 = df[df["Classification"] ==
                                "GC"]["AID+/CD20+_hyper"].dropna()
                    group2 = df[df["Classification"] ==
                                "non-GC"]["AID+/CD20+_hyper"].dropna()

                    t_stat, p_val = ttest_ind(
                        group1, group2, equal_var=False)

                    st.markdown(
                        f"**Two-tailed t-test (GC vs. non-GC overall)**: t = {t_stat:.3f}, p = {p_val:.3e}")
                    if p_val < 0.05:
                        st.markdown(
                            "**Result:** Significant difference (p < 0.05)")
                    else:
                        st.markdown(
                            "**Result:** No significant difference (p ≥ 0.05)")
                else:
                    st.info(f"T-test is only calculated for the overall GC vs. non-GC difference. Select 'Image (Default/Patient ID)' to run the t-test.")
                    
        else:
            # Added "Num Detections" to the error message as it's required for the filtering logic
            required_cols_str = ", ".join(required_columns)
            st.error(
                f"The uploaded file must contain the following columns: {required_cols_str}."
            )