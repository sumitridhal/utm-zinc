#!/usr/bin/env python3
"""
Interactive Tensile Test Data Analyzer UI
Web-based interface using Streamlit for analyzing zinc tensile test results
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

# Configure page
st.set_page_config(
    page_title="Tensile Test Data Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        border-radius: 10px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .sample-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class TensileDataAnalyzerUI:
    def __init__(self):
        self.result_dir = Path("data/result")
        self.main_files_dir = self.result_dir / "main_files"
        self.individual_analyses_dir = self.result_dir / "individual_analyses"
        self.csv_files_dir = self.result_dir / "csv_files"
        self.individual_plots_dir = self.result_dir / "individual_plots"
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load all analysis data"""
        try:
            # Load main dataset
            if (self.main_files_dir / "tensile_test_data.csv").exists():
                self.main_data = pd.read_csv(self.main_files_dir / "tensile_test_data.csv")
            else:
                self.main_data = pd.DataFrame()
            
            # Load summary statistics
            if (self.main_files_dir / "summary_statistics.json").exists():
                with open(self.main_files_dir / "summary_statistics.json", 'r') as f:
                    self.summary_stats = json.load(f)
            else:
                self.summary_stats = {}
            
            # Get available experiments and samples
            if not self.main_data.empty:
                self.experiments = sorted(self.main_data['folder'].dropna().astype(str).unique())
            else:
                self.experiments = []
            
            # Get available individual CSV files
            self.available_stress_strain = []
            if self.csv_files_dir.exists():
                for csv_file in self.csv_files_dir.glob("stress_strain_*.csv"):
                    self.available_stress_strain.append(csv_file.stem.replace("stress_strain_", ""))
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            self.main_data = pd.DataFrame()
            self.summary_stats = {}
            self.experiments = []
            self.available_stress_strain = []
    
    def display_header(self):
        """Display main header"""
        st.markdown("""
        <div class="main-header">
            🔬 Tensile Test Data Analyzer
            <br><small>Interactive Analysis of Zinc Sample Test Results</small>
        </div>
        """, unsafe_allow_html=True)
    
    def display_overview(self):
        """Display overview dashboard"""
        st.header("📊 Data Overview")
        
        if self.summary_stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Samples",
                    value=self.summary_stats.get('total_samples', 0)
                )
            
            with col2:
                st.metric(
                    label="Experiments",
                    value=self.summary_stats.get('folders_processed', 0)
                )
            
            with col3:
                st.metric(
                    label="With Current",
                    value=self.summary_stats.get('samples_with_current', 0)
                )
            
            with col4:
                st.metric(
                    label="Stress-Strain Data",
                    value=self.summary_stats.get('samples_with_stress_strain_data', 0)
                )
            
            # Key properties summary
            st.subheader("🔍 Key Properties Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                if '0.2% Offset yield stress_mean' in self.summary_stats:
                    st.write("**Yield Stress (MPa)**")
                    st.write(f"Mean: {self.summary_stats['0.2% Offset yield stress_mean']:.1f}")
                    st.write(f"Std: {self.summary_stats['0.2% Offset yield stress_std']:.1f}")
                    st.write(f"Range: {self.summary_stats['0.2% Offset yield stress_min']:.1f} - {self.summary_stats['0.2% Offset yield stress_max']:.1f}")
                
                if 'Rate_mean' in self.summary_stats:
                    st.write("**Test Rate (mm/s)**")
                    st.write(f"Mean: {self.summary_stats['Rate_mean']:.3f}")
                    st.write(f"Range: {self.summary_stats['Rate_min']:.3f} - {self.summary_stats['Rate_max']:.1f}")
            
            with col2:
                if 'Elogation at break (using Strain)_mean' in self.summary_stats:
                    st.write("**Elongation at Break (%)**")
                    st.write(f"Mean: {self.summary_stats['Elogation at break (using Strain)_mean']:.1f}")
                    st.write(f"Std: {self.summary_stats['Elogation at break (using Strain)_std']:.1f}")
                
                if 'amplitude_mean' in self.summary_stats:
                    st.write("**Pulse Amplitude (A)**")
                    st.write(f"Mean: {self.summary_stats['amplitude_mean']:.1f}")
                    st.write(f"Range: {self.summary_stats['amplitude_min']:.0f} - {self.summary_stats['amplitude_max']:.0f}")
    
    def display_experiment_selector(self):
        """Display experiment and sample selector"""
        st.header("🧪 Experiment & Sample Selection")
        
        if not self.experiments:
            st.warning("No experimental data found. Please run the analysis first.")
            return None, None
        
        # Experiment selection - Multi-select
        st.subheader("Select Experiment Folders")
        selected_experiments = st.multiselect(
            "Choose Experiment Folders:",
            self.experiments,
            default=self.experiments[:3] if len(self.experiments) > 3 else self.experiments,
            help="Choose specific experiments to analyze (multiple selection allowed)"
        )
        
        # Add "Select All" option
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Select All Experiments"):
                selected_experiments = self.experiments
        with col2:
            if st.button("Clear Selection"):
                selected_experiments = []
        
        # Filter data based on experiment selection
        if selected_experiments:
            filtered_data = self.main_data[self.main_data['folder'].astype(str).isin(selected_experiments)]
        else:
            filtered_data = pd.DataFrame()  # Empty dataframe if no experiments selected
            st.warning("Please select at least one experiment folder.")
        
        # Sample selection - Multi-select
        if not filtered_data.empty:
            st.subheader("Select Samples")
            
            # Get unique samples from filtered data
            unique_samples = filtered_data['sample_name'].dropna().astype(str).unique()
            sorted_samples = sorted(unique_samples)
            
            selected_samples = st.multiselect(
                "Choose Samples:",
                sorted_samples,
                default=sorted_samples[:5] if len(sorted_samples) > 5 else sorted_samples,
                help="Choose specific samples to analyze (multiple selection allowed)"
            )
            
            # Add "Select All" option for samples
            sample_col1, sample_col2 = st.columns([1, 4])
            with sample_col1:
                if st.button("Select All Samples"):
                    selected_samples = sorted_samples
            with sample_col2:
                if st.button("Clear Sample Selection"):
                    selected_samples = []
            
            # Filter by selected samples
            if selected_samples:
                filtered_data = filtered_data[filtered_data['sample_name'].astype(str).isin(selected_samples)]
            else:
                filtered_data = pd.DataFrame()  # Empty dataframe if no samples selected
                st.warning("Please select at least one sample.")
        else:
            selected_samples = []
        
        # Display selection summary
        if not filtered_data.empty:
            st.subheader("📋 Selection Summary")
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.metric(
                    label="Selected Experiments",
                    value=len(selected_experiments)
                )
                with st.expander("View Selected Experiments"):
                    for exp in selected_experiments:
                        st.write(f"• {exp}")
            
            with summary_col2:
                st.metric(
                    label="Selected Samples",
                    value=len(selected_samples)
                )
                with st.expander("View Selected Samples"):
                    for sample in selected_samples:
                        st.write(f"• {sample}")
            
            with summary_col3:
                st.metric(
                    label="Total Data Points",
                    value=len(filtered_data)
                )
                # Show breakdown by experiment
                with st.expander("Data Points by Experiment"):
                    for exp in selected_experiments:
                        count = len(filtered_data[filtered_data['folder'].astype(str) == exp])
                        st.write(f"• {exp}: {count} samples")
        
        return selected_experiments, filtered_data
    
    def display_sample_comparison(self, data):
        """Display sample comparison and analysis"""
        if data.empty:
            st.warning("No data available for the selected criteria.")
            return
        
        st.header("📈 Sample Analysis & Comparison")
        
        # Sample summary table
        st.subheader("Sample Summary")
        
        # Select columns to display
        display_columns = [
            'folder', 'sample_name', 'filename', 'has_current', 
            'pulse_on_time', 'amplitude', 'strain_rate',
            '0.2% Offset yield stress', 'Elogation at break (using Strain)',
            'Rate', 'has_stress_strain_data'
        ]
        
        # Filter available columns
        available_columns = [col for col in display_columns if col in data.columns]
        display_data = data[available_columns]
        
        # Format the data for better display
        if not display_data.empty:
            formatted_data = display_data.copy()
            
            # Round numerical columns
            numerical_cols = ['pulse_on_time', 'amplitude', 'strain_rate', 
                            '0.2% Offset yield stress', 'Elogation at break (using Strain)', 'Rate']
            for col in numerical_cols:
                if col in formatted_data.columns:
                    formatted_data[col] = pd.to_numeric(formatted_data[col], errors='coerce').round(3)
            
            st.dataframe(formatted_data, use_container_width=True)
        
        # Statistical comparison
        if len(data) > 1:
            st.subheader("Statistical Comparison")
            
            # Select property for comparison
            numerical_properties = [
                '0.2% Offset yield stress',
                'Elogation at break (using Strain)',
                'Strain handening exponent',
                'Rate',
                'pulse_on_time',
                'amplitude'
            ]
            
            available_props = [prop for prop in numerical_properties if prop in data.columns]
            
            if available_props:
                selected_property = st.selectbox(
                    "Select Property for Comparison:",
                    available_props
                )
                
                # Create comparison plots
                col1, col2 = st.columns(2)
                
                with col1:
                    # Box plot by experiment
                    if 'folder' in data.columns:
                        fig = px.box(
                            data, 
                            x='folder', 
                            y=selected_property,
                            title=f"{selected_property} by Experiment",
                            color='folder'
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Histogram
                    clean_data = pd.to_numeric(data[selected_property], errors='coerce').dropna()
                    if not clean_data.empty:
                        fig = px.histogram(
                            clean_data,
                            title=f"Distribution of {selected_property}",
                            nbins=20
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    def display_stress_strain_analysis(self, data):
        """Display stress-strain curve analysis"""
        st.header("📊 Stress-Strain Curve Analysis")
        
        if data.empty:
            st.warning("No data available.")
            return
        
        # Get samples with stress-strain data
        samples_with_data = data[data['has_stress_strain_data'] == True]
        
        if samples_with_data.empty:
            st.warning("No stress-strain data available for selected samples.")
            return
        
        # Multi-select for curves to display
        sample_keys = []
        for _, row in samples_with_data.iterrows():
            key = f"{row['folder']}_{row['sample_name']}"
            if key in self.available_stress_strain:
                sample_keys.append(key)
        
        if not sample_keys:
            st.warning("No stress-strain CSV files found for selected samples.")
            return
        
        # Add tabs for different analysis types
        tab1, tab2 = st.tabs(["📊 Multi-Curve Comparison", "🔍 Two-Dataset Comparison with Cropping"])
        
        with tab1:
            st.subheader("Multi-Curve Comparison")
            selected_curves = st.multiselect(
                "Select Stress-Strain Curves to Display:",
                sample_keys,
                default=sample_keys[:5] if len(sample_keys) > 5 else sample_keys,
                help="Select up to 10 curves for comparison"
            )
            
            if selected_curves:
                self._display_multi_curve_comparison(selected_curves, samples_with_data)
        
        with tab2:
            st.subheader("Two-Dataset Comparison with Cropping")
            self._display_two_dataset_comparison(sample_keys, samples_with_data)
    
    def _display_multi_curve_comparison(self, selected_curves, samples_with_data):
        """Display multi-curve comparison (original functionality)"""
        # Load and plot stress-strain curves
        fig = go.Figure()
        
        for key in selected_curves[:10]:  # Limit to 10 curves
            csv_file = self.csv_files_dir / f"stress_strain_{key}.csv"
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Find stress and strain columns
                    stress_col = None
                    strain_col = None
                    
                    for col in df.columns:
                        if 'true stress' in col.lower():
                            stress_col = col
                        elif 'true strain' in col.lower():
                            strain_col = col
                    
                    if stress_col and strain_col:
                        # Limit data points for performance
                        max_points = 500
                        step = max(1, len(df) // max_points)
                        
                        strain = df[strain_col][::step]
                        stress = df[stress_col][::step]
                        
                        fig.add_trace(go.Scatter(
                            x=strain,
                            y=stress,
                            mode='lines',
                            name=key,
                            line=dict(width=2)
                        ))
                
                except Exception as e:
                    st.warning(f"Error loading {key}: {e}")
        
        if fig.data:
            fig.update_layout(
                title="True Stress-Strain Curves Comparison",
                xaxis_title="True Strain (%)",
                yaxis_title="True Stress (MPa)",
                hovermode='x unified',
                width=None,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add curve analysis
            st.subheader("Curve Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Selected Curves:**")
                for key in selected_curves:
                    sample_info = samples_with_data[
                        (samples_with_data['folder'] + '_' + samples_with_data['sample_name']) == key
                    ]
                    if not sample_info.empty:
                        row = sample_info.iloc[0]
                        current_status = "✅ With Current" if row.get('has_current', True) else "❌ No Current"
                        st.write(f"• **{key}**: {current_status}")
            
            with col2:
                st.write("**Curve Statistics:**")
                st.write(f"• Number of curves: {len(selected_curves)}")
                with_current = len([k for k in selected_curves if samples_with_data[
                    (samples_with_data['folder'] + '_' + samples_with_data['sample_name']) == k
                ]['has_current'].iloc[0] if not samples_with_data[
                    (samples_with_data['folder'] + '_' + samples_with_data['sample_name']) == k
                ].empty])
                st.write(f"• With current: {with_current}")
                st.write(f"• Without current: {len(selected_curves) - with_current}")
    
    def _display_two_dataset_comparison(self, sample_keys, samples_with_data):
        """Display two-dataset comparison with cropping functionality"""
        if len(sample_keys) < 2:
            st.warning("At least 2 datasets are required for comparison.")
            return
        
        # Dataset selection
        col1, col2 = st.columns(2)
        
        with col1:
            dataset1 = st.selectbox(
                "Select First Dataset:",
                sample_keys,
                key="dataset1",
                help="Choose the first dataset for comparison"
            )
        
        with col2:
            dataset2 = st.selectbox(
                "Select Second Dataset:",
                [key for key in sample_keys if key != dataset1],
                key="dataset2",
                help="Choose the second dataset for comparison"
            )
        
        if dataset1 and dataset2:
            # Load both datasets
            dataset1_df = self._load_stress_strain_data(dataset1)
            dataset2_df = self._load_stress_strain_data(dataset2)
            
            if dataset1_df is not None and dataset2_df is not None:
                # Find common columns
                stress_col1, strain_col1 = self._find_stress_strain_columns(dataset1_df)
                stress_col2, strain_col2 = self._find_stress_strain_columns(dataset2_df)
                
                if stress_col1 and strain_col1 and stress_col2 and strain_col2:
                    # Cropping controls
                    st.subheader("🔧 Cropping Controls")
                    
                    crop_col1, crop_col2 = st.columns(2)
                    
                    with crop_col1:
                        st.write(f"**{dataset1} Cropping:**")
                        crop_start1 = st.slider(
                            "Crop from start (%):",
                            0, 50, 0,
                            key="crop_start1",
                            help="Percentage of data to remove from the beginning"
                        )
                        crop_end1 = st.slider(
                            "Crop from end (%):",
                            0, 50, 0,
                            key="crop_end1",
                            help="Percentage of data to remove from the end"
                        )
                    
                    with crop_col2:
                        st.write(f"**{dataset2} Cropping:**")
                        crop_start2 = st.slider(
                            "Crop from start (%):",
                            0, 50, 0,
                            key="crop_start2",
                            help="Percentage of data to remove from the beginning"
                        )
                        crop_end2 = st.slider(
                            "Crop from end (%):",
                            0, 50, 0,
                            key="crop_end2",
                            help="Percentage of data to remove from the end"
                        )
                    
                    # Apply cropping
                    cropped_data1 = self._apply_cropping(dataset1_df, stress_col1, strain_col1, crop_start1, crop_end1)
                    cropped_data2 = self._apply_cropping(dataset2_df, stress_col2, strain_col2, crop_start2, crop_end2)
                    
                    # Display comparison plot
                    st.subheader("📊 Comparison Plot")
                    
                    fig = go.Figure()
                    
                    # Add first dataset
                    fig.add_trace(go.Scatter(
                        x=cropped_data1['strain'],
                        y=cropped_data1['stress'],
                        mode='lines',
                        name=f"{dataset1} (cropped)",
                        line=dict(width=3, color='#1f77b4')
                    ))
                    
                    # Add second dataset
                    fig.add_trace(go.Scatter(
                        x=cropped_data2['strain'],
                        y=cropped_data2['stress'],
                        mode='lines',
                        name=f"{dataset2} (cropped)",
                        line=dict(width=3, color='#ff7f0e')
                    ))
                    
                    fig.update_layout(
                        title=f"Stress-Strain Comparison: {dataset1} vs {dataset2}",
                        xaxis_title="True Strain (%)",
                        yaxis_title="True Stress (MPa)",
                        hovermode='x unified',
                        width=None,
                        height=600,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display statistics
                    st.subheader("📈 Comparison Statistics")
                    
                    stat_col1, stat_col2 = st.columns(2)
                    
                    with stat_col1:
                        st.write(f"**{dataset1} Statistics:**")
                        st.write(f"• Original data points: {len(dataset1_df)}")
                        st.write(f"• Cropped data points: {len(cropped_data1['strain'])}")
                        st.write(f"• Max stress: {cropped_data1['stress'].max():.2f} MPa")
                        st.write(f"• Max strain: {cropped_data1['strain'].max():.2f}%")
                        st.write(f"• Strain range: {cropped_data1['strain'].min():.3f}% to {cropped_data1['strain'].max():.2f}%")
                    
                    with stat_col2:
                        st.write(f"**{dataset2} Statistics:**")
                        st.write(f"• Original data points: {len(dataset2_df)}")
                        st.write(f"• Cropped data points: {len(cropped_data2['strain'])}")
                        st.write(f"• Max stress: {cropped_data2['stress'].max():.2f} MPa")
                        st.write(f"• Max strain: {cropped_data2['strain'].max():.2f}%")
                        st.write(f"• Strain range: {cropped_data2['strain'].min():.3f}% to {cropped_data2['strain'].max():.2f}%")
                    
                    # Export cropped data option
                    st.subheader("💾 Export Cropped Data")
                    
                    export_col1, export_col2 = st.columns(2)
                    
                    with export_col1:
                        if st.button("Download Dataset 1 (Cropped)", key="download1"):
                            csv_data1 = pd.DataFrame({
                                'True Strain (%)': cropped_data1['strain'],
                                'True Stress (MPa)': cropped_data1['stress']
                            })
                            csv_string1 = csv_data1.to_csv(index=False)
                            b64_1 = base64.b64encode(csv_string1.encode()).decode()
                            href_1 = f'<a href="data:file/csv;base64,{b64_1}" download="{dataset1}_cropped.csv">Download CSV</a>'
                            st.markdown(href_1, unsafe_allow_html=True)
                    
                    with export_col2:
                        if st.button("Download Dataset 2 (Cropped)", key="download2"):
                            csv_data2 = pd.DataFrame({
                                'True Strain (%)': cropped_data2['strain'],
                                'True Stress (MPa)': cropped_data2['stress']
                            })
                            csv_string2 = csv_data2.to_csv(index=False)
                            b64_2 = base64.b64encode(csv_string2.encode()).decode()
                            href_2 = f'<a href="data:file/csv;base64,{b64_2}" download="{dataset2}_cropped.csv">Download CSV</a>'
                            st.markdown(href_2, unsafe_allow_html=True)
                
                else:
                    st.error("Could not find stress and strain columns in the selected datasets.")
            else:
                st.error("Could not load the selected datasets.")
    
    def _load_stress_strain_data(self, dataset_key):
        """Load stress-strain data from CSV file"""
        csv_file = self.csv_files_dir / f"stress_strain_{dataset_key}.csv"
        if csv_file.exists():
            try:
                return pd.read_csv(csv_file)
            except Exception as e:
                st.error(f"Error loading {dataset_key}: {e}")
                return None
        else:
            st.error(f"CSV file not found for {dataset_key}")
            return None
    
    def _find_stress_strain_columns(self, df):
        """Find stress and strain columns in the dataframe"""
        stress_col = None
        strain_col = None
        
        for col in df.columns:
            if 'true stress' in col.lower():
                stress_col = col
            elif 'true strain' in col.lower():
                strain_col = col
        
        return stress_col, strain_col
    
    def _apply_cropping(self, df, stress_col, strain_col, crop_start_pct, crop_end_pct):
        """Apply cropping to the dataset"""
        total_points = len(df)
        
        # Calculate crop indices
        start_idx = int(total_points * crop_start_pct / 100)
        end_idx = total_points - int(total_points * crop_end_pct / 100)
        
        # Ensure valid indices
        start_idx = max(0, start_idx)
        end_idx = min(total_points, end_idx)
        end_idx = max(start_idx + 1, end_idx)  # Ensure at least one data point
        
        # Extract cropped data
        cropped_strain = df[strain_col].iloc[start_idx:end_idx]
        cropped_stress = df[stress_col].iloc[start_idx:end_idx]
        
        return {
            'strain': cropped_strain,
            'stress': cropped_stress
        }
    
    def display_individual_sample_details(self, data):
        """Display detailed analysis for individual samples"""
        st.header("🔍 Individual Sample Details")
        
        if data.empty:
            st.warning("No data available.")
            return
        
        # Sample selector
        sample_list = []
        for _, row in data.iterrows():
            folder = str(row.get('folder', 'Unknown'))
            sample_name = str(row.get('sample_name', 'Unknown'))
            filename = str(row.get('filename', 'Unknown'))
            sample_list.append(f"{folder}_{sample_name} ({filename})")
        
        if not sample_list:
            return
        
        selected_sample_display = st.selectbox(
            "Select Sample for Detailed Analysis:",
            sample_list
        )
        
        # Extract actual sample info
        sample_key = selected_sample_display.split(' (')[0]
        folder, sample_name = sample_key.split('_', 1)
        
        # Handle string comparison for sample filtering
        sample_row = data[
            (data['folder'].astype(str) == folder) & (data['sample_name'].astype(str) == sample_name)
        ].iloc[0]
        
        # Display sample details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sample Information")
            st.write(f"**Experiment:** {sample_row['folder']}")
            st.write(f"**Sample:** {sample_row['sample_name']}")
            st.write(f"**Filename:** {sample_row['filename']}")
            st.write(f"**Current Applied:** {'Yes' if sample_row.get('has_current', True) else 'No'}")
            
            st.subheader("Processing Parameters")
            st.write(f"**Pulse On Time:** {sample_row.get('pulse_on_time', 'N/A')} s")
            st.write(f"**Pulse Total Time:** {sample_row.get('pulse_total_time', 'N/A')} s")
            st.write(f"**Amplitude:** {sample_row.get('amplitude', 'N/A')} A")
            st.write(f"**Strain Rate:** {sample_row.get('strain_rate', 'N/A')} s⁻¹")
        
        with col2:
            st.subheader("Mechanical Properties")
            st.write(f"**Test Rate:** {sample_row.get('Rate', 'N/A')} mm/s")
            st.write(f"**0.2% Offset Yield Stress:** {sample_row.get('0.2% Offset yield stress', 'N/A')} MPa")
            st.write(f"**Strain Hardening Exponent:** {sample_row.get('Strain handening exponent', 'N/A')}")
            st.write(f"**Strain Hardening Coefficient:** {sample_row.get('Strain handening coefficient', 'N/A')}")
            st.write(f"**Elongation at Break:** {sample_row.get('Elogation at break (using Strain)', 'N/A')}%")
            
            st.subheader("Data Availability")
            st.write(f"**Stress-Strain Data:** {'✅ Available' if sample_row.get('has_stress_strain_data', False) else '❌ Not Available'}")
        
        # Display individual plot if available
        plot_file = self.individual_plots_dir / f"plot_{sample_key}.png"
        if plot_file.exists():
            st.subheader("Stress-Strain Curve")
            st.image(str(plot_file), caption=f"Stress-Strain Curve for {sample_key}")
        
        # Display individual analysis if available
        analysis_file = self.individual_analyses_dir / f"analysis_{sample_key}.md"
        if analysis_file.exists():
            st.subheader("Detailed Analysis")
            with open(analysis_file, 'r') as f:
                analysis_content = f.read()
            
            # Extract AI analysis section if it exists
            if "## AI Analysis" in analysis_content:
                ai_section = analysis_content.split("## AI Analysis")[1].split("## Data Quality Assessment")[0]
                if ai_section.strip() and "AI analysis skipped" not in ai_section:
                    st.markdown("**AI Analysis:**")
                    st.markdown(ai_section.strip())
    
    def run(self):
        """Run the Streamlit app"""
        self.display_header()
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Select Analysis Page:",
            ["Overview", "Experiment Selection", "Sample Comparison", "Stress-Strain Analysis", "Individual Sample Details"]
        )
        
        # Main content based on page selection
        if page == "Overview":
            self.display_overview()
        
        elif page == "Experiment Selection":
            selected_experiments, filtered_data = self.display_experiment_selector()
            
            if filtered_data is not None and not filtered_data.empty:
                st.success(f"Found {len(filtered_data)} samples for analysis")
                
                # Quick stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Samples with Current", len(filtered_data[filtered_data['has_current'] == True]))
                with col2:
                    st.metric("Samples without Current", len(filtered_data[filtered_data['has_current'] == False]))
                with col3:
                    st.metric("With Stress-Strain Data", len(filtered_data[filtered_data['has_stress_strain_data'] == True]))
        
        elif page == "Sample Comparison":
            selected_experiments, filtered_data = self.display_experiment_selector()
            if filtered_data is not None:
                self.display_sample_comparison(filtered_data)
        
        elif page == "Stress-Strain Analysis":
            selected_experiments, filtered_data = self.display_experiment_selector()
            if filtered_data is not None:
                self.display_stress_strain_analysis(filtered_data)
        
        elif page == "Individual Sample Details":
            selected_experiments, filtered_data = self.display_experiment_selector()
            if filtered_data is not None:
                self.display_individual_sample_details(filtered_data)
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Tensile Test Data Analyzer**")
        st.sidebar.markdown("Interactive analysis of zinc sample tensile test results")

def main():
    """Main function to run the Streamlit app"""
    analyzer = TensileDataAnalyzerUI()
    analyzer.run()

if __name__ == "__main__":
    main() 