#!/usr/bin/env python3
"""
Tensile Test Data Analysis Script
Analyzes tensile test data from zinc sample Excel files
"""

import pandas as pd
import numpy as np
import os
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import requests
import ollama
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Ollama configuration
OLLAMA_MODEL = "mistral-nemo"
OLLAMA_HOST = "http://localhost:11434"

class TensileDataAnalyzer:
    def __init__(self, data_dir="data/zinc", result_dir="data/result"):
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        
        # Create organized folder structure
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.individual_analyses_dir = self.result_dir / "individual_analyses"
        self.individual_plots_dir = self.result_dir / "individual_plots"
        self.csv_files_dir = self.result_dir / "csv_files"
        self.main_files_dir = self.result_dir / "main_files"
        
        # Create subdirectories
        for dir_path in [self.individual_analyses_dir, self.individual_plots_dir, 
                        self.csv_files_dir, self.main_files_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Store all processed data
        self.all_data = []
        self.stress_strain_data = {}
        
        # Initialize Ollama client
        self.ollama_client = self.connect_ollama()
    
    def connect_ollama(self):
        """
        Connects to the local Ollama instance and verifies the model.
        """
        print(f"Connecting to Ollama at {OLLAMA_HOST}...")
        try:
            client = ollama.Client(host=OLLAMA_HOST)
            client.show(OLLAMA_MODEL)
            print(f"Successfully connected to Ollama and found model '{OLLAMA_MODEL}'.")
            return client
        except Exception as e:
            print(f"Failed to connect to Ollama or find model '{OLLAMA_MODEL}': {e}")
            return None
        
    def parse_filename(self, filename):
        """
        Parse experimental parameters from filename
        e.g., ep1_.3_5_100A_Tensile_ -> sample=ep1, pulse_on=0.3, pulse_total=5, amplitude=100A
        """
        # Remove file extension and Tensile part
        base_name = filename.replace('.xls', '').replace('_Tensile', '').split('_Tensile')[0]
        
        params = {
            'filename': filename,
            'sample_name': '',
            'pulse_on_time': None,
            'pulse_total_time': None,
            'amplitude': None,
            'strain_rate': None,
            'has_current': True,
            'temperature': None
        }
        
        # Extract sample name (ep1, ep2, etc. or zp1, zp2, etc.)
        sample_match = re.search(r'^(ep\d+|zp\d+|\d+)', base_name)
        if sample_match:
            params['sample_name'] = sample_match.group(1)
        
        # Check if it's a no current experiment
        if 'no current' in base_name.lower() or 'no curremt' in base_name.lower():
            params['has_current'] = False
            return params
        
        # Extract strain rate if present
        sr_match = re.search(r'sr([\d.]+)', base_name)
        if sr_match:
            params['strain_rate'] = float(sr_match.group(1))
        
        # Extract pulse parameters (pulse_on_time, pulse_total_time, amplitude)
        # Pattern: _number_number_numberA
        pulse_pattern = r'_(\d*\.?\d+)_(\d*\.?\d+)_(\d+)A'
        pulse_match = re.search(pulse_pattern, base_name)
        if pulse_match:
            params['pulse_on_time'] = float(pulse_match.group(1))
            params['pulse_total_time'] = float(pulse_match.group(2))
            params['amplitude'] = int(pulse_match.group(3))
        
        # Extract temperature if present
        temp_match = re.search(r'(\d+)_', base_name)
        if temp_match and 'A' not in temp_match.group(0):
            params['temperature'] = int(temp_match.group(1))
        
        return params
    
    def read_excel_data(self, file_path):
        """
        Read data from Excel file - both TrueStress-TrueStrain data and Report sheets
        """
        try:
            # Read TrueStress-TrueStrain data
            stress_strain_df = None
            try:
                stress_strain_df = pd.read_excel(file_path, sheet_name='TrueStress-TrueStrain data')
            except:
                print(f"Could not read 'TrueStress-TrueStrain data' sheet from {file_path}")
            
            # Read Report data
            report_data = {}
            try:
                report_df = pd.read_excel(file_path, sheet_name='Report', header=None)
                
                # Extract key properties from Report sheet
                properties_to_extract = {
                    'Rate': ['rate', 'speed', 'velocity'],
                    '0.2% Offset yield stress': ['yield', '0.2%', 'offset', 'proof'],
                    'Strain handening exponent': ['strain hardening', 'n-value', 'hardening exponent', 'strain exponent'],
                    'Strain handening coefficient': ['strain hardening coefficient', 'k-value', 'hardening coefficient'],
                    'Elogation at break (using Strain)': ['elongation', 'break', 'failure', 'fracture', 'strain at break']
                }
                
                # Convert all cells to string for searching
                search_data = report_df.astype(str).fillna('')
                
                for prop, keywords in properties_to_extract.items():
                    found = False
                    for idx, row in search_data.iterrows():
                        for col_idx, cell in enumerate(row):
                            cell_lower = cell.lower()
                            # Check if any keyword matches
                            if any(keyword in cell_lower for keyword in keywords):
                                # Look for value in adjacent columns
                                for next_col in range(col_idx + 1, min(col_idx + 4, len(row))):  # Check next 3 columns
                                    try:
                                        value = report_df.iloc[idx, next_col]
                                        if pd.notna(value) and str(value).strip() != '':
                                            # Try to convert to number if possible
                                            try:
                                                report_data[prop] = float(value)
                                            except:
                                                report_data[prop] = value
                                            found = True
                                            break
                                    except:
                                        continue
                                if found:
                                    break
                        if found:
                            break
                
            except Exception as e:
                print(f"Could not read 'Report' sheet from {file_path}: {e}")
            
            return stress_strain_df, report_data
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None, {}
    
    def process_all_files(self):
        """
        Process all Excel files in the zinc data folders
        """
        folders = ['Zn1', 'Zn2', 'Zn3', 'Zn4']
        
        for folder in folders:
            folder_path = self.data_dir / folder
            if not folder_path.exists():
                continue
                
            print(f"Processing folder: {folder}")
            
            # Get all .xls files
            xls_files = list(folder_path.glob('*.xls'))
            
            for file_path in xls_files:
                print(f"  Processing: {file_path.name}")
                
                # Parse filename to get experimental parameters
                params = self.parse_filename(file_path.name)
                params['folder'] = folder
                
                # Read Excel data
                stress_strain_df, report_data = self.read_excel_data(file_path)
                
                # Combine all data
                combined_data = {**params, **report_data}
                
                # Store stress-strain data separately if available
                if stress_strain_df is not None:
                    key = f"{folder}_{params['sample_name']}"
                    self.stress_strain_data[key] = stress_strain_df
                    combined_data['has_stress_strain_data'] = True
                    combined_data['stress_strain_key'] = key  # Store the key for later use
                else:
                    combined_data['has_stress_strain_data'] = False
                    combined_data['stress_strain_key'] = None
                
                self.all_data.append(combined_data)
    
    def analyze_with_ollama(self, data_summary):
        """
        Use Ollama mistral-nemo model for analysis
        """
        try:
            # Prepare prompt for analysis
            prompt = f"""
            Please analyze the following tensile test data from zinc samples:
            
            {data_summary}
            
            Please provide insights on:
            1. The effect of pulse parameters (on time, total time, amplitude) on mechanical properties
            2. Comparison between samples with and without current
            3. Strain rate effects on the material properties
            4. Trends in yield stress, strain hardening behavior, and elongation
            5. Recommendations for optimal processing parameters
            
            Provide a detailed technical analysis suitable for materials science research.
            """
            
            # Call Ollama API using client
            if self.ollama_client:
                response = self.ollama_client.generate(
                    model=OLLAMA_MODEL,
                    prompt=prompt
                )
                return response['response']
            else:
                return "Could not connect to Ollama model for analysis."
                
        except Exception as e:
            return f"Error in Ollama analysis: {e}"
    
    def create_visualizations(self):
        """
        Create visualizations of the data
        """
        df = pd.DataFrame(self.all_data)
        
        # Set up the plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                pass  # Use default style
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Tensile Test Data Analysis - Zinc Samples', fontsize=16)
        
        # 1. Yield Stress vs Amplitude
        if '0.2% Offset yield stress' in df.columns:
            df_with_amplitude = df[df['amplitude'].notna()]
            if not df_with_amplitude.empty:
                axes[0,0].scatter(df_with_amplitude['amplitude'], df_with_amplitude['0.2% Offset yield stress'])
                axes[0,0].set_xlabel('Amplitude (A)')
                axes[0,0].set_ylabel('0.2% Offset Yield Stress')
                axes[0,0].set_title('Yield Stress vs Amplitude')
        
        # 2. Strain Rate Effects
        if 'strain_rate' in df.columns and '0.2% Offset yield stress' in df.columns:
            df_with_sr = df[df['strain_rate'].notna()]
            if not df_with_sr.empty:
                axes[0,1].scatter(df_with_sr['strain_rate'], df_with_sr['0.2% Offset yield stress'])
                axes[0,1].set_xlabel('Strain Rate')
                axes[0,1].set_ylabel('0.2% Offset Yield Stress')
                axes[0,1].set_title('Yield Stress vs Strain Rate')
                axes[0,1].set_xscale('log')
        
        # 3. Current vs No Current comparison
        if 'has_current' in df.columns and '0.2% Offset yield stress' in df.columns:
            current_data = df[df['has_current'] == True]['0.2% Offset yield stress'].dropna()
            no_current_data = df[df['has_current'] == False]['0.2% Offset yield stress'].dropna()
            
            axes[0,2].boxplot([current_data, no_current_data], labels=['With Current', 'No Current'])
            axes[0,2].set_ylabel('0.2% Offset Yield Stress')
            axes[0,2].set_title('Current Effect on Yield Stress')
        
        # 4. Strain Hardening Exponent
        if 'Strain handening exponent' in df.columns:
            df_clean = df[df['Strain handening exponent'].notna()]
            if not df_clean.empty:
                axes[1,0].hist(df_clean['Strain handening exponent'], bins=15, alpha=0.7)
                axes[1,0].set_xlabel('Strain Hardening Exponent')
                axes[1,0].set_ylabel('Frequency')
                axes[1,0].set_title('Distribution of Strain Hardening Exponent')
        
        # 5. Elongation at Break
        if 'Elogation at break (using Strain)' in df.columns:
            df_clean = df[df['Elogation at break (using Strain)'].notna()]
            if not df_clean.empty:
                axes[1,1].hist(df_clean['Elogation at break (using Strain)'], bins=15, alpha=0.7)
                axes[1,1].set_xlabel('Elongation at Break')
                axes[1,1].set_ylabel('Frequency')
                axes[1,1].set_title('Distribution of Elongation at Break')
        
        # 6. Folder comparison
        if 'folder' in df.columns and '0.2% Offset yield stress' in df.columns:
            folder_data = []
            labels = []
            for folder in ['Zn1', 'Zn2', 'Zn3', 'Zn4']:
                data = df[df['folder'] == folder]['0.2% Offset yield stress'].dropna()
                if len(data) > 0:
                    folder_data.append(data)
                    labels.append(folder)
            
            if folder_data:
                axes[1,2].boxplot(folder_data, labels=labels)
                axes[1,2].set_ylabel('0.2% Offset Yield Stress')
                axes[1,2].set_title('Yield Stress by Sample Set')
        
        plt.tight_layout()
        plt.savefig(self.main_files_dir / 'tensile_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create stress-strain curves plot
        if self.stress_strain_data:
            plt.figure(figsize=(12, 8))
            
            for key, df_ss in self.stress_strain_data.items():
                if 'True Stress' in df_ss.columns and 'True Strain' in df_ss.columns:
                    # Limit to first 50 points for clarity
                    stress = df_ss['True Stress'].iloc[:50]
                    strain = df_ss['True Strain'].iloc[:50]
                    plt.plot(strain, stress, label=key, alpha=0.7)
            
            plt.xlabel('True Strain')
            plt.ylabel('True Stress (MPa)')
            plt.title('True Stress-Strain Curves')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.main_files_dir / 'stress_strain_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_summary_statistics(self):
        """
        Generate comprehensive summary statistics
        """
        df = pd.DataFrame(self.all_data)
        
        summary = {
            'total_samples': len(df),
            'folders_processed': df['folder'].nunique() if 'folder' in df.columns else 0,
            'samples_with_current': len(df[df['has_current'] == True]) if 'has_current' in df.columns else 0,
            'samples_without_current': len(df[df['has_current'] == False]) if 'has_current' in df.columns else 0,
        }
        
        # Numerical properties statistics
        numerical_props = [
            '0.2% Offset yield stress',
            'Strain handening exponent', 
            'Strain handening coefficient',
            'Elogation at break (using Strain)',
            'Rate',
            'pulse_on_time',
            'pulse_total_time', 
            'amplitude',
            'strain_rate'
        ]
        
        for prop in numerical_props:
            if prop in df.columns:
                data = pd.to_numeric(df[prop], errors='coerce').dropna()
                if len(data) > 0:
                    summary[f'{prop}_mean'] = float(data.mean())
                    summary[f'{prop}_std'] = float(data.std())
                    summary[f'{prop}_min'] = float(data.min())
                    summary[f'{prop}_max'] = float(data.max())
                    summary[f'{prop}_count'] = int(len(data))
        
        return summary
    
    def sanitize_filename(self, filename):
        """
        Sanitize filename by removing/replacing problematic characters
        """
        import re
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[<>:"/\\|?*\s]', '_', str(filename))
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized
    
    def create_individual_sample_analysis(self, sample_data, stress_strain_df=None, key=None):
        """
        Create individual analysis for each sample including AI insights
        """
        try:
            # Prepare individual sample data for AI analysis
            sample_summary = {
                'sample_name': sample_data.get('sample_name', 'Unknown'),
                'folder': sample_data.get('folder', 'Unknown'),
                'has_current': sample_data.get('has_current', True),
                'pulse_on_time': sample_data.get('pulse_on_time'),
                'pulse_total_time': sample_data.get('pulse_total_time'),
                'amplitude': sample_data.get('amplitude'),
                'strain_rate': sample_data.get('strain_rate'),
                'yield_stress': sample_data.get('0.2% Offset yield stress'),
                'strain_hardening_exp': sample_data.get('Strain handening exponent'),
                'strain_hardening_coeff': sample_data.get('Strain handening coefficient'),
                'elongation_at_break': sample_data.get('Elogation at break (using Strain)'),
                'rate': sample_data.get('Rate')
            }
            
            # Create AI analysis prompt for individual sample
            prompt = f"""
            Analyze this individual zinc tensile test sample:
            
            Sample: {sample_summary['sample_name']} from {sample_summary['folder']}
            
            Processing Parameters:
            - Current Applied: {sample_summary['has_current']}
            - Pulse On Time: {sample_summary['pulse_on_time']} s
            - Pulse Total Time: {sample_summary['pulse_total_time']} s  
            - Amplitude: {sample_summary['amplitude']} A
            - Strain Rate: {sample_summary['strain_rate']} s⁻¹
            
            Mechanical Properties:
            - 0.2% Offset Yield Stress: {sample_summary['yield_stress']} MPa
            - Strain Hardening Exponent: {sample_summary['strain_hardening_exp']}
            - Strain Hardening Coefficient: {sample_summary['strain_hardening_coeff']}
            - Elongation at Break: {sample_summary['elongation_at_break']}
            - Test Rate: {sample_summary['rate']}
            
            Please provide:
            1. Analysis of the mechanical properties for this specific sample
            2. Effect of the processing parameters on the observed properties
            3. Comparison with typical zinc behavior
            4. Quality assessment of the test results
            5. Any notable observations or anomalies
            """
            
            # Get AI analysis for individual sample
            try:
                if self.ollama_client:
                    response = self.ollama_client.generate(
                        model=OLLAMA_MODEL,
                        prompt=prompt
                    )
                    individual_ai_analysis = response['response']
                else:
                    individual_ai_analysis = "Could not connect to Ollama model for individual sample analysis."
            except Exception as e:
                individual_ai_analysis = f"Error in individual sample AI analysis: {e}"
            
            return individual_ai_analysis
            
        except Exception as e:
            return f"Error creating individual sample analysis: {e}"
    
    def create_individual_sample_plot(self, sample_data, stress_strain_df, key):
        """
        Create individual stress-strain plot for a sample
        """
        try:
            if stress_strain_df is not None and 'True Stress' in stress_strain_df.columns and 'True Strain' in stress_strain_df.columns:
                plt.figure(figsize=(10, 6))
                
                stress = stress_strain_df['True Stress']
                strain = stress_strain_df['True Strain']
                
                plt.plot(strain, stress, 'b-', linewidth=2, label=f'{key}')
                plt.xlabel('True Strain')
                plt.ylabel('True Stress (MPa)')
                plt.title(f'True Stress-Strain Curve: {key}')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Add annotations for key properties
                if sample_data.get('0.2% Offset yield stress'):
                    try:
                        yield_stress = float(sample_data['0.2% Offset yield stress'])
                        plt.axhline(y=yield_stress, 
                                  color='r', linestyle='--', alpha=0.7, 
                                  label=f"Yield Stress: {yield_stress:.1f} MPa")
                        plt.legend()
                    except:
                        pass
                
                plt.tight_layout()
                plot_filename = self.individual_plots_dir / f'plot_{key}.png'
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                return plot_filename
            else:
                return None
        except Exception as e:
            print(f"Error creating plot for {key}: {e}")
            return None
    
    def create_individual_markdown(self, sample_data, ai_analysis, plot_filename, key):
        """
        Create individual markdown report for each sample
        """
        def safe_format(value, default='N/A'):
            return value if value is not None else default
        
        markdown_content = f"""# Individual Sample Analysis: {key}

## Sample Information
- **Sample Name:** {safe_format(sample_data.get('sample_name'))}
- **Folder/Set:** {safe_format(sample_data.get('folder'))}
- **Filename:** {safe_format(sample_data.get('filename'))}
- **Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Processing Parameters
- **Current Applied:** {'Yes' if sample_data.get('has_current', True) else 'No'}
- **Pulse On Time:** {safe_format(sample_data.get('pulse_on_time'))} s
- **Pulse Total Time:** {safe_format(sample_data.get('pulse_total_time'))} s
- **Amplitude:** {safe_format(sample_data.get('amplitude'))} A
- **Strain Rate:** {safe_format(sample_data.get('strain_rate'))} s⁻¹
- **Temperature:** {safe_format(sample_data.get('temperature'))} °C

## Mechanical Properties (from Report Sheet)
- **Test Rate:** {safe_format(sample_data.get('Rate'))}
- **0.2% Offset Yield Stress:** {safe_format(sample_data.get('0.2% Offset yield stress'))} MPa
- **Strain Hardening Exponent:** {safe_format(sample_data.get('Strain handening exponent'))}
- **Strain Hardening Coefficient:** {safe_format(sample_data.get('Strain handening coefficient'))}
- **Elongation at Break:** {safe_format(sample_data.get('Elogation at break (using Strain)'))}

## Stress-Strain Data
- **Stress-Strain Data Available:** {'Yes' if sample_data.get('has_stress_strain_data', False) else 'No'}

{f'![Stress-Strain Curve](plot_{key}.png)' if plot_filename else '**Note:** Stress-strain curve not available for this sample'}

## AI Analysis

{ai_analysis}

## Data Quality Assessment
- **Report Sheet Data:** {'✓ Available' if any(sample_data.get(prop) for prop in ['Rate', '0.2% Offset yield stress', 'Strain handening exponent']) else '✗ Limited/Missing'}
- **Stress-Strain Data:** {'✓ Available' if sample_data.get('has_stress_strain_data', False) else '✗ Not Available'}

## Notes
This analysis was generated automatically from the tensile test data. The processing parameters were extracted from the filename convention, and mechanical properties were extracted from the Excel Report sheet.

---
*Generated by Individual Sample Analyzer on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save individual markdown file
        md_filename = self.individual_analyses_dir / f'analysis_{key}.md'
        with open(md_filename, 'w') as f:
            f.write(markdown_content)
        
        return md_filename

    def save_results(self):
        """
        Save all results to files including individual sample analyses
        """
        # Save raw data
        df = pd.DataFrame(self.all_data)
        df.to_csv(self.main_files_dir / 'tensile_test_data.csv', index=False)
        df.to_excel(self.main_files_dir / 'tensile_test_data.xlsx', index=False)
        
        # Save stress-strain data and create individual analyses
        individual_analyses = []
        
        for i, sample_data in enumerate(self.all_data):
            # Use stored key or create a fallback key
            key = sample_data.get('stress_strain_key')
            if not key:
                key = f"{sample_data.get('folder', 'Unknown')}_{sample_data.get('sample_name', f'sample_{i}')}"
            
            # Sanitize key for safe file naming
            safe_key = self.sanitize_filename(key)
            
            # Save individual CSV if stress-strain data exists
            if sample_data.get('has_stress_strain_data', False) and key in self.stress_strain_data:
                ss_df = self.stress_strain_data[key]
                ss_df.to_csv(self.csv_files_dir / f'stress_strain_{safe_key}.csv', index=False)
                
                # Create individual plot
                plot_filename = self.create_individual_sample_plot(sample_data, ss_df, safe_key)
                
                # Get AI analysis for individual sample
                individual_ai_analysis = self.create_individual_sample_analysis(sample_data, ss_df, key)
                
                # Create individual markdown
                md_filename = self.create_individual_markdown(sample_data, individual_ai_analysis, plot_filename, safe_key)
                
                individual_analyses.append({
                    'key': key,
                    'safe_key': safe_key,
                    'csv_file': f'stress_strain_{safe_key}.csv',
                    'plot_file': f'plot_{safe_key}.png' if plot_filename else None,
                    'markdown_file': f'analysis_{safe_key}.md',
                    'ai_analysis': individual_ai_analysis
                })
            else:
                # Create analysis even without stress-strain data
                individual_ai_analysis = self.create_individual_sample_analysis(sample_data, None, key)
                md_filename = self.create_individual_markdown(sample_data, individual_ai_analysis, None, safe_key)
                
                individual_analyses.append({
                    'key': key,
                    'safe_key': safe_key,
                    'csv_file': None,
                    'plot_file': None,
                    'markdown_file': f'analysis_{safe_key}.md',
                    'ai_analysis': individual_ai_analysis
                })
        
        # Generate and save summary
        summary = self.generate_summary_statistics()
        summary['individual_analyses_count'] = len(individual_analyses)
        
        with open(self.main_files_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Get overall AI analysis
        data_summary = json.dumps(summary, indent=2)
        ai_analysis = self.analyze_with_ollama(data_summary)
        
        with open(self.main_files_dir / 'ai_analysis.txt', 'w') as f:
            f.write(ai_analysis)
        
        # Save individual analyses summary
        with open(self.main_files_dir / 'individual_analyses_summary.json', 'w') as f:
            json.dump(individual_analyses, f, indent=2)
        
        print(f"Created {len(individual_analyses)} individual sample analyses")
        
        return summary, ai_analysis
    
    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline
        """
        print("Starting tensile test data analysis...")
        
        # Process all files
        self.process_all_files()
        print(f"Processed {len(self.all_data)} samples")
        
        # Create visualizations
        print("Creating visualizations...")
        self.create_visualizations()
        
        # Save results
        print("Saving results...")
        summary, ai_analysis = self.save_results()
        
        print(f"Analysis complete! Results saved to {self.result_dir}")
        return summary, ai_analysis

def create_analysis_markdown(summary, ai_analysis):
    """
    Create a comprehensive markdown report
    """
    
    def format_value(value, fmt):
        """Helper function to safely format numerical values"""
        if value == 'N/A' or value is None:
            return 'N/A'
        try:
            return fmt.format(value)
        except:
            return str(value)
    
    # Get values with safe formatting
    yield_mean = format_value(summary.get('0.2% Offset yield stress_mean', 'N/A'), '{:.2f}')
    yield_std = format_value(summary.get('0.2% Offset yield stress_std', 'N/A'), '{:.2f}')
    yield_min = format_value(summary.get('0.2% Offset yield stress_min', 'N/A'), '{:.2f}')
    yield_max = format_value(summary.get('0.2% Offset yield stress_max', 'N/A'), '{:.2f}')
    
    strain_exp_mean = format_value(summary.get('Strain handening exponent_mean', 'N/A'), '{:.3f}')
    strain_exp_std = format_value(summary.get('Strain handening exponent_std', 'N/A'), '{:.3f}')
    strain_exp_min = format_value(summary.get('Strain handening exponent_min', 'N/A'), '{:.3f}')
    strain_exp_max = format_value(summary.get('Strain handening exponent_max', 'N/A'), '{:.3f}')
    
    elong_mean = format_value(summary.get('Elogation at break (using Strain)_mean', 'N/A'), '{:.3f}')
    elong_std = format_value(summary.get('Elogation at break (using Strain)_std', 'N/A'), '{:.3f}')
    elong_min = format_value(summary.get('Elogation at break (using Strain)_min', 'N/A'), '{:.3f}')
    elong_max = format_value(summary.get('Elogation at break (using Strain)_max', 'N/A'), '{:.3f}')
    
    amp_mean = format_value(summary.get('amplitude_mean', 'N/A'), '{:.1f}')
    amp_min = format_value(summary.get('amplitude_min', 'N/A'), '{:.0f}')
    amp_max = format_value(summary.get('amplitude_max', 'N/A'), '{:.0f}')
    
    pulse_on_mean = format_value(summary.get('pulse_on_time_mean', 'N/A'), '{:.2f}')
    pulse_on_min = format_value(summary.get('pulse_on_time_min', 'N/A'), '{:.2f}')
    pulse_on_max = format_value(summary.get('pulse_on_time_max', 'N/A'), '{:.2f}')
    
    sr_mean = format_value(summary.get('strain_rate_mean', 'N/A'), '{:.4f}')
    sr_min = format_value(summary.get('strain_rate_min', 'N/A'), '{:.4f}')
    sr_max = format_value(summary.get('strain_rate_max', 'N/A'), '{:.4f}')
    
    markdown_content = f"""# Tensile Test Data Analysis Report

## Overview
This report presents a comprehensive analysis of tensile test data from zinc samples across four sample sets (Zn1, Zn2, Zn3, Zn4).

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics

- **Total Samples Analyzed:** {summary.get('total_samples', 'N/A')}
- **Sample Sets Processed:** {summary.get('folders_processed', 'N/A')}
- **Samples with Current:** {summary.get('samples_with_current', 'N/A')}
- **Samples without Current:** {summary.get('samples_without_current', 'N/A')}

## Mechanical Properties Summary

### 0.2% Offset Yield Stress
- **Mean:** {yield_mean} MPa
- **Standard Deviation:** {yield_std} MPa
- **Range:** {yield_min} - {yield_max} MPa
- **Sample Count:** {summary.get('0.2% Offset yield stress_count', 'N/A')}

### Strain Hardening Exponent
- **Mean:** {strain_exp_mean}
- **Standard Deviation:** {strain_exp_std}
- **Range:** {strain_exp_min} - {strain_exp_max}

### Elongation at Break
- **Mean:** {elong_mean}
- **Standard Deviation:** {elong_std}
- **Range:** {elong_min} - {elong_max}

## Processing Parameters Summary

### Pulse Amplitude
- **Mean:** {amp_mean} A
- **Range:** {amp_min} - {amp_max} A

### Pulse On Time
- **Mean:** {pulse_on_mean} s
- **Range:** {pulse_on_min} - {pulse_on_max} s

### Strain Rate
- **Mean:** {sr_mean} s⁻¹
- **Range:** {sr_min} - {sr_max} s⁻¹

## AI Analysis and Insights

{ai_analysis}

## Data Files Generated

### Main Analysis Files
1. **tensile_test_data.csv/xlsx** - Complete dataset with all extracted parameters and properties
2. **summary_statistics.json** - Statistical summary in JSON format
3. **tensile_analysis_plots.png** - Comprehensive visualization plots
4. **stress_strain_curves.png** - True stress-strain curves overlay
5. **ai_analysis.txt** - Detailed overall AI analysis

### Individual Sample Files
6. **stress_strain_[folder]_[sample].csv** - Individual stress-strain data for each sample
7. **plot_[folder]_[sample].png** - Individual stress-strain plots for each sample
8. **analysis_[folder]_[sample].md** - Individual markdown analysis reports for each sample
9. **individual_analyses_summary.json** - Summary of all individual analyses created

**Total Individual Analyses Created:** {summary.get('individual_analyses_count', 'N/A')}

## Methodology

### Data Extraction
- Experimental parameters extracted from filename parsing
- Mechanical properties extracted from Excel 'Report' sheets
- True stress-strain data extracted from 'TrueStress-TrueStrain data' sheets

### Filename Convention Parsing
- Format: `epX_pulse_on_pulse_total_amplitudeA_Tensile_`
- No current samples identified by "no current" in filename
- Strain rate extracted when specified (sr parameter)

### Analysis Approach
- Statistical analysis of mechanical properties
- Correlation analysis between processing parameters and properties
- Visualization of trends and distributions
- AI-powered insights using Ollama Gemma3:4b model

## Conclusions

The analysis provides comprehensive insights into the relationship between electroplastic processing parameters and mechanical properties of zinc samples. Key findings include the effects of pulse parameters, current application, and strain rate on yield strength, strain hardening behavior, and ductility.

---
*Generated by Tensile Test Data Analyzer*
"""
    
    with open('tensile_analysis_report.md', 'w') as f:
        f.write(markdown_content)
    
    print("Markdown report created: tensile_analysis_report.md")

if __name__ == "__main__":
    # Run the complete analysis
    analyzer = TensileDataAnalyzer()
    summary, ai_analysis = analyzer.run_complete_analysis()
    
    # Create markdown report
    create_analysis_markdown(summary, ai_analysis)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Results saved to: {analyzer.result_dir}")
    print("Markdown report: tensile_analysis_report.md") 