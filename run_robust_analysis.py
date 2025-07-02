#!/usr/bin/env python3
"""
Robust Tensile Test Data Analysis Script
Analyzes tensile test data from zinc sample Excel files with progress tracking
"""

import pandas as pd
import numpy as np
import os
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import ollama
from datetime import datetime
import warnings
import time
warnings.filterwarnings('ignore')

# Ollama configuration
OLLAMA_MODEL = "mistral-nemo"
OLLAMA_HOST = "http://localhost:11434"

class RobustTensileDataAnalyzer:
    def __init__(self, data_dir="data/zinc", result_dir="data/result", skip_ai_analysis=False):
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.skip_ai_analysis = skip_ai_analysis
        
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
        
        # Initialize Ollama client if AI analysis is enabled
        if not self.skip_ai_analysis:
            self.ollama_client = self.connect_ollama()
        else:
            self.ollama_client = None
            print("AI analysis disabled - processing will be faster")
    
    def connect_ollama(self):
        """Connects to the local Ollama instance and verifies the model."""
        print(f"Connecting to Ollama at {OLLAMA_HOST}...")
        try:
            client = ollama.Client(host=OLLAMA_HOST)
            client.show(OLLAMA_MODEL)
            print(f"Successfully connected to Ollama and found model '{OLLAMA_MODEL}'.")
            return client
        except Exception as e:
            print(f"Failed to connect to Ollama or find model '{OLLAMA_MODEL}': {e}")
            print("Proceeding without AI analysis...")
            return None
    
    def parse_filename(self, filename):
        """Parse experimental parameters from filename"""
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
        
        # Extract sample name
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
        
        # Extract pulse parameters
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
        """Read data from Excel file - both TrueStress-TrueStrain data and Report sheets"""
        try:
            # Read TrueStress-TrueStrain data
            stress_strain_df = None
            try:
                stress_strain_df = pd.read_excel(file_path, sheet_name='TrueStress-TrueStrain data')
            except:
                pass  # Silently continue if sheet doesn't exist
            
            # Read Report data with improved extraction
            report_data = {}
            try:
                report_df = pd.read_excel(file_path, sheet_name='Report', header=None)
                
                properties_to_extract = {
                    'Rate': ['rate', 'speed', 'velocity'],
                    '0.2% Offset yield stress': ['yield', '0.2%', 'offset', 'proof'],
                    'Strain handening exponent': ['strain hardening', 'n-value', 'hardening exponent', 'strain exponent'],
                    'Strain handening coefficient': ['strain hardening coefficient', 'k-value', 'hardening coefficient'],
                    'Elogation at break (using Strain)': ['elongation', 'break', 'failure', 'fracture', 'strain at break']
                }
                
                search_data = report_df.astype(str).fillna('')
                
                for prop, keywords in properties_to_extract.items():
                    found = False
                    for idx, row in search_data.iterrows():
                        for col_idx, cell in enumerate(row):
                            cell_lower = cell.lower()
                            if any(keyword in cell_lower for keyword in keywords):
                                for next_col in range(col_idx + 1, min(col_idx + 4, len(row))):
                                    try:
                                        value = report_df.iloc[idx, next_col]
                                        if pd.notna(value) and str(value).strip() != '':
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
                pass  # Silently continue if Report sheet doesn't exist
            
            return stress_strain_df, report_data
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None, {}
    
    def process_all_files(self):
        """Process all Excel files with progress tracking"""
        folders = ['Zn1', 'Zn2', 'Zn3', 'Zn4']
        total_files = 0
        
        # Count total files first
        for folder in folders:
            folder_path = self.data_dir / folder
            if folder_path.exists():
                total_files += len(list(folder_path.glob('*.xls')))
        
        print(f"Found {total_files} Excel files to process")
        
        processed_files = 0
        
        for folder in folders:
            folder_path = self.data_dir / folder
            if not folder_path.exists():
                continue
                
            print(f"\nProcessing folder: {folder}")
            xls_files = list(folder_path.glob('*.xls'))
            
            for file_path in xls_files:
                processed_files += 1
                progress = (processed_files / total_files) * 100
                print(f"  [{progress:5.1f}%] Processing: {file_path.name}")
                
                try:
                    # Parse filename
                    params = self.parse_filename(file_path.name)
                    params['folder'] = folder
                    
                    # Read Excel data
                    stress_strain_df, report_data = self.read_excel_data(file_path)
                    
                    # Combine all data
                    combined_data = {**params, **report_data}
                    
                    # Store stress-strain data
                    if stress_strain_df is not None:
                        key = f"{folder}_{params['sample_name']}"
                        self.stress_strain_data[key] = stress_strain_df
                        combined_data['has_stress_strain_data'] = True
                        combined_data['stress_strain_key'] = key
                    else:
                        combined_data['has_stress_strain_data'] = False
                        combined_data['stress_strain_key'] = None
                    
                    self.all_data.append(combined_data)
                    
                except Exception as e:
                    print(f"    Error processing {file_path.name}: {e}")
                    continue
        
        print(f"\nCompleted processing {len(self.all_data)} samples")
    
    def create_individual_plots_batch(self):
        """Create all individual plots in batch"""
        print("\nCreating individual stress-strain plots...")
        plot_count = 0
        
        for i, sample_data in enumerate(self.all_data):
            if sample_data.get('has_stress_strain_data', False):
                key = sample_data.get('stress_strain_key')
                if key and key in self.stress_strain_data:
                    safe_key = self.sanitize_filename(key)
                    
                    try:
                        ss_df = self.stress_strain_data[key]
                        
                        # Check for different possible column names
                        stress_col = None
                        strain_col = None
                        
                        for col in ss_df.columns:
                            if 'true stress' in col.lower():
                                stress_col = col
                            elif 'true strain' in col.lower():
                                strain_col = col
                        
                        if stress_col and strain_col:
                            plt.figure(figsize=(10, 6))
                            
                            stress = ss_df[stress_col]
                            strain = ss_df[strain_col]
                            
                            plt.plot(strain, stress, 'b-', linewidth=2, label=f'{key}')
                            plt.xlabel('True Strain (%)')
                            plt.ylabel('True Stress (MPa)')
                            plt.title(f'True Stress-Strain Curve: {key}')
                            plt.grid(True, alpha=0.3)
                            plt.legend()
                            
                            # Add yield stress line if available
                            if sample_data.get('0.2% Offset yield stress'):
                                try:
                                    yield_stress = float(sample_data['0.2% Offset yield stress'])
                                    if yield_stress > 0:
                                        plt.axhline(y=yield_stress, color='r', linestyle='--', alpha=0.7, 
                                                  label=f"Yield Stress: {yield_stress:.1f} MPa")
                                        plt.legend()
                                except:
                                    pass
                            
                            plt.tight_layout()
                            plot_filename = self.individual_plots_dir / f'plot_{safe_key}.png'
                            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                            plt.close()
                            
                            plot_count += 1
                            if plot_count % 10 == 0:
                                print(f"  Created {plot_count} plots...")
                    
                    except Exception as e:
                        print(f"  Error creating plot for {key}: {e}")
                        continue
        
        print(f"Created {plot_count} individual plots")
    
    def sanitize_filename(self, filename):
        """Sanitize filename by removing/replacing problematic characters"""
        import re
        sanitized = re.sub(r'[<>:"/\\|?*\s]', '_', str(filename))
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_')
        return sanitized
    
    def create_individual_analyses_batch(self):
        """Create individual analysis files in batch"""
        if self.skip_ai_analysis:
            print("\nCreating individual analysis files (without AI analysis)...")
        else:
            print("\nCreating individual analysis files with AI analysis...")
        
        analysis_count = 0
        
        for i, sample_data in enumerate(self.all_data):
            try:
                # Create unique key
                key = sample_data.get('stress_strain_key')
                if not key:
                    key = f"{sample_data.get('folder', 'Unknown')}_{sample_data.get('sample_name', f'sample_{i}')}"
                
                safe_key = self.sanitize_filename(key)
                
                # Get AI analysis if enabled
                if self.ollama_client and not self.skip_ai_analysis:
                    ai_analysis = self.get_individual_ai_analysis(sample_data, key)
                else:
                    ai_analysis = "AI analysis skipped for faster processing. Enable AI analysis by setting skip_ai_analysis=False."
                
                # Create markdown content
                markdown_content = self.create_individual_markdown_content(sample_data, ai_analysis, safe_key)
                
                # Save markdown file
                md_filename = self.individual_analyses_dir / f'analysis_{safe_key}.md'
                with open(md_filename, 'w') as f:
                    f.write(markdown_content)
                
                analysis_count += 1
                if analysis_count % 10 == 0:
                    print(f"  Created {analysis_count} analysis files...")
                    
            except Exception as e:
                print(f"  Error creating analysis for sample {i}: {e}")
                continue
        
        print(f"Created {analysis_count} individual analysis files")
    
    def get_individual_ai_analysis(self, sample_data, key):
        """Get AI analysis for individual sample"""
        try:
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
            
            prompt = f"""
            Analyze this zinc tensile test sample briefly (max 200 words):
            
            Sample: {sample_summary['sample_name']} from {sample_summary['folder']}
            
            Processing: Current={sample_summary['has_current']}, Pulse On={sample_summary['pulse_on_time']}s, 
            Amplitude={sample_summary['amplitude']}A, Strain Rate={sample_summary['strain_rate']}s⁻¹
            
            Properties: Yield={sample_summary['yield_stress']}MPa, n={sample_summary['strain_hardening_exp']}, 
            Elongation={sample_summary['elongation_at_break']}%
            
            Provide brief insights on the mechanical properties and processing effects.
            """
            
            if self.ollama_client:
                response = self.ollama_client.generate(model=OLLAMA_MODEL, prompt=prompt)
                return response['response']
            else:
                return "Could not connect to Ollama model for individual sample analysis."
                
        except Exception as e:
            return f"Error in individual sample AI analysis: {e}"
    
    def create_individual_markdown_content(self, sample_data, ai_analysis, safe_key):
        """Create markdown content for individual sample"""
        def safe_format(value, default='N/A'):
            return value if value is not None else default
        
        has_plot = (self.individual_plots_dir / f'plot_{safe_key}.png').exists()
        
        return f"""# Individual Sample Analysis: {safe_key}

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
- **Test Rate:** {safe_format(sample_data.get('Rate'))} mm/s
- **0.2% Offset Yield Stress:** {safe_format(sample_data.get('0.2% Offset yield stress'))} MPa
- **Strain Hardening Exponent:** {safe_format(sample_data.get('Strain handening exponent'))}
- **Strain Hardening Coefficient:** {safe_format(sample_data.get('Strain handening coefficient'))}
- **Elongation at Break:** {safe_format(sample_data.get('Elogation at break (using Strain)'))}

## Stress-Strain Data
- **Stress-Strain Data Available:** {'Yes' if sample_data.get('has_stress_strain_data', False) else 'No'}

{f'![Stress-Strain Curve](../individual_plots/plot_{safe_key}.png)' if has_plot else '**Note:** Stress-strain curve not available for this sample'}

## AI Analysis

{ai_analysis}

## Data Quality Assessment
- **Report Sheet Data:** {'✓ Available' if any(sample_data.get(prop) for prop in ['Rate', '0.2% Offset yield stress', 'Strain handening exponent']) else '✗ Limited/Missing'}
- **Stress-Strain Data:** {'✓ Available' if sample_data.get('has_stress_strain_data', False) else '✗ Not Available'}

---
*Generated by Robust Tensile Data Analyzer on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    def create_main_files(self):
        """Create main analysis files"""
        print("\nCreating main analysis files...")
        
        # Save raw data
        df = pd.DataFrame(self.all_data)
        df.to_csv(self.main_files_dir / 'tensile_test_data.csv', index=False)
        df.to_excel(self.main_files_dir / 'tensile_test_data.xlsx', index=False)
        
        # Save individual CSV files
        csv_count = 0
        for sample_data in self.all_data:
            if sample_data.get('has_stress_strain_data', False):
                key = sample_data.get('stress_strain_key')
                if key and key in self.stress_strain_data:
                    safe_key = self.sanitize_filename(key)
                    ss_df = self.stress_strain_data[key]
                    ss_df.to_csv(self.csv_files_dir / f'stress_strain_{safe_key}.csv', index=False)
                    csv_count += 1
        
        print(f"Saved {csv_count} individual CSV files")
        
        # Create visualizations
        self.create_main_visualizations()
        
        # Generate summary statistics
        summary = self.generate_summary_statistics()
        with open(self.main_files_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Get overall AI analysis if enabled
        if self.ollama_client and not self.skip_ai_analysis:
            print("Generating overall AI analysis...")
            data_summary = json.dumps(summary, indent=2)
            ai_analysis = self.get_overall_ai_analysis(data_summary)
        else:
            ai_analysis = "Overall AI analysis skipped for faster processing."
        
        with open(self.main_files_dir / 'ai_analysis.txt', 'w') as f:
            f.write(ai_analysis)
        
        return summary, ai_analysis
    
    def create_main_visualizations(self):
        """Create main visualization plots"""
        df = pd.DataFrame(self.all_data)
        
        # Set up plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                pass
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Tensile Test Data Analysis - Zinc Samples', fontsize=16)
        
        # Create plots (simplified for speed)
        self.create_summary_plots(df, axes)
        
        plt.tight_layout()
        plt.savefig(self.main_files_dir / 'tensile_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create stress-strain overlay
        if self.stress_strain_data:
            plt.figure(figsize=(12, 8))
            plot_count = 0
            
            for key, df_ss in self.stress_strain_data.items():
                if plot_count < 20:  # Limit for performance
                    # Check for different possible column names
                    stress_col = None
                    strain_col = None
                    
                    for col in df_ss.columns:
                        if 'true stress' in col.lower():
                            stress_col = col
                        elif 'true strain' in col.lower():
                            strain_col = col
                    
                    if stress_col and strain_col:
                        stress = df_ss[stress_col].iloc[:50]
                        strain = df_ss[strain_col].iloc[:50]
                        plt.plot(strain, stress, label=key, alpha=0.7)
                        plot_count += 1
            
            plt.xlabel('True Strain (%)')
            plt.ylabel('True Stress (MPa)')
            plt.title('True Stress-Strain Curves (Sample)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.main_files_dir / 'stress_strain_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_summary_plots(self, df, axes):
        """Create summary plots"""
        # Plot 1: Amplitude distribution
        if 'amplitude' in df.columns:
            amplitude_data = df['amplitude'].dropna()
            if not amplitude_data.empty:
                axes[0,0].hist(amplitude_data, bins=15, alpha=0.7)
                axes[0,0].set_xlabel('Amplitude (A)')
                axes[0,0].set_ylabel('Frequency')
                axes[0,0].set_title('Pulse Amplitude Distribution')
        
        # Plot 2: Pulse time distribution
        if 'pulse_on_time' in df.columns:
            pulse_data = df['pulse_on_time'].dropna()
            if not pulse_data.empty:
                axes[0,1].hist(pulse_data, bins=15, alpha=0.7)
                axes[0,1].set_xlabel('Pulse On Time (s)')
                axes[0,1].set_ylabel('Frequency')
                axes[0,1].set_title('Pulse On Time Distribution')
        
        # Plot 3: Current vs No Current
        if 'has_current' in df.columns:
            current_counts = df['has_current'].value_counts()
            axes[0,2].pie(current_counts.values, labels=['With Current', 'No Current'], autopct='%1.1f%%')
            axes[0,2].set_title('Current Application Distribution')
        
        # Plot 4: Folder distribution
        if 'folder' in df.columns:
            folder_counts = df['folder'].value_counts()
            axes[1,0].bar(folder_counts.index, folder_counts.values)
            axes[1,0].set_xlabel('Sample Folder')
            axes[1,0].set_ylabel('Number of Samples')
            axes[1,0].set_title('Samples per Folder')
        
        # Plot 5: Yield stress distribution (if available)
        if '0.2% Offset yield stress' in df.columns:
            yield_data = pd.to_numeric(df['0.2% Offset yield stress'], errors='coerce').dropna()
            yield_data = yield_data[yield_data > 0]  # Remove zero values
            if not yield_data.empty:
                axes[1,1].hist(yield_data, bins=15, alpha=0.7)
                axes[1,1].set_xlabel('0.2% Offset Yield Stress (MPa)')
                axes[1,1].set_ylabel('Frequency')
                axes[1,1].set_title('Yield Stress Distribution')
        
        # Plot 6: Elongation distribution (if available)
        if 'Elogation at break (using Strain)' in df.columns:
            elong_data = pd.to_numeric(df['Elogation at break (using Strain)'], errors='coerce').dropna()
            if not elong_data.empty:
                axes[1,2].hist(elong_data, bins=15, alpha=0.7)
                axes[1,2].set_xlabel('Elongation at Break')
                axes[1,2].set_ylabel('Frequency')
                axes[1,2].set_title('Elongation Distribution')
    
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        df = pd.DataFrame(self.all_data)
        
        summary = {
            'total_samples': len(df),
            'folders_processed': df['folder'].nunique() if 'folder' in df.columns else 0,
            'samples_with_current': len(df[df['has_current'] == True]) if 'has_current' in df.columns else 0,
            'samples_without_current': len(df[df['has_current'] == False]) if 'has_current' in df.columns else 0,
            'samples_with_stress_strain_data': len(df[df['has_stress_strain_data'] == True]) if 'has_stress_strain_data' in df.columns else 0,
        }
        
        # Numerical properties statistics
        numerical_props = [
            '0.2% Offset yield stress', 'Strain handening exponent', 'Strain handening coefficient',
            'Elogation at break (using Strain)', 'Rate', 'pulse_on_time', 'pulse_total_time', 
            'amplitude', 'strain_rate'
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
    
    def get_overall_ai_analysis(self, data_summary):
        """Get overall AI analysis"""
        try:
            prompt = f"""
            Analyze this tensile test dataset from zinc samples:
            
            {data_summary}
            
            Provide insights on:
            1. Overall trends in mechanical properties
            2. Effect of processing parameters
            3. Comparison between current and no-current samples
            4. Recommendations for optimization
            
            Keep the analysis concise but comprehensive.
            """
            
            if self.ollama_client:
                response = self.ollama_client.generate(model=OLLAMA_MODEL, prompt=prompt)
                return response['response']
            else:
                return "Could not connect to Ollama model for overall analysis."
                
        except Exception as e:
            return f"Error in overall AI analysis: {e}"
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        start_time = time.time()
        print("="*60)
        print("ROBUST TENSILE TEST DATA ANALYSIS")
        print("="*60)
        
        # Process all files
        self.process_all_files()
        
        # Create individual plots
        self.create_individual_plots_batch()
        
        # Create individual analyses
        self.create_individual_analyses_batch()
        
        # Create main files
        summary, ai_analysis = self.create_main_files()
        
        elapsed_time = time.time() - start_time
        
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Total processing time: {elapsed_time:.1f} seconds")
        print(f"Processed {len(self.all_data)} samples")
        print(f"Results saved to: {self.result_dir}")
        print("="*60)
        
        return summary, ai_analysis

if __name__ == "__main__":
    # Run with AI analysis disabled for faster processing
    # Set skip_ai_analysis=False to enable AI analysis (much slower)
    analyzer = RobustTensileDataAnalyzer(skip_ai_analysis=True)
    summary, ai_analysis = analyzer.run_complete_analysis() 