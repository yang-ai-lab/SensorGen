import re
from typing import List, Tuple, Union
import random


def generate_clinical_prompts(reports: List[str], format_type: str = "numbered", random_format: bool = False) -> List[str]:
    """
    Convert clinical ECG reports into structured, fluent sentences for CLIP text encoders.
    
    Args:
        reports: List of clinical reports, each containing multiple diagnoses
        format_type: One of "numbered", "sequential", "hierarchical", or "summary"
        random_format: If True, randomly choose format for each report in the batch
    
    Returns:
        List of formatted prompts
    """
    
    def clean_diagnosis(diagnosis: str) -> str:
        """Clean and normalize individual diagnosis text."""
        # Remove extra whitespace and punctuation
        diagnosis = re.sub(r'\s+', ' ', diagnosis.strip())
        # Remove trailing punctuation
        diagnosis = re.sub(r'[.,;]+$', '', diagnosis)
        # Capitalize first letter
        diagnosis = diagnosis[0].upper() + diagnosis[1:] if diagnosis else ""
        return diagnosis
    
    def split_diagnoses(report: str) -> List[str]:
        """Split a report into individual diagnoses."""
        # Split by common separators
        diagnoses = re.split(r'[.;]', report)
        # Clean and filter out empty diagnoses
        diagnoses = [clean_diagnosis(d) for d in diagnoses if clean_diagnosis(d)]
        return diagnoses
    
    def format_numbered(diagnoses: List[str]) -> str:
        """Format: "The first diagnosis is __. The second diagnosis is __..." """
        if not diagnoses:
            return "No specific diagnoses found."
        
        formatted_parts = []
        for i, diagnosis in enumerate(diagnoses, 1):
            ordinal = get_ordinal(i)
            formatted_parts.append(f"The {ordinal} diagnosis is {diagnosis}.")
        
        return " ".join(formatted_parts)
    
    def format_sequential(diagnoses: List[str]) -> str:
        """Format: "This ECG shows __, followed by __, and also indicates __" """
        if not diagnoses:
            return "No specific findings detected on this ECG."
        
        if len(diagnoses) == 1:
            return f"This ECG shows {diagnoses[0]}."
        
        formatted_parts = [f"This ECG shows {diagnoses[0]}"]
        
        if len(diagnoses) == 2:
            formatted_parts.append(f"and also indicates {diagnoses[1]}.")
        else:
            for i, diagnosis in enumerate(diagnoses[1:-1], 2):
                formatted_parts.append(f"followed by {diagnosis}")
            formatted_parts.append(f"and also indicates {diagnoses[-1]}.")
        
        return " ".join(formatted_parts)
    
    def format_hierarchical(diagnoses: List[str]) -> str:
        """Format: "Primary finding: __. Secondary findings include __ and __" """
        if not diagnoses:
            return "No significant findings detected."
        
        if len(diagnoses) == 1:
            return f"Primary finding: {diagnoses[0]}."
        
        primary = diagnoses[0]
        secondary = diagnoses[1:]
        
        if len(secondary) == 1:
            return f"Primary finding: {primary}. Secondary finding: {secondary[0]}."
        else:
            secondary_text = ", ".join(secondary[:-1]) + f" and {secondary[-1]}"
            return f"Primary finding: {primary}. Secondary findings include {secondary_text}."
    
    def format_summary(diagnoses: List[str]) -> str:
        """Format: "ECG interpretation reveals multiple abnormalities including __" """
        if not diagnoses:
            return "ECG interpretation shows normal findings."
        
        if len(diagnoses) == 1:
            return f"ECG interpretation reveals {diagnoses[0]}."
        
        # Join all diagnoses with commas and "and"
        if len(diagnoses) == 2:
            diagnosis_text = f"{diagnoses[0]} and {diagnoses[1]}"
        else:
            diagnosis_text = ", ".join(diagnoses[:-1]) + f" and {diagnoses[-1]}"
        
        return f"ECG interpretation reveals multiple abnormalities including {diagnosis_text}."
    
    def get_ordinal(n: int) -> str:
        """Convert number to ordinal string."""
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"
    
    # Format mapping
    format_functions = {
        "numbered": format_numbered,
        "sequential": format_sequential,
        "hierarchical": format_hierarchical,
        "summary": format_summary
    }
    
    if format_type not in format_functions:
        raise ValueError(f"format_type must be one of {list(format_functions.keys())}")
    
    formatted_prompts = []
    available_formats = list(format_functions.keys())
    
    for report in reports:
        # Randomly choose format if requested
        if random_format:
            chosen_format = random.choice(available_formats)
        else:
            chosen_format = format_type
            
        format_func = format_functions[chosen_format]
        diagnoses = split_diagnoses(report)
        formatted_prompt = format_func(diagnoses)
        formatted_prompts.append(formatted_prompt)
    
    return formatted_prompts


def generate_all_formats(reports: List[str]) -> dict:
    """
    Generate all four formats for comparison.
    
    Args:
        reports: List of clinical reports
    
    Returns:
        Dictionary with all formatted versions
    """
    formats = ["numbered", "sequential", "hierarchical", "summary"]
    results = {}
    
    for format_type in formats:
        results[format_type] = generate_clinical_prompts(reports, format_type)
    
    return results


def generate_random_formats(reports: List[str], seed: int = 42) -> List[str]:
    """
    Generate prompts with randomly chosen formats for each report.
    
    Args:
        reports: List of clinical reports
        seed: Random seed for reproducibility
    
    Returns:
        List of formatted prompts with random formats
    """
    if seed is not None:
        random.seed(seed)
    
    return generate_clinical_prompts(reports, format_type="numbered", random_format=True)


# Example usage and testing
if __name__ == "__main__":
    # Example reports
    sample_reports = [
        "sinus bradycardia. prolonged qt interval. inferior infarct - age undetermined. lateral t wave changes are nonspecific. abnormal ecg.",
        "normal sinus rhythm. no significant st segment changes. normal axis. normal intervals.",
        "atrial fibrillation. rapid ventricular response. left bundle branch block. st depression in lateral leads."
    ]
    
    print("Original Reports:")
    for i, report in enumerate(sample_reports, 1):
        print(f"{i}. {report}")
    
    print("\n" + "="*80)
    
    # Generate all formats
    all_formats = generate_all_formats(sample_reports)
    
    for format_name, formatted_reports in all_formats.items():
        print(f"\n{format_name.upper()} FORMAT:")
        print("-" * 40)
        for i, prompt in enumerate(formatted_reports, 1):
            print(f"{i}. {prompt}")
    
    # Test individual format
    print("\n" + "="*80)
    print("Testing individual format (numbered):")
    numbered_prompts = generate_clinical_prompts(sample_reports, "numbered")
    for prompt in numbered_prompts:
        print(f"- {prompt}")
    
    # Test random format generation
    print("\n" + "="*80)
    print("Testing random format generation:")
    random_prompts = generate_random_formats(sample_reports, seed=42)
    for i, prompt in enumerate(random_prompts, 1):
        print(f"{i}. {prompt}")
    
    # Test another batch with different seed
    print("\n" + "="*80)
    print("Testing random format generation (different seed):")
    random_prompts_2 = generate_random_formats(sample_reports, seed=123)
    for i, prompt in enumerate(random_prompts_2, 1):
        print(f"{i}. {prompt}")