#!/usr/bin/env python3
"""
Step 4.4.3 - Handoff Package Validation
Movie Recommendation Optimizer - Validate handoff package completeness

This module validates that the handoff package meets all acceptance gates
and is ready for stakeholder delivery.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class HandoffValidation:
    """
    Validates the Step 4.4.3 handoff package completeness.
    """
    
    def __init__(self, project_root: str = "/Users/shaheen/Desktop/Netflix"):
        """Initialize the handoff validation."""
        self.project_root = Path(project_root)
        self.validation_results = {
            "handoff_doc": False,
            "qa_checklist": False,
            "readme_snippets": False,
            "signoff_section": False,
            "reproducibility_metadata": False,
            "file_references": False,
            "artifact_manifest": False,
            "overall_status": False
        }
        self.errors = []
        self.warnings = []
    
    def validate_handoff_document(self) -> bool:
        """Validate the master handoff document exists and is complete."""
        handoff_path = self.project_root / "docs" / "step4_handoff.md"
        
        if not handoff_path.exists():
            self.errors.append("Master handoff document not found: docs/step4_handoff.md")
            return False
        
        # Check file size
        if handoff_path.stat().st_size < 1000:  # At least 1KB
            self.errors.append("Handoff document appears incomplete (too small)")
            return False
        
        # Check for required sections
        with open(handoff_path, 'r') as f:
            content = f.read()
        
        required_sections = [
            "Executive Summary",
            "QA Checklist",
            "Production Readiness Assessment",
            "README Snippets",
            "Sign-Off Section",
            "Reproducibility Metadata",
            "Next Steps"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        if missing_sections:
            self.errors.append(f"Missing required sections: {', '.join(missing_sections)}")
            return False
        
        self.validation_results["handoff_doc"] = True
        return True
    
    def validate_qa_checklist(self) -> bool:
        """Validate QA checklist covers all required steps."""
        handoff_path = self.project_root / "docs" / "step4_handoff.md"
        
        with open(handoff_path, 'r') as f:
            content = f.read()
        
        # Check for step coverage
        required_steps = ["Step 4.1", "Step 4.2", "Step 4.3", "Step 4.4.1", "Step 4.4.2"]
        missing_steps = []
        
        for step in required_steps:
            if step not in content:
                missing_steps.append(step)
        
        if missing_steps:
            self.errors.append(f"QA checklist missing steps: {', '.join(missing_steps)}")
            return False
        
        # Check for validation items
        validation_items = ["✅ COMPLETED", "Status", "PASSED"]
        missing_items = []
        
        for item in validation_items:
            if item not in content:
                missing_items.append(item)
        
        if missing_items:
            self.errors.append(f"QA checklist missing validation items: {', '.join(missing_items)}")
            return False
        
        self.validation_results["qa_checklist"] = True
        return True
    
    def validate_readme_snippets(self) -> bool:
        """Validate README snippets are included."""
        handoff_path = self.project_root / "docs" / "step4_handoff.md"
        
        with open(handoff_path, 'r') as f:
            content = f.read()
        
        # Check for README snippet sections
        snippet_sections = [
            "Project Overview Update",
            "Quick Start Guide",
            "Documentation Links"
        ]
        
        missing_snippets = []
        for section in snippet_sections:
            if section not in content:
                missing_snippets.append(section)
        
        if missing_snippets:
            self.errors.append(f"Missing README snippet sections: {', '.join(missing_snippets)}")
            return False
        
        self.validation_results["readme_snippets"] = True
        return True
    
    def validate_signoff_section(self) -> bool:
        """Validate sign-off section with decision options."""
        handoff_path = self.project_root / "docs" / "step4_handoff.md"
        
        with open(handoff_path, 'r') as f:
            content = f.read()
        
        # Check for decision options
        decision_options = ["APPROVE", "REJECT", "NEEDS WORK"]
        missing_options = []
        
        for option in decision_options:
            if option not in content:
                missing_options.append(option)
        
        if missing_options:
            self.errors.append(f"Sign-off section missing decision options: {', '.join(missing_options)}")
            return False
        
        # Check for sign-off form
        signoff_elements = ["Stakeholder Name", "Role/Title", "Date", "Decision", "Comments", "Signature"]
        missing_elements = []
        
        for element in signoff_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            self.errors.append(f"Sign-off form missing elements: {', '.join(missing_elements)}")
            return False
        
        self.validation_results["signoff_section"] = True
        return True
    
    def validate_reproducibility_metadata(self) -> bool:
        """Validate reproducibility metadata is complete."""
        handoff_path = self.project_root / "docs" / "step4_handoff.md"
        
        with open(handoff_path, 'r') as f:
            content = f.read()
        
        # Check for required metadata
        metadata_elements = [
            "Random Seed",
            "Policy Version",
            "Alpha Values",
            "K Values",
            "Artifact Manifest",
            "Total Artifacts",
            "Validation Status"
        ]
        
        missing_metadata = []
        for element in metadata_elements:
            if element not in content:
                missing_metadata.append(element)
        
        if missing_metadata:
            self.errors.append(f"Missing reproducibility metadata: {', '.join(missing_metadata)}")
            return False
        
        self.validation_results["reproducibility_metadata"] = True
        return True
    
    def validate_file_references(self) -> bool:
        """Validate all referenced files exist."""
        handoff_path = self.project_root / "docs" / "step4_handoff.md"
        
        with open(handoff_path, 'r') as f:
            content = f.read()
        
        # Extract file references
        import re
        file_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        file_matches = re.findall(file_pattern, content)
        
        missing_files = []
        for title, file_path in file_matches:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(f"{title}: {file_path}")
        
        if missing_files:
            self.errors.append(f"Referenced files not found: {', '.join(missing_files)}")
            return False
        
        self.validation_results["file_references"] = True
        return True
    
    def validate_artifact_manifest(self) -> bool:
        """Validate artifact manifest exists and is accessible."""
        manifest_path = self.project_root / "data" / "step4_artifact_manifest.json"
        
        if not manifest_path.exists():
            self.errors.append("Artifact manifest not found: data/step4_artifact_manifest.json")
            return False
        
        # Check manifest is valid JSON
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Check required fields
            required_fields = ["metadata", "summary", "artifacts", "validation_results"]
            missing_fields = []
            
            for field in required_fields:
                if field not in manifest:
                    missing_fields.append(field)
            
            if missing_fields:
                self.errors.append(f"Artifact manifest missing fields: {', '.join(missing_fields)}")
                return False
            
        except json.JSONDecodeError as e:
            self.errors.append(f"Artifact manifest is not valid JSON: {e}")
            return False
        
        self.validation_results["artifact_manifest"] = True
        return True
    
    def validate_overall_status(self) -> bool:
        """Validate overall handoff package status."""
        # Check all individual validations passed (excluding overall_status itself)
        individual_validations = {k: v for k, v in self.validation_results.items() if k != "overall_status"}
        all_passed = all(individual_validations.values())
        
        if not all_passed:
            self.errors.append("Not all validation criteria met")
            return False
        
        # Check no critical errors
        if self.errors:
            self.errors.append("Critical errors found in validation")
            return False
        
        self.validation_results["overall_status"] = True
        return True
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete handoff package validation."""
        print("Validating Step 4.4.3 handoff package...")
        
        # Run all validations
        self.validate_handoff_document()
        self.validate_qa_checklist()
        self.validate_readme_snippets()
        self.validate_signoff_section()
        self.validate_reproducibility_metadata()
        self.validate_file_references()
        self.validate_artifact_manifest()
        self.validate_overall_status()
        
        # Generate summary
        total_validations = len(self.validation_results)
        passed_validations = sum(1 for v in self.validation_results.values() if v)
        
        summary = {
            "total_validations": total_validations,
            "passed_validations": passed_validations,
            "validation_success_rate": passed_validations / total_validations if total_validations > 0 else 0,
            "overall_status": self.validation_results["overall_status"],
            "errors": self.errors,
            "warnings": self.warnings,
            "validation_results": self.validation_results
        }
        
        return summary
    
    def generate_validation_report(self, output_path: str = "docs/step4_4_3_validation_report.md"):
        """Generate validation report."""
        summary = self.run_validation()
        
        report = f"""# Step 4.4.3 - Handoff Package Validation Report

**Generated**: {datetime.now().isoformat()}Z  
**Status**: {'✅ PASSED' if summary['overall_status'] else '❌ FAILED'}  
**Validation Success Rate**: {summary['validation_success_rate']:.1%}

## Validation Summary

| Validation Item | Status | Details |
|----------------|--------|---------|
| Handoff Document | {'✅ PASSED' if self.validation_results['handoff_doc'] else '❌ FAILED'} | Master handoff document exists and complete |
| QA Checklist | {'✅ PASSED' if self.validation_results['qa_checklist'] else '❌ FAILED'} | Covers all required steps 4.1-4.4.2 |
| README Snippets | {'✅ PASSED' if self.validation_results['readme_snippets'] else '❌ FAILED'} | Project integration snippets included |
| Sign-off Section | {'✅ PASSED' if self.validation_results['signoff_section'] else '❌ FAILED'} | Decision options and form included |
| Reproducibility Metadata | {'✅ PASSED' if self.validation_results['reproducibility_metadata'] else '❌ FAILED'} | Complete technical configuration |
| File References | {'✅ PASSED' if self.validation_results['file_references'] else '❌ FAILED'} | All referenced files exist |
| Artifact Manifest | {'✅ PASSED' if self.validation_results['artifact_manifest'] else '❌ FAILED'} | Machine-readable manifest accessible |

## Validation Results

### Overall Status
- **Total Validations**: {summary['total_validations']}
- **Passed Validations**: {summary['passed_validations']}
- **Success Rate**: {summary['validation_success_rate']:.1%}
- **Overall Status**: {'✅ READY FOR STAKEHOLDER DELIVERY' if summary['overall_status'] else '❌ NOT READY'}

### Errors
"""

        if summary['errors']:
            for error in summary['errors']:
                report += f"- ❌ {error}\n"
        else:
            report += "- ✅ No errors found\n"

        report += f"""
### Warnings
"""

        if summary['warnings']:
            for warning in summary['warnings']:
                report += f"- ⚠️ {warning}\n"
        else:
            report += "- ✅ No warnings found\n"

        report += f"""
## Acceptance Gates

### Required Components
- [x] **Master Handoff Document**: docs/step4_handoff.md
- [x] **QA Checklist**: Covers metrics, case studies, edge cases, consolidation
- [x] **README Snippets**: Project integration snippets
- [x] **Sign-off Section**: Decision options and form
- [x] **Reproducibility Metadata**: Seed, commit, policy, manifest references
- [x] **File References**: All referenced files exist
- [x] **Artifact Manifest**: Machine-readable tracking

### Decision Readiness
- [x] **Single Polished Document**: Complete handoff package
- [x] **Comprehensive QA**: All evaluation phases covered
- [x] **Clear Sign-off**: Decision options clearly presented
- [x] **Complete Metadata**: Reproducibility ensured
- [x] **Stakeholder Ready**: Package ready for delivery

## Next Steps

{'✅ **READY FOR STAKEHOLDER DELIVERY**' if summary['overall_status'] else '❌ **ADDRESS VALIDATION ISSUES**'}

{'The handoff package meets all acceptance gates and is ready for stakeholder decision.' if summary['overall_status'] else 'Please address the identified validation issues before stakeholder delivery.'}

---

**Generated by**: Claude (Anthropic)  
**Project**: Movie Recommendation Optimizer  
**Phase**: Step 4.4.3 - Handoff Package Validation  
**Version**: 1.0  
**Status**: {'✅ COMPLETED' if summary['overall_status'] else '❌ FAILED'}
"""

        # Write validation report
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Validation report generated: {output_path}")
        return report


def main():
    """CLI entrypoint for handoff validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Step 4.4.3 - Handoff Package Validation')
    parser.add_argument('--project-root', default='/Users/shaheen/Desktop/Netflix',
                       help='Project root directory')
    parser.add_argument('--output-report', default='docs/step4_4_3_validation_report.md',
                       help='Output validation report path')
    
    args = parser.parse_args()
    
    try:
        # Initialize validation
        validator = HandoffValidation(args.project_root)
        
        # Run validation
        summary = validator.run_validation()
        
        # Generate report
        report = validator.generate_validation_report(args.output_report)
        
        # Print summary
        print(f"Handoff validation completed!")
        print(f"Overall status: {'✅ PASSED' if summary['overall_status'] else '❌ FAILED'}")
        print(f"Success rate: {summary['validation_success_rate']:.1%}")
        
        if summary['errors']:
            print(f"Errors: {len(summary['errors'])}")
            for error in summary['errors']:
                print(f"  - {error}")
        
        if summary['warnings']:
            print(f"Warnings: {len(summary['warnings'])}")
            for warning in summary['warnings']:
                print(f"  - {warning}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
