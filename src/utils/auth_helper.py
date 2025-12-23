"""Helper utilities for GCP authentication verification."""

import os
from pathlib import Path
from typing import Optional


def get_default_service_account_key() -> Optional[Path]:
    """
    Check for default service account key in project root.
    Looks for zesty-*.json pattern.
    
    Returns:
        Path to key file if found, None otherwise
    """
    project_root = Path(__file__).resolve().parents[2]  # Go up from src/utils/ to project root
    # Look for zesty-*.json pattern
    for key_file in project_root.glob("zesty-*.json"):
        if key_file.is_file() and key_file.name.endswith(".json"):
            return key_file
    return None


def verify_gcp_auth(project_id: Optional[str] = None) -> tuple[bool, str]:
    """
    Verify that GCP authentication is properly configured.
    
    Automatically uses default service account key (zesty-*.json) in project root
    if GOOGLE_APPLICATION_CREDENTIALS is not set.
    
    Args:
        project_id: Optional project ID to verify against.
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if authentication is properly configured
        - error_message: Empty string if valid, otherwise error description
    """
    try:
        from google.cloud import bigquery
        from google.auth import default
        from google.auth.exceptions import DefaultCredentialsError
    except ImportError:
        return False, "google-cloud-bigquery is not installed. Run: pip install google-cloud-bigquery"
    
    # Auto-detect default service account key if not set
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        default_key = get_default_service_account_key()
        if default_key:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(default_key)
            print(f"✓ Using default service account key: {default_key.name}")
    
    # Check if credentials are available
    try:
        credentials, detected_project = default()
        
        # If project_id is provided, verify it matches
        if project_id and detected_project and detected_project != project_id:
            return False, (
                f"Project ID mismatch: credentials are for '{detected_project}' "
                f"but GOOGLE_CLOUD_PROJECT is '{project_id}'. "
                "Ensure your service account key or gcloud auth matches the project ID."
            )
        
        # Try to create a BigQuery client to verify permissions
        test_project = project_id or detected_project
        if not test_project:
            return False, (
                "No project ID detected. Set GOOGLE_CLOUD_PROJECT in .env file or "
                "ensure your credentials include a project ID."
            )
        
        client = bigquery.Client(project=test_project)
        # Try a simple operation to verify permissions
        list(client.list_datasets(max_results=1))
        
        return True, ""
        
    except DefaultCredentialsError:
        # Check if GOOGLE_APPLICATION_CREDENTIALS is set but file doesn't exist
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path:
            if not os.path.exists(creds_path):
                return False, (
                    f"Service account key file not found: {creds_path}\n"
                    "Please ensure the file exists and the path is correct."
                )
            else:
                return False, (
                    f"Failed to load credentials from {creds_path}.\n"
                    "The file may be corrupted or invalid. Please regenerate the service account key."
                )
        else:
            return False, (
                "GCP authentication not configured. Choose one of the following:\n"
                "1. Set GOOGLE_APPLICATION_CREDENTIALS to a service account key file path\n"
                "2. Run: gcloud auth application-default login\n"
                "3. Use a compute engine service account (if running on GCP)"
            )
    except Exception as e:
        error_msg = str(e)
        if "PERMISSION_DENIED" in error_msg or "403" in error_msg:
            return False, (
                "Permission denied. Your service account needs:\n"
                "- roles/bigquery.admin\n"
                "- roles/aiplatform.user\n"
                "Please grant these roles to your service account."
            )
        return False, f"Authentication error: {error_msg}"


def print_auth_setup_instructions() -> None:
    """Print helpful instructions for setting up GCP authentication."""
    print("\n" + "="*70)
    print("GCP Authentication Setup Instructions")
    print("="*70)
    print("\nOption 1: Service Account Key (Recommended for reviewers)")
    print("-" * 70)
    print("1. Create a service account in your GCP project")
    print("2. Grant roles: roles/bigquery.admin and roles/aiplatform.user")
    print("3. Download the service account key JSON file")
    print("4. Set environment variable:")
    print("   export GOOGLE_APPLICATION_CREDENTIALS='path/to/key.json'")
    print("\nOption 2: gcloud CLI (For local development)")
    print("-" * 70)
    print("Run: gcloud auth application-default login")
    print("\nFor detailed instructions, see README.md")
    print("="*70 + "\n")

