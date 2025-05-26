import os
from google.cloud import storage

def test_gcs_authentication_and_list_buckets(key_file_path=None):
    """
    Tests GCS authentication using either GOOGLE_APPLICATION_CREDENTIALS
    or Application Default Credentials (if key_file_path is None for ADC test).
    Tries to list buckets in the project associated with the credentials.
    """
    print("-" * 30)
    if key_file_path:
        print(f"Attempting authentication using explicit key file: {key_file_path}")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_file_path
    else:
        # Unset it to ensure we test ADC if GOOGLE_APPLICATION_CREDENTIALS was set before
        if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            print(f"Temporarily unsetting GOOGLE_APPLICATION_CREDENTIALS ('{os.environ['GOOGLE_APPLICATION_CREDENTIALS']}') to test ADC.")
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        print("Attempting authentication using Application Default Credentials (ADC).")
        print("Ensure you have run 'gcloud auth activate-service-account --key-file=YOUR_KEY.json' or 'gcloud auth application-default login'.")

    try:
        # Explicitly create a client. If GOOGLE_APPLICATION_CREDENTIALS is set, it uses that.
        # Otherwise, it falls back to ADC.
        storage_client = storage.Client()
        
        # Attempt to list buckets - a simple authenticated API call.
        # This will use the project ID associated with the credentials.
        print(f"Successfully authenticated. Client project: {storage_client.project}")
        print("Listing buckets (up to 5):")
        buckets = list(storage_client.list_buckets(max_results=5))
        if buckets:
            for bucket in buckets:
                print(f"- {bucket.name}")
        else:
            print("No buckets found or no permission to list buckets for this project.")
        
        # Try to access your specific DVC bucket (checks listObjects permission at bucket level)
        dvc_bucket_name = "minedvcstore-1" # Your DVC bucket
        print(f"\nAttempting to access DVC bucket: {dvc_bucket_name}")
        bucket = storage_client.bucket(dvc_bucket_name)
        # Try to list first 5 blobs (objects) in the root of the bucket.
        # This requires storage.objects.list on that bucket.
        print(f"Listing objects in '{dvc_bucket_name}' (up to 5):")
        blobs = list(bucket.list_blobs(max_results=5)) 
        if blobs:
            for blob in blobs:
                print(f"  - {blob.name}")
        else:
            print(f"No objects found in the root of '{dvc_bucket_name}' or no permission to list.")

    except Exception as e:
        print(f"Authentication or GCS API call FAILED.")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up env var if we set it, to not interfere with other tests
        if key_file_path and "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    print("-" * 30)

if __name__ == "__main__":
    # --- Test Case 1: Using GOOGLE_APPLICATION_CREDENTIALS environment variable ---
    key_file = "gcp-creds3.json" # Make sure this file is in the same directory or provide full path
    print(f"\n=== TEST 1: Using GOOGLE_APPLICATION_CREDENTIALS pointing to '{key_file}' ===")
    if not os.path.exists(key_file):
        print(f"ERROR: Key file '{key_file}' not found. Skipping Test 1.")
    else:
        test_gcs_authentication_and_list_buckets(key_file_path=os.path.abspath(key_file))

    # --- Test Case 2: Using Application Default Credentials (ADC) ---
    # This assumes you have already run `gcloud auth activate-service-account --key-file=gcp-creds3.json`
    # which should set up ADC.
    print(f"\n=== TEST 2: Using Application Default Credentials (ADC) ===")
    print("Ensure 'gcloud auth activate-service-account --key-file=gcp-creds3.json' was run previously.")
    test_gcs_authentication_and_list_buckets()