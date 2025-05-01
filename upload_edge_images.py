import os
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, TransferOptions

# Authenticate via the VM's Managed Identity
service = BlobServiceClient(
    account_url="https://<your-storage-account>.blob.core.windows.net",
    credential=DefaultAzureCredential()
)

# Get a client to the 'edge-images' container
container = service.get_container_client("edge-images")

# STEP 1: Fetch existing blobs to support resume logic
print("Retrieving list of existing blobs...")
existing_blobs = set()
for blob in container.list_blobs():         # List every blob
    existing_blobs.add(blob.name)           # Track its name
print(f"Found {len(existing_blobs)} existing blobs—will skip these.")

# STEP 2: Configure parallel, block-based transfer
opts = TransferOptions(
    max_concurrency=8,          # Number of parallel upload threads
    max_block_size=8 * 1024**2  # 8 MiB blocks
)

# STEP 3: Walk local edge folder
edge_folder = "/mnt/edge_storage/images/"
for root, _, files in os.walk(edge_folder):
    for fname in files:
        # Only consider image files
        if not fname.lower().endswith((".jpg", ".png", ".bmp")):
            continue

        # Build local path and blob name
        local_path = os.path.join(root, fname)
        blob_name  = os.path.relpath(local_path, edge_folder).replace("\\", "/")

        # STEP 4: Skip if already uploaded
        if blob_name in existing_blobs:
            print(f"Skipping already-uploaded: {blob_name}")
            continue

        # STEP 5: Upload in parallel
        print(f"Uploading: {blob_name}")
        with open(local_path, "rb") as data:
            container.upload_blob(
                name=blob_name,
                data=data,
                overwrite=False,
                transfer_options=opts
            )

print("Upload complete.")
