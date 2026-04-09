import os
from google.cloud import storage

BUCKET_NAME = "vm-football-data"
LOCAL_DIR = "datasets/test_seq_116"
GCS_DEST_DIR = "datasets/test_seq_116"

def upload_folder_to_gcs():
    print(f"Subiendo {LOCAL_DIR} a gs://{BUCKET_NAME}/{GCS_DEST_DIR} ...")
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    count = 0
    for root, dirs, files in os.walk(LOCAL_DIR):
        for file in files:
            local_path = os.path.join(root, file)
            # Create relative path for GCS
            relative_path = os.path.relpath(local_path, LOCAL_DIR)
            gcs_path = f"{GCS_DEST_DIR}/{relative_path}".replace("\\", "/")
            
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            count += 1
            if count % 100 == 0:
                print(f"  Subidos {count} archivos...")

    print(f"¡Exito! Se subieron {count} archivos a gs://{BUCKET_NAME}/{GCS_DEST_DIR}/")

if __name__ == "__main__":
    upload_folder_to_gcs()
