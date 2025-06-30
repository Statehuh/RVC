import re
import os
import sys
import shutil

sys.path.append(os.getcwd())

from modules.utils import HF_download_file
from modules import gdown, meganz, mediafire, pixeldrain

def move_files_from_directory(src_dir, dest_models, model_name):
    for root, _, files in os.walk(src_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".index"):
                filepath = os.path.join(dest_models, file.replace(' ', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(",", "").replace('"', "").replace("'", "").replace("|", "").strip())

                shutil.move(file_path, filepath)
            elif file.endswith(".pth") and not file.startswith("D_") and not file.startswith("G_"):
                pth_path = os.path.join(dest_models, model_name + ".pth")

                shutil.move(file_path, pth_path)

def save_drop_model(dropbox):
    model_folders = "rvc_models" 
    save_model_temp = "save_model_temp"

    if not os.path.exists(model_folders): os.makedirs(model_folders, exist_ok=True)
    if not os.path.exists(save_model_temp): os.makedirs(save_model_temp, exist_ok=True)

    shutil.move(dropbox, save_model_temp)

    try:
        print("[INFO] Start uploading...")

        file_name = os.path.basename(dropbox)
        model_folders = os.path.join(model_folders, file_name.replace(".zip", "").replace(".pth", "").replace(".index", ""))

        if file_name.endswith(".zip"):
            shutil.unpack_archive(os.path.join(save_model_temp, file_name), save_model_temp)
            move_files_from_directory(save_model_temp, model_folders, file_name.replace(".zip", ""))
        elif file_name.endswith(".pth"): 
            output_file = os.path.join(model_folders, file_name)
            shutil.move(os.path.join(save_model_temp, file_name), output_file)
        elif file_name.endswith(".index"):
            def extract_name_model(filename):
                match = re.search(r"([A-Za-z]+)(?=_v|\.|$)", filename)
                return match.group(1) if match else None
            
            model_logs = os.path.join(model_folders, extract_name_model(file_name))
            if not os.path.exists(model_logs): os.makedirs(model_logs, exist_ok=True)
            shutil.move(os.path.join(save_model_temp, file_name), model_logs)
        else: 
            print("[WARNING] Format not supported. Supported formats ('.zip', '.pth', '.index')")
            return
        
        print("[INFO] Completed upload.")
    except Exception as e:
        print(f"[ERROR] An error occurred during unpack: {e}")
    finally:
        shutil.rmtree(save_model_temp, ignore_errors=True)

def download_model(url=None, model=None):
    if not url: 
        print("[WARNING] Please provide a valid url.")
        return

    if not model: 
        print("[WARNING] Please provide a valid model name.")
        return

    model = model.replace(".pth", "").replace(".index", "").replace(".zip", "").replace(" ", "_").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(",", "").replace('"', "").replace("'", "").replace("|", "").strip()
    url = url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()

    download_dir = "download_model"
    model_folders = "rvc_models" 

    if not os.path.exists(download_dir): os.makedirs(download_dir, exist_ok=True)
    if not os.path.exists(model_folders): os.makedirs(model_folders, exist_ok=True)

    model_folders = os.path.join(model_folders, model)
    os.makedirs(model_folders, exist_ok=True)
    
    try:
        print("[INFO] Start downloading...")

        if url.endswith(".pth"): HF_download_file(url, os.path.join(model_folders, f"{model}.pth"))
        elif url.endswith(".index"): HF_download_file(url, os.path.join(model_folders, f"{model}.index"))
        elif url.endswith(".zip"):
            output_path = HF_download_file(url, os.path.join(download_dir, model + ".zip"))
            shutil.unpack_archive(output_path, download_dir)

            move_files_from_directory(download_dir, model_folders, model)
        else:
            if "drive.google.com" in url or "drive.usercontent.google.com" in url:
                file_id = None

                if "/file/d/" in url: file_id = url.split("/d/")[1].split("/")[0]
                elif "open?id=" in url: file_id = url.split("open?id=")[1].split("/")[0]
                elif "/download?id=" in url: file_id = url.split("/download?id=")[1].split("&")[0]
                
                if file_id:
                    file = gdown.gdown_download(id=file_id, output=download_dir)
                    if file.endswith(".zip"): shutil.unpack_archive(file, download_dir)

                    move_files_from_directory(download_dir, model_folders, model)
            elif "mega.nz" in url:
                meganz.mega_download_url(url, download_dir)

                file_download = next((f for f in os.listdir(download_dir)), None)
                if file_download.endswith(".zip"): shutil.unpack_archive(os.path.join(download_dir, file_download), download_dir)

                move_files_from_directory(download_dir, model_folders, model)
            elif "mediafire.com" in url:
                file = mediafire.Mediafire_Download(url, download_dir)
                if file.endswith(".zip"): shutil.unpack_archive(file, download_dir)

                move_files_from_directory(download_dir, model_folders, model)
            elif "pixeldrain.com" in url:
                file = pixeldrain.pixeldrain(url, download_dir)
                if file.endswith(".zip"): shutil.unpack_archive(file, download_dir)

                move_files_from_directory(download_dir, model_folders, model)
            else:
                print("[WARNING] The url path is not supported.")
                return
        
        print("[INFO] Model download complete.")
    except Exception as e:
        print(f"[INFO] An error has occurred: {e}")
    finally:
        shutil.rmtree(download_dir, ignore_errors=True)