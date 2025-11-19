import os
import requests

LANGFLOW_URL = ""
API_KEY = ""
FOLDER_ID = ""  # ID папки (проекта), куда загружать

# Папка с локальными json-файлами flow
FLOWS_DIR = "test_import/"


def import_flow(json_path: str):
    url = f"{LANGFLOW_URL}/api/v1/flows/upload/"
    params = {"folder_id": FOLDER_ID}
    headers = {
        "x-api-key": API_KEY
    }

    with open(json_path, "rb") as f:
        files = {
            "file": (os.path.basename(json_path), f, "application/json")
        }
        resp = requests.post(url, headers=headers, files=files)

    if resp.status_code == 201 or resp.status_code == 200:
        print(f"Успешно импортирован {json_path}: {resp.json()}")
    else:
        print(f"Ошибка при импорте {json_path}: {resp.status_code}, {resp.text}")


def import_all_flows():
    for filename in os.listdir(FLOWS_DIR):
        if filename.endswith(".json"):
            path = os.path.join(FLOWS_DIR, filename)
            import_flow(path)


if __name__ == "__main__":
    import_all_flows()
