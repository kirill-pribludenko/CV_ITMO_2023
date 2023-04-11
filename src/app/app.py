import asyncio
import os
import shutil
import subprocess
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

app = FastAPI()


# Функция для выполнения предсказаний


@app.get("/")
async def serve_index():
    return FileResponse("./src/app/ing.html")


# Путь до директории, где будут сохраняться изображения
STATIC_FILES_PATH = "./models/inference/example_imgs"


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    # Удаляем все файлы в указанной папке
    for filename in os.listdir(STATIC_FILES_PATH):
        file_path = os.path.join(STATIC_FILES_PATH, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(e)

    # сохраняем загруженный файл в указанной папке
    file_location = os.path.join(STATIC_FILES_PATH, file.filename)
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    # запускаем скрипт с предсказаниями в отдельном процессе
    cmd = (
        "python ./src/models/predict_torchgeo.py --weights ./models/test_torchgeo.pt "
        f"--source {STATIC_FILES_PATH} --img-size 256"
    )

    subprocess.Popen(cmd.split())

    # Возвращаем исходное изображение и его результаты
    img_path = os.path.join(STATIC_FILES_PATH, file.filename)
    predictions_path = os.path.join(
        STATIC_FILES_PATH, f"predictions_{os.path.splitext(file.filename)[0]}.jpg"
    )
    while not os.path.exists(predictions_path):
        await asyncio.sleep(1)

    return {
        "original_image": FileResponse(img_path),
        "predictions": FileResponse(predictions_path),
    }


@app.get("/download/")
async def download_file():
    folder_path = "./models/inference/results/"
    folder_files = os.listdir(folder_path)
    folder_files = [
        os.path.join(folder_path, file) for file in folder_files
    ]
    folder_files.sort(key=os.path.getctime)
    latest_file = folder_files[-1]
    return FileResponse(latest_file)

