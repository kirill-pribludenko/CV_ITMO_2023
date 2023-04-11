import io
import os
import warnings

import numpy as np
import onnxruntime as ort
import rasterio as rio
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
from torchgeo.transforms import indices
from torchvision.utils import draw_segmentation_masks

warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)

app = FastAPI()

# Load the ONNX model
sess = ort.InferenceSession("./models/test_torchgeo.onnx")

# Define the HTML template for the main page
html_template = """
<!DOCTYPE html>
<html>
    <head>
        <title>Heracleum Segmentation</title>
    </head>
    <body>
        <h1>Heracleum Segmentation</h1>
        <form action="/predict/" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <button type="submit">Predict</button>
        </form>
    </body>
</html>
"""


# Define the endpoint to serve the main page
@app.get("/")
async def main():
    return HTMLResponse(content=html_template, status_code=200)


tfms = torch.nn.Sequential(indices.AppendNDVI(index_nir=3, index_red=0))


# Define the endpoint to handle file uploads
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file from the request
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Prepare image
    img_a = np.array(image).astype(np.float32)
    # from 0 - 255 to 0.0 - 1.0
    img_a /= 255.0
    img_t = torch.from_numpy(img_a.transpose((2, 0, 1)))
    # add 1 dimension
    img_t = img_t[None, :, :, :]
    img_t = tfms(img_t)
    img_f = img_t.numpy()

    # Run the image through the model
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred = sess.run([output_name], {input_name: img_f})[0]

    # Convert the predicted result to an image
    pred_t = torch.from_numpy(pred)
    normalized_masks = torch.nn.functional.softmax(pred_t, dim=1)
    class_to_idx = {"no_heracleum": 0, "heracleum": 1}
    boolean_masks = normalized_masks.argmax(1) == class_to_idx["heracleum"]

    img_t = (255 * img_t[0, :3]).type(torch.uint8)
    img_and_pred = draw_segmentation_masks(
        img_t, masks=boolean_masks[0], alpha=0.6, colors="red"
    )

    # Save the predicted image to a temporary file
    temp_file = rio.open(
        "./models/inference/temp_imgs/predict.png",
        "w",
        width=256,
        height=256,
        count=3,
        dtype=np.uint8,
    )
    temp_file.write(img_and_pred)
    temp_file.close()

    # Find last file
    folder_path = "./models/inference/temp_imgs/"
    folder_files = os.listdir(folder_path)
    folder_files = [os.path.join(folder_path, file) for file in folder_files]
    folder_files.sort(key=os.path.getctime)
    latest_file = folder_files[-1]

    # Return the predicted image as a file response
    return FileResponse(latest_file, media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
