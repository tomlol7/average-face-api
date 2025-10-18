from flask import Flask, request, send_file, jsonify
import tempfile, os
from facer import facer
import matplotlib.pyplot as plt
import os
import urllib.request
import bz2

model_dir = "./model"
model_file = "shape_predictor_68_face_landmarks.dat"
model_path = os.path.join(model_dir, model_file)
model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

# Create model folder if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# Download and decompress if the model does not exist
if not os.path.exists(model_path):
    print("Downloading dlib model (this may take a while)...")
    bz2_path = model_path + ".bz2"
    urllib.request.urlretrieve(model_url, bz2_path)
    print("Download complete. Decompressing...")
    with bz2.open(bz2_path, "rb") as f_in:
        with open(model_path, "wb") as f_out:
            f_out.write(f_in.read())
    os.remove(bz2_path)
    print("Model ready!")

# Set environment variable for Facer
os.environ["FACER_PREDICTOR_PATH"] = model_path

app = Flask(__name__)

@app.route("/average", methods=["POST"])
def average_faces():
    # Get uploaded files
    files = request.files.getlist("images")

    # Limit: up to 50 faces
    if len(files) == 0:
        return jsonify({"error": "No images uploaded"}), 400
    if len(files) > 50:
        return jsonify({"error": "You can upload up to 50 images only"}), 400

    # Save uploaded images temporarily
    temp_dir = tempfile.mkdtemp()
    for f in files:
        f.save(os.path.join(temp_dir, f.filename))

    # Run facer averaging
    try:
        images = facer.load_images(temp_dir)
        landmarks, faces = facer.detect_face_landmarks(images)
        average_face = facer.create_average_face(faces, landmarks, save_image=True)
        
        # Save composite to temp file
        output_path = os.path.join(temp_dir, "average_face.jpg")
        plt.imsave(output_path, average_face)

        return send_file(output_path, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp files after request ends
        for file in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, file))
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass

@app.route("/")
def home():
    return "Average Facer API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
