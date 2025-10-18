from flask import Flask, request, send_file, jsonify
import tempfile, os
from facer import facer
import matplotlib.pyplot as plt

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
