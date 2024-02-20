from flask import Flask, request, jsonify,send_file
from PIL import Image
from io import BytesIO
from cnn import CNNET
app = Flask(__name__)


@app.route("/img_cnn", methods=["POST"])
def process_image():
    try:
        file = request.files['image']

        # Check if the file is an image
        if file and allowed_file(file.filename):
            # Read the image via file.stream
            img = Image.open(BytesIO(file.stream.read()))
            processed_image_path = CNNET.cnet_con(img)

            # Return the processed image file
            return send_file(processed_image_path, mimetype='image/png', as_attachment=True)
        else:
            return jsonify({'error': 'Invalid file format'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    # Add any additional file format checks if needed
    
   return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif'}

if __name__ == "__main__":
    app.run(debug=True)
