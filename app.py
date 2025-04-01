from flask import Flask, render_template, request
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn’t exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Load audio file
        try:
            y, sr = librosa.load(file_path, sr=None)
        except Exception as e:
            return f'Error loading audio file: {str(e)}'
        
        # Generate waveform in memory
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr)
        plt.title("Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        waveform_buf = io.BytesIO()
        plt.savefig(waveform_buf, format='png')
        waveform_buf.seek(0)
        waveform_data = base64.b64encode(waveform_buf.getvalue()).decode('utf-8')
        plt.close()
        
        # Generate spectrogram in memory
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram")
        spectrogram_buf = io.BytesIO()
        plt.savefig(spectrogram_buf, format='png')
        spectrogram_buf.seek(0)
        spectrogram_data = base64.b64encode(spectrogram_buf.getvalue()).decode('utf-8')
        plt.close()
        
        return render_template('result.html', waveform_data=waveform_data, spectrogram_data=spectrogram_data)
    else:
        return 'Invalid file type. Please upload a WAV, MP3, OGG, or FLAC file.'

if __name__ == '__main__':
    app.run(debug=True)