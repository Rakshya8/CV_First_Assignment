🦖 Webcam AR & Image Processing App

This is a Python application that lets you explore computer vision features using your webcam. It comes with multiple modes such as:

View Mode – plain webcam feed

Geometric Transformations – rotate, translate, scale the video

Color Conversion – switch between RGB, grayscale, HSV

Brightness & Contrast – adjust live

Histogram Visualization – see pixel distributions

Gaussian & Bilateral Blur – smoothing filters

Canny Edge Detection – detect edges

Hough Line Detection – detect straight lines

Panorama Mode – stitch frames left to right

AR Mode – Augmented Reality with ArUco markers + 3D dinosaur OBJ model 🦖

Camera Calibration – use a chessboard pattern to calibrate your webcam

Undistort Mode – correct lens distortion

This is great for learning OpenCV (a computer vision library) and experimenting with augmented reality.

🔧 Requirements

Python 3.7, 3.8, or 3.9
⚠️ Do not use Python 3.10 or higher, as OpenCV and some libraries may break on certain systems.

A working webcam

A terminal / command prompt (built into Windows, macOS, Linux)

📥 Installation (Step by Step for Beginners)
1. Install Python

Go to Python Downloads

Choose Python 3.9.x for your system (Windows or macOS).

During installation on Windows, check the box that says:

"Add Python to PATH"

After installation, open a terminal and check:

python --version


It should say something like:

Python 3.9.13

2. Download the Project

If you received this project as a folder:

Place it somewhere easy to find (e.g., Desktop).

If you are using GitHub:

git clone <your-repo-url>
cd <your-repo-folder>

3. Open a Terminal in the Project Folder

On Windows: Shift + Right Click inside the folder → "Open PowerShell window here"

On macOS/Linux: Open Terminal, then type:

cd path/to/your/project

4. Create a Virtual Environment (Recommended)

This keeps project libraries separate.

python -m venv venv


Activate it:

Windows (PowerShell):

venv\Scripts\activate


macOS/Linux:

source venv/bin/activate


You should now see (venv) at the start of your terminal line.

5. Install Requirements

Run:

pip install --upgrade pip
pip install -r requirements.txt


This will install:

opencv-python → for computer vision

numpy → for math operations

And other needed libraries

6. Run the Application

Now start the app:

python app.py


Your webcam window will open in fullscreen. 🎥

📸 Assets Provided

Inside the project there is an assets folder with useful images:

Calibration Chessboard → assets/calibrate.jpg

Used for camera calibration (press K in the app).

Print this out and hold it in front of your webcam so the program can capture calibration images.

ArUco Marker → assets/ar.jpg

Used for Augmented Reality mode (press A in the app).

Print this out and place it in front of your webcam — the dinosaur 🦖 will appear on top of the marker.

🕹 Controls

ESC → quit

f → toggle fullscreen

v → View mode (plain webcam)

g → Geometric transforms (rotate/translate/scale)

c → Color modes (RGB, Gray, HSV)

b → Brightness & Contrast

h → Histogram

i → Bilateral Filter

e → Canny Edge Detection

t → Hough Line Detection

y → Panorama mode

a → AR mode (3D dinosaur 🦖 on ArUco marker)

k → Calibration (with chessboard)

u → Undistort preview

In Geometric mode:

m → toggle rigid vs. similarity

SPACE → auto-rotate

a / d → rotate left/right

0 → reset

In Panorama mode:

n → add frame

r → reset panorama

s → save panorama

🦖 Augmented Reality (AR Mode)

Print the ArUco marker from assets/ar.jpg.

Place it in front of your webcam.

Press a to enter AR mode.

The green dinosaur OBJ model will appear sitting on the marker.

🎯 Tips for Beginners

Don’t be scared of the terminal! You’ll only type a few commands.

If something doesn’t work, copy the error message and search it online (or ask your teacher/mentor).

This app is designed for learning computer vision step by step. Try each mode!

✅ Summary of Commands
# 1. Check Python version
python --version

# 2. Navigate to project
cd path/to/project

# 3. Create virtual environment
python -m venv venv

# 4. Activate it
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 5. Install requirements
pip install -r requirements.txt

# 6. Run the app
python app.py


✨ Now you’re ready to explore computer vision with your webcam!