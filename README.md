# 🦖 Webcam AR & Image Processing App

This is a Python application that lets you explore computer vision features using your webcam.  
It comes with multiple interactive modes:

- **View Mode** – plain webcam feed  
- **Geometric Transformations** – rotate, translate, scale the video  
- **Color Conversion** – switch between RGB, grayscale, HSV  
- **Brightness & Contrast** – adjust live  
- **Histogram Visualization** – see pixel distributions  
- **Gaussian & Bilateral Blur** – smoothing filters  
- **Canny Edge Detection** – detect edges  
- **Hough Line Detection** – detect straight lines  
- **Panorama Mode** – stitch frames left to right  
- **AR Mode** – Augmented Reality with ArUco markers + 3D dinosaur OBJ model 🦖  
- **Camera Calibration** – use a chessboard pattern to calibrate your webcam  
- **Undistort Mode** – correct lens distortion  

This project is great for learning **OpenCV** and experimenting with **Augmented Reality**.

---

## 🔧 Requirements

- **Python** 3.7, 3.8, or 3.9  
  ⚠️ Do **not** use Python 3.10 or higher — OpenCV and some libraries may break.  
- A **working webcam**  
- A **terminal / command prompt** (Windows, macOS, or Linux)

---

## 📥 Installation (Step by Step for Beginners)

### 1. Install Python
- Go to [Python Downloads](https://www.python.org/downloads/).  
- Download **Python 3.9.x** for your system.  
- On Windows, during installation, check:  
  ✅ **Add Python to PATH**  

### Check installation: <In your terminal>
```bash
python --version
```

You should see something like:

```bash
Python 3.9.13
```
### 2. Download the Project
If you received a folder: put it on your Desktop (or anywhere easy).

If using GitHub: <In your terminal>

```bash

git clone https://github.com/Rakshya8/CV_First_Assignment.git
cd <your-repo-folder>
```

### 3. Open a Terminal in the Project Folder
Windows: Shift + Right Click inside folder → Open PowerShell/terminal here

macOS/Linux:

```bash
cd path/to/your/project
```

### 4. Create a Virtual Environment (Recommended) <In your project>
This keeps libraries separate for this project.

```bash
python -m venv venv
```
Activate it:

Windows (PowerShell):

```bash
venv\Scripts\activate
```
macOS/Linux:

```bash
source venv/bin/activate
```
You should now see (venv) in your terminal.

### 5. Install Requirements
Upgrade pip and install dependencies: <In your terminal>

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
This installs:

1. opencv-python → computer vision

2. numpy → math operations

3. and other required packages

### 6. Run the Application <In your terminal>
```bash
python app.py
```

Note: Your webcam will open in fullscreen. 🎥

#### 📸 Assets Provided
Inside the project there is an assets folder:

***Calibration Chessboard*** → assets/calibrate.jpg

1. Used for camera calibration (press K).

2. Print and hold in front of your webcam.

***ArUco Marker → assets/ar.jpg***

1. Used for AR mode (press A).

2. Print and place in front of your webcam — a 🦖 dinosaur will appear!

🕹 Controls
Key	Function
ESC	Quit
f	Toggle fullscreen
v	View mode
g	Geometric transforms
c	Color modes (RGB/Gray/HSV)
b	Brightness & Contrast
h	Histogram
i	Bilateral Filter
e	Canny Edge Detection
t	Hough Line Detection
y	Panorama mode
a	AR mode (3D dinosaur)
k	Calibration (chessboard)
u	Undistort preview

Geometric Mode
m → toggle rigid vs similarity

SPACE → auto-rotate

a / d → rotate left / right

0 → reset

Panorama Mode
n → add frame

r → reset panorama

s → save panorama

🦖 Augmented Reality (AR Mode)
Print the ArUco marker (assets/ar.jpg).

Place it in front of your webcam.

Press A → AR mode starts.

A green dinosaur model appears on top of the marker!

🎯 Tips for Beginners
Don’t be scared of the terminal! You’ll only type a few commands.

If you get errors, copy them and search online (lots of OpenCV tutorials exist).

Experiment with each mode to learn computer vision concepts.

✅ Summary of Commands
bash
Copy code
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
✨ You’re now ready to explore computer vision and augmented reality with your webcam! 🦖🎥