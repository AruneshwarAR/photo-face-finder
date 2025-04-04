# Personal Photo Face Finder GUI

A simple Python application with a GUI (built using Tkinter) to scan photo collections, find pictures containing a specific person using facial recognition, and display/export the results. Built to help organize large, distributed photo libraries.

## Problem Solved

Managing thousands of photos spread across different drives makes it hard to find pictures of specific people. This tool provides an efficient way to index faces once and quickly search for matches using a reference photo, providing the file locations of the matches.

## Features

- **Indexing:** Scans specified directories for images, detects faces, and stores efficient face embeddings in an SQLite database. Only indexes new or modified files on subsequent runs.
- **Searching:** Compares a face from a reference image against the indexed faces.
- **GUI:** Simple Tkinter interface for selecting directories/files, running indexing/searching, and viewing results.
- **Preview:** Displays thumbnail previews of found images.
- **Export:** Allows copying the found image files to a new folder.
- Supports common image formats (JPG, PNG, HEIC/HEIF if `pillow-heif` is installed).
- Uses the `face_recognition` library.

## Screenshots

_(Insert 1-2 good screenshots of your GUI here. You can upload them to the GitHub repo later and link them)_
![GUI Screenshot 1](link/to/your/screenshot1.png)
![GUI Screenshot 2](link/to/your/screenshot2.png)

## Setup & Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AruneshwarAR/photo-face-finder.git
   cd photo-face-finder
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   # OR manually:
   # pip3 install face_recognition Pillow numpy pillow-heif
   ```
   _(Note: Installing `dlib` (a dependency of `face_recognition`) might require `cmake`: `brew install cmake`)_

## Usage

1. **Run the GUI:**

   ```bash
   python3 face_finder_gui.py
   ```

2. **Index Photos:**

   - Use the "Select Dir..." button in the "Indexing" section to choose a top-level folder containing photos.
   - Click "Start Indexing". The first run on a large collection can take a long time. Status will be shown in the log. An index database (`photo_face_index.db` by default) will be created/updated.

3. **Search for a Person:**

   - Use the "Select Image..." button in the "Searching" section to choose a clear photo of the person you want to find.
   - Adjust the "Tolerance" if needed (lower is stricter, 0.6 is default).
   - Click "Search for Person".

4. **View & Export Results:**
   - Matching photos will appear as thumbnails in the bottom section.
   - Click "Export Found Images..." to copy the original files to a folder of your choice.

## Disclaimer

**This is a personal project developed in my own time and is not affiliated with, endorsed by, or related to my employer in any way.**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
