#!/usr/bin/env python3

import face_recognition
import os
import sqlite3
import argparse
import time
import sys
from pathlib import Path
from PIL import Image, UnidentifiedImageError # Pillow for better error handling
import numpy as np
import multiprocessing
import queue # Added for GUI communication

# --- Optional HEIC/HEIF Support ---
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print("HEIC/HEIF support enabled.")
    HEIC_SUPPORT = True
except ImportError:
    print("Warning: HEIC/HEIF support not available. Run 'pip3 install pillow-heif' to add it.")
    HEIC_SUPPORT = False

# --- Configuration ---
# Define supported image file extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
if HEIC_SUPPORT:
    SUPPORTED_EXTENSIONS.update({'.heic', '.heif'})

# Default database file name
DEFAULT_DB_FILE = "photo_face_index.db"

# Face detection model ('hog' is faster, 'cnn' is more accurate but GPU helps)
# Start with 'hog' for broader compatibility and speed on CPU.
DETECTION_MODEL = "hog"
# Face encoding model ('small' or 'large' - large is default/more accurate)
ENCODING_MODEL = "large"
# Number of times to re-sample the face when calculating encoding (higher is more accurate but slower)
NUM_JITTERS = 1

# --- Database Functions ---

def setup_database(db_file):
    """Initializes the SQLite database and creates the necessary table."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        # Storing one row per detected face
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                last_modified REAL NOT NULL,
                face_encoding BLOB NOT NULL,
                face_location_css TEXT  -- Store face bounding box 'top, right, bottom, left' as string
            )
        ''')
        # Indexes for faster searching and cleanup
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_path ON faces (file_path)')
        conn.commit()
        print(f"Database '{db_file}' setup complete.")
    except sqlite3.Error as e:
        print(f"Database error during setup: {e}", file=sys.stderr)
        sys.exit(1) # Exit if DB setup fails
    finally:
        if conn:
            conn.close()

def np_array_to_blob(arr):
    """Converts a NumPy array to SQLite-compatible BLOB."""
    return arr.tobytes()

def get_indexed_files(db_file):
    """Returns a dictionary of {Path(file_path): last_modified_time} for indexed files."""
    indexed = {}
    if not Path(db_file).exists():
        return indexed

    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        # Get the latest modification time stored for each unique file path
        cursor.execute("SELECT file_path, MAX(last_modified) FROM faces GROUP BY file_path")
        rows = cursor.fetchall()
        # Use Path objects as keys for easier comparison later
        indexed = {Path(row[0]): row[1] for row in rows}
    except sqlite3.Error as e:
        print(f"Database error reading indexed files: {e}", file=sys.stderr)
        # Don't exit, maybe the DB is partially readable or empty
    finally:
        if conn:
            conn.close()
    return indexed

# --- File System Functions ---

def find_image_files(directories):
    """Recursively yields Path objects for image files within the given directories."""
    file_count = 0
    for directory in directories:
        abs_directory = Path(directory).resolve() # Use absolute paths
        if not abs_directory.is_dir():
            print(f"Warning: '{directory}' is not a valid directory. Skipping.", file=sys.stderr)
            continue

        print(f"Scanning directory: {abs_directory}...")
        for root, _, files in os.walk(abs_directory):
            for filename in files:
                # Check extension using lower case
                if Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield Path(root) / filename
                    file_count += 1
    print(f"Found {file_count} potential image files across all specified directories.")

# --- Image Processing Function (for worker processes) ---

def process_image(image_path):
    """
    Loads an image, detects faces, computes encodings.
    Returns a tuple: (image_path, list_of_face_data or None).
    Each face_data is: (last_modified_time, encoding_blob, location_str)
    Returns None in list position on error or no faces found.
    """
    image_path_str = str(image_path) # Keep string version for potential logging
    try:
        # Get mod time before loading, in case loading is slow
        last_modified = image_path.stat().st_mtime

        # Load the image file using face_recognition's loader (handles orientation)
        # It uses Pillow internally.
        image_array = face_recognition.load_image_file(image_path)

        # --- Face Detection ---
        # Use the configured model
        face_locations = face_recognition.face_locations(image_array, model=DETECTION_MODEL)

        if not face_locations:
            # No faces found is not an error, just return path and None for data
            # print(f"  No faces found: {image_path.name}")
            return image_path, None

        # --- Face Encoding ---
        # Compute face encodings for the found locations
        face_encodings = face_recognition.face_encodings(
            image_array,
            face_locations,
            num_jitters=NUM_JITTERS,
            model=ENCODING_MODEL
        )

        if not face_encodings:
             # This is unexpected if locations were found, treat as potential issue
             print(f"Warning: Could not generate encodings for {image_path_str} despite finding locations.", file=sys.stderr)
             return image_path, None # Indicate issue by returning None data

        # Prepare results for this image
        results = []
        for encoding, location in zip(face_encodings, face_locations):
            encoding_blob = np_array_to_blob(encoding)
            location_str = f"{location[0]},{location[1]},{location[2]},{location[3]}" # top, right, bottom, left
            results.append((last_modified, encoding_blob, location_str))

        # print(f"  Found {len(results)} faces: {image_path.name}")
        return image_path, results

    except FileNotFoundError:
        # This might happen if file deleted between find and process
        print(f"Warning: File not found during processing: {image_path_str}", file=sys.stderr)
        return image_path, None # Indicate error
    except UnidentifiedImageError:
        print(f"Warning: Could not load image (unsupported/corrupt?): {image_path_str}", file=sys.stderr)
        return image_path, None # Indicate error
    except Exception as e:
        print(f"Error processing {image_path_str}: {type(e).__name__} - {e}", file=sys.stderr)
        # import traceback # Uncomment for detailed debugging if needed
        # traceback.print_exc()
        return image_path, None # Indicate error

# --- Function to be called by GUI ---
def run_indexing(directories_to_scan, db_file, force_rescan, detection_model, num_workers, status_queue):
    """
    Runs the indexing process and puts status updates into the queue.
    Args:
        directories_to_scan (list): List of directory paths (str or Path).
        db_file (str or Path): Path to the database file.
        force_rescan (bool): Whether to force rescan.
        detection_model (str): 'hog' or 'cnn'.
        num_workers (int): Number of worker processes.
        status_queue (queue.Queue): Queue to put status messages for the GUI.
    """
    global DETECTION_MODEL # Ensure the global is updated if needed within this scope
    DETECTION_MODEL = detection_model

    # Helper to put messages in the queue
    def report_status(message):
        try:
            status_queue.put(message)
        except Exception as e:
            # If GUI is closed or queue fails, just print
            print(f"Queue Error: {e}\nMessage: {message}", file=sys.stderr)

    report_status(f"INFO: Using database: {db_file}")
    report_status(f"INFO: Scanning directories: {', '.join(map(str, directories_to_scan))}")
    report_status(f"INFO: Using {num_workers} worker processes.")
    report_status(f"INFO: Using face detection model: {DETECTION_MODEL}")
    if force_rescan:
        report_status("INFO: Forcing rescan of all found images.")

    start_time = time.time()

    # --- Database and File Discovery ---
    try:
        setup_database(db_file)
        indexed_files_data = get_indexed_files(db_file)
        report_status(f"INFO: Found {len(indexed_files_data)} files previously indexed.")

        all_image_paths = list(find_image_files(directories_to_scan))
        total_found = len(all_image_paths)

        # --- Filter Files Needing Processing ---
        files_to_process = []
        skipped_count = 0
        check_errors = 0

        if force_rescan:
            report_status("INFO: Force rescanning enabled, processing all found images.")
            files_to_process = all_image_paths
        else:
            report_status("INFO: Checking which files are new or modified...")
            # Quick check to estimate work
            approx_new_modified = 0
            for image_path in all_image_paths:
                 try:
                     current_mod_time = image_path.stat().st_mtime
                     if image_path not in indexed_files_data or current_mod_time > indexed_files_data[image_path]:
                         approx_new_modified += 1
                 except: # Ignore errors here, handle below
                    pass
            report_status(f"INFO: Estimated {approx_new_modified} files need processing.")

            for image_path in all_image_paths:
                try:
                    current_mod_time = image_path.stat().st_mtime
                    if image_path not in indexed_files_data or current_mod_time > indexed_files_data[image_path]:
                        files_to_process.append(image_path)
                    else:
                        skipped_count += 1
                except FileNotFoundError:
                    report_status(f"WARN: File found during scan no longer exists: {image_path}")
                    check_errors += 1
                except Exception as e:
                    report_status(f"ERROR: checking file status {image_path}: {e}")
                    check_errors += 1

            report_status(f"INFO: Identified {len(files_to_process)} new or modified files to process.")
            report_status(f"INFO: Skipped {skipped_count} already indexed and unchanged files.")
            if check_errors > 0:
                report_status(f"WARN: Encountered {check_errors} errors during file status checks.")

        if not files_to_process:
            report_status("INFO: No new or modified files to process.")
            end_time = time.time()
            report_status(f"INFO: Total time: {end_time - start_time:.2f} seconds.")
            report_status("DONE") # Signal completion
            return

        # --- Parallel Processing ---
        processed_files_count = 0
        faces_stored_count = 0
        file_process_errors = 0
        db_errors = 0
        conn = None

        try:
            conn = sqlite3.connect(db_file, timeout=10)
            cursor = conn.cursor()
            report_status(f"INFO: Starting face detection and encoding with {num_workers} workers...")

            pool = multiprocessing.Pool(processes=num_workers)
            try:
                results_iterator = pool.imap_unordered(process_image, files_to_process)
                total_to_process = len(files_to_process)

                for i, (img_path_result, face_data_list) in enumerate(results_iterator):
                    percent_done = (i + 1) / total_to_process * 100
                    # Report progress more granularly
                    report_status(f"PROGRESS:{percent_done:.1f}:{img_path_result.name}")

                    img_path_str = str(img_path_result)

                    try:
                        cursor.execute("DELETE FROM faces WHERE file_path = ?", (img_path_str,))
                        if face_data_list:
                            insert_count = 0
                            for last_modified, encoding_blob, location_str in face_data_list:
                               cursor.execute('''
                                    INSERT INTO faces (file_path, last_modified, face_encoding, face_location_css)
                                    VALUES (?, ?, ?, ?)
                               ''', (img_path_str, last_modified, encoding_blob, location_str))
                               insert_count += 1
                            conn.commit()
                            processed_files_count += 1
                            faces_stored_count += insert_count
                        elif face_data_list is None:
                            file_process_errors += 1
                            conn.commit() # Commit delete

                    except sqlite3.Error as e:
                        report_status(f"ERROR: Database error updating for {img_path_str}: {e}")
                        try:
                            conn.rollback()
                        except: pass # Ignore rollback errors if connection is bad
                        db_errors += 1
                        # Maybe try to reconnect or just report failure for this file?

                report_status("\nINFO: Processing complete.")

            except Exception as e:
                report_status(f"ERROR: during multiprocessing pool execution: {e}")
            finally:
                pool.close()
                pool.join()

        except sqlite3.Error as e:
            report_status(f"ERROR: Failed to connect to database {db_file}: {e}")
            report_status("ABORTED") # Signal failure
            return # Can't proceed
        finally:
            if conn:
                conn.close()

        # --- Final Summary ---
        end_time = time.time()
        duration = end_time - start_time
        report_status("\n--- Indexing Summary ---")
        report_status(f"Total potential image files found: {total_found}")
        report_status(f"Files checked for processing: {len(files_to_process)}")
        report_status(f"Files successfully processed (faces stored): {processed_files_count}")
        report_status(f"Total face embeddings stored in this run: {faces_stored_count}")
        report_status(f"Files skipped (unchanged): {skipped_count}")
        report_status(f"Files with processing errors or no faces: {file_process_errors}")
        report_status(f"Database update errors: {db_errors}")
        report_status(f"Check/Stat errors before processing: {check_errors}")
        report_status(f"Total execution time: {duration:.2f} seconds.")
        report_status("DONE") # Signal completion

    except Exception as e:
        # Catch any other unexpected errors
        import traceback
        error_details = traceback.format_exc()
        report_status(f"FATAL ERROR during indexing: {e}\n{error_details}")
        report_status("ABORTED")

# --- Command-Line Execution Logic ---
def main_cli(): # Renamed from main to avoid conflict if needed
    parser = argparse.ArgumentParser(
        description="Scan directories for photos, detect faces, and store embeddings in a database.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('directories', metavar='DIR', type=str, nargs='+',
                        help='One or more directories to scan recursively for photos.')
    parser.add_argument('--db', type=str, default=DEFAULT_DB_FILE,
                        help=f'Path to the SQLite database file.\n(default: {DEFAULT_DB_FILE})')
    parser.add_argument('--workers', type=int, default=multiprocessing.cpu_count(),
                        help='Number of worker processes to use for parallel processing.\n(default: number of CPU cores)')
    parser.add_argument('--force-rescan', action='store_true',
                        help='Force reprocessing of all images, even if they seem unchanged in the database.')
    parser.add_argument('--detection-model', type=str, default=DETECTION_MODEL, choices=['hog', 'cnn'],
                        help=f'Face detection model to use (hog is faster, cnn is more accurate).\n(default: {DETECTION_MODEL})')
    args = parser.parse_args()

    # --- Use a dummy queue for CLI output ---
    cli_queue = queue.Queue()

    # Function to print messages from the dummy queue
    def print_status():
        while not cli_queue.empty():
            message = cli_queue.get()
            # Simple parsing for CLI output
            if message.startswith("PROGRESS:"):
                parts = message.split(":")
                print(f"[{parts[1]}%] Completed: {parts[2]}", end='\r')
            elif message.startswith("INFO:") or message.startswith("WARN:") or message.startswith("ERROR:"):
                print(message.split(":", 1)[1].strip())
            elif message in ["DONE", "ABORTED"]:
                print(f"\nIndexer finished with status: {message}")
            else: # Print raw message otherwise
                 print(message)

    # Run indexing in a separate thread to allow printing progress
    indexing_thread = threading.Thread(
        target=run_indexing,
        args=(args.directories, args.db, args.force_rescan, args.detection_model, max(1, args.workers), cli_queue)
    )
    indexing_thread.start()

    # Check the queue periodically and print status
    while indexing_thread.is_alive():
        print_status()
        time.sleep(0.1) # Small delay to avoid busy-waiting

    # Print any remaining messages
    print_status()

# --- Entry Point ---
if __name__ == "__main__":
    import threading # Need threading for CLI version too now
    multiprocessing.freeze_support()
    main_cli() # Call the CLI main function 