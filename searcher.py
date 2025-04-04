#!/usr/bin/env python3

import face_recognition
import sqlite3
import argparse
import time
import sys
import numpy as np
from pathlib import Path
import queue # Added for GUI communication

# --- Configuration ---
DEFAULT_DB_FILE = "photo_face_index.db"
# Default tolerance for face matching. Lower is stricter. 0.6 is typical.
DEFAULT_TOLERANCE = 0.6

# --- Database Functions ---

def blob_to_np_array(blob):
    """Converts SQLite BLOB back to a NumPy array."""
    return np.frombuffer(blob, dtype=np.float64) # Make sure dtype matches encoding

def load_all_encodings_from_db(db_file, status_queue): # Added status_queue
    """Loads all face encodings and their file paths, reports status."""
    if not Path(db_file).exists():
        status_queue.put(f"ERROR: Database file '{db_file}' not found.")
        status_queue.put("Please run the indexer script first.")
        return None, None # Indicate failure

    encodings = []
    file_paths = []
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path, face_encoding FROM faces")
        rows = cursor.fetchall()

        if not rows:
            status_queue.put(f"WARN: Database '{db_file}' appears to be empty. No faces indexed.")
            return [], [] # Return empty lists, not necessarily an error state

        for row in rows:
            file_path = row[0]
            encoding_blob = row[1]
            try:
                encoding_array = blob_to_np_array(encoding_blob)
                if encoding_array.shape[0] == 128:
                    encodings.append(encoding_array)
                    file_paths.append(file_path)
                else:
                    status_queue.put(f"WARN: Skipping invalid encoding shape {encoding_array.shape} for {file_path}")
            except Exception as e:
                status_queue.put(f"WARN: Could not decode blob for {file_path}: {e}")

    except sqlite3.Error as e:
        status_queue.put(f"ERROR: Database error reading encodings: {e}")
        return None, None # Indicate failure
    finally:
        if conn:
            conn.close()

    if not encodings and rows: # We read rows but none were valid
        status_queue.put("ERROR: No valid encodings were loaded from the database.")
        return None, None # Indicate failure
    elif not encodings and not rows: # DB was empty
        pass # Already warned above

    status_queue.put(f"INFO: Loaded {len(encodings)} known face encodings from the database.")
    return file_paths, encodings

# --- Function to be called by GUI ---
def run_search(reference_image_path_str, db_file, tolerance, status_queue):
    """
    Runs the search process and returns matching file paths. Puts status in queue.
    Args:
        reference_image_path_str (str): Path to the reference image.
        db_file (str or Path): Path to the database file.
        tolerance (float): Matching tolerance.
        status_queue (queue.Queue): Queue for status messages.
    Returns:
        list: A list of matching file paths, or None if a fatal error occurs.
    """
    def report_status(message):
        try:
            status_queue.put(message)
        except: # Ignore queue errors
            print(f"Queue Error. Message: {message}", file=sys.stderr)

    reference_image_path = Path(reference_image_path_str)
    if not reference_image_path.is_file():
        report_status(f"ERROR: Reference image not found at '{reference_image_path_str}'")
        report_status("ABORTED")
        return None

    report_status(f"INFO: Using database: {db_file}")
    report_status(f"INFO: Using reference image: {reference_image_path_str}")
    report_status(f"INFO: Using matching tolerance: {tolerance}")

    start_time = time.time()

    # --- Load Known Faces from DB ---
    known_file_paths, known_face_encodings = load_all_encodings_from_db(db_file, status_queue)
    if known_face_encodings is None: # Indicates DB load error
        report_status("ABORTED")
        return None
    if not known_face_encodings: # DB might be empty
        report_status("WARN: No faces loaded from DB to compare against.")
        report_status("DONE")
        return [] # Return empty list

    # --- Process Reference Image ---
    report_status("INFO: Processing reference image...")
    try:
        ref_image = face_recognition.load_image_file(reference_image_path)
        ref_face_locations = face_recognition.face_locations(ref_image, model="hog") # Keep consistent or make configurable
        ref_face_encodings = face_recognition.face_encodings(ref_image, ref_face_locations)

    except Exception as e:
        report_status(f"ERROR: loading or processing reference image: {e}")
        report_status("ABORTED")
        return None

    if not ref_face_encodings:
        report_status(f"ERROR: No faces found in the reference image '{reference_image_path_str}'.")
        report_status("ABORTED")
        return None

    target_encoding = None
    if len(ref_face_encodings) == 1:
        target_encoding = ref_face_encodings[0]
        report_status("INFO: Found 1 face in reference image. Using it for searching.")
    else:
        target_encoding = ref_face_encodings[0]
        report_status(f"WARN: Found {len(ref_face_encodings)} faces in reference image. Using the first one.")

    # --- Compare Faces ---
    report_status("INFO: Comparing reference face against database...")
    matches = face_recognition.compare_faces(known_face_encodings, target_encoding, tolerance=tolerance)

    # --- Collect Results ---
    matching_file_paths = set()
    for i, match in enumerate(matches):
        if match:
            matching_file_paths.add(known_file_paths[i])

    end_time = time.time()
    duration = end_time - start_time

    # --- Final Summary (via status queue) ---
    report_status(f"INFO: Comparison finished in {duration:.2f} seconds.")

    sorted_paths = sorted(list(matching_file_paths))

    if not sorted_paths:
        report_status(f"INFO: No matches found with tolerance {tolerance}.")
    else:
        report_status(f"INFO: Found {len(sorted_paths)} photos containing a matching face:")
        # Send results one by one or as a single block? Let's send one by one prefixed.
        for file_path in sorted_paths:
            report_status(f"RESULT:{file_path}")

    report_status("DONE") # Signal completion
    return sorted_paths # Return the list for potential programmatic use

# --- Command-Line Execution Logic ---
def main_cli():
    parser = argparse.ArgumentParser(
        description="Search for photos matching a person in a reference image using a pre-built index.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('reference_image', metavar='IMAGE_PATH', type=str,
                        help='Path to the photo containing the person to search for.')
    parser.add_argument('--db', type=str, default=DEFAULT_DB_FILE,
                        help=f'Path to the SQLite database file.\n(default: {DEFAULT_DB_FILE})')
    parser.add_argument('--tolerance', type=float, default=DEFAULT_TOLERANCE,
                        help=f'Face recognition tolerance (lower is stricter).\n(default: {DEFAULT_TOLERANCE})')
    args = parser.parse_args()

    # Use a dummy queue for CLI output
    cli_queue = queue.Queue()
    results = []

    # Function to print status and collect results
    def handle_status():
        final_results = None
        while True:
            try:
                message = cli_queue.get_nowait()
                if message.startswith("RESULT:"):
                    results.append(message.split(":", 1)[1])
                elif message.startswith("INFO:") or message.startswith("WARN:") or message.startswith("ERROR:"):
                     print(message.split(":", 1)[1].strip())
                elif message == "DONE":
                    # print("Search process finished.")
                    break # Exit loop on DONE
                elif message == "ABORTED":
                    print("Search process aborted due to error.")
                    break
                else:
                    print(message) # Print raw message
            except queue.Empty:
                break # No more messages for now
        return final_results

    # Run search in a separate thread
    search_thread = threading.Thread(
        target=lambda: setattr(search_thread, 'result', run_search(args.reference_image, args.db, args.tolerance, cli_queue)),
        daemon=True # Allows main program to exit even if this thread hangs (though it shouldn't)
    )
    search_thread.start()

    # Check queue while thread is running
    while search_thread.is_alive():
        handle_status()
        time.sleep(0.1)

    # Final check for messages and process result
    handle_status()

    # Access the result stored in the thread object if needed (optional)
    # final_results = getattr(search_thread, 'result', None)
    # if final_results is not None:
    #    print("\n--- Final Results List (from return value) ---")
    #    for path in final_results:
    #        print(f"- {path}")
    # elif not results: # Check if we got results via queue if return value failed
    #    print("No matching photos found.")

if __name__ == "__main__":
    import threading # Need threading for CLI version
    main_cli() 