import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import shutil # Added for copying files
from pathlib import Path
import multiprocessing
import time # Added for export timing

# --- Image handling ---
try:
    from PIL import Image, ImageTk, UnidentifiedImageError
except ImportError:
    messagebox.showerror("Missing Library", "Pillow library not found.\nPlease install it using: pip3 install Pillow")
    exit()

# Import the refactored functions from our scripts
# Make sure indexer.py and searcher.py are in the same directory or Python path
try:
    import indexer
    import searcher
except ImportError as e:
    messagebox.showerror("Import Error", f"Could not import indexer or searcher modules.\nMake sure indexer.py and searcher.py are in the same directory.\nError: {e}")
    exit()

# Default DB file (can be overridden)
DEFAULT_DB_FILE = "photo_face_index.db"
THUMBNAIL_SIZE = (120, 120) # Size for preview tiles

class FaceFinderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Finder")
        self.root.geometry("800x750") # Increased size for preview

        # --- Shared Variables ---
        self.db_path = tk.StringVar(value=DEFAULT_DB_FILE)
        self.index_dirs = []
        self.reference_image_path = tk.StringVar()
        self.status_queue = queue.Queue()
        self.is_processing = False # Flag to prevent multiple operations
        self.current_results = [] # To store result paths for export
        self.thumbnail_cache = [] # To prevent GC of PhotoImage objects

        # --- GUI Setup ---
        self.create_widgets()
        self.check_queue() # Start checking the queue for updates

    def create_widgets(self):
        # Main container with two paned windows
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top section for controls (indexing, searching, status)
        top_section = ttk.PanedWindow(main_container, orient=tk.VERTICAL)
        top_section.pack(fill=tk.X, expand=False)

        # Bottom section for results preview (will be resizable)
        bottom_section = ttk.PanedWindow(main_container, orient=tk.VERTICAL)
        bottom_section.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # --- Controls Section ---
        controls_frame = ttk.Frame(top_section)
        top_section.add(controls_frame, weight=1)

        # Compact indexing section
        index_frame = ttk.LabelFrame(controls_frame, text="1. Indexing", padding="5")
        index_frame.pack(fill=tk.X, pady=2)

        dir_select_frame = ttk.Frame(index_frame)
        dir_select_frame.pack(fill=tk.X)
        ttk.Label(dir_select_frame, text="Dir:").pack(side=tk.LEFT, padx=(0, 5))
        self.index_dir_button = ttk.Button(dir_select_frame, text="Select...", command=self.select_index_dirs, width=8)
        self.index_dir_button.pack(side=tk.LEFT)
        self.index_dir_label = ttk.Label(dir_select_frame, text="None selected", wraplength=300)
        self.index_dir_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        index_options_frame = ttk.Frame(index_frame)
        index_options_frame.pack(fill=tk.X, pady=2)
        self.force_rescan_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(index_options_frame, text="Force Rescan", variable=self.force_rescan_var).pack(side=tk.LEFT)
        self.index_button = ttk.Button(index_options_frame, text="Start Indexing", command=self.start_indexing, width=12)
        self.index_button.pack(side=tk.RIGHT, padx=5)

        # Compact searching section
        search_frame = ttk.LabelFrame(controls_frame, text="2. Searching", padding="5")
        search_frame.pack(fill=tk.X, pady=2)

        ref_img_frame = ttk.Frame(search_frame)
        ref_img_frame.pack(fill=tk.X)
        ttk.Label(ref_img_frame, text="Ref:").pack(side=tk.LEFT, padx=(0, 5))
        self.ref_img_button = ttk.Button(ref_img_frame, text="Select...", command=self.select_ref_image, width=8)
        self.ref_img_button.pack(side=tk.LEFT)
        self.ref_img_label = ttk.Label(ref_img_frame, text="None selected", wraplength=300)
        self.ref_img_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        search_options_frame = ttk.Frame(search_frame)
        search_options_frame.pack(fill=tk.X, pady=2)
        ttk.Label(search_options_frame, text="Tolerance:").pack(side=tk.LEFT)
        self.tolerance_var = tk.StringVar(value="0.6")
        ttk.Entry(search_options_frame, textvariable=self.tolerance_var, width=4).pack(side=tk.LEFT, padx=5)
        ttk.Label(search_options_frame, text="(0.4-0.7)").pack(side=tk.LEFT)
        self.search_button = ttk.Button(search_options_frame, text="Search", command=self.start_search, width=8)
        self.search_button.pack(side=tk.RIGHT, padx=5)

        # Compact database path section
        db_frame = ttk.Frame(controls_frame)
        db_frame.pack(fill=tk.X, pady=2)
        ttk.Label(db_frame, text="DB:").pack(side=tk.LEFT)
        ttk.Entry(db_frame, textvariable=self.db_path, width=30).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(db_frame, text="Browse", command=self.select_db_path, width=8).pack(side=tk.LEFT)

        # --- Status Section ---
        status_frame = ttk.LabelFrame(top_section, text="Status", padding="5")
        top_section.add(status_frame, weight=1)

        self.status_text = scrolledtext.ScrolledText(status_frame, height=4, width=80, wrap=tk.WORD, state=tk.DISABLED)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        self.progress_bar = ttk.Progressbar(status_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))

        # --- Results Preview Section ---
        results_frame = ttk.LabelFrame(bottom_section, text="3. Results Preview & Export", padding="5")
        bottom_section.add(results_frame, weight=1)

        # Export controls
        export_frame = ttk.Frame(results_frame)
        export_frame.pack(fill=tk.X, pady=(0, 5))
        self.export_button = ttk.Button(export_frame, text="Export Found Images...", command=self.export_results, state=tk.DISABLED)
        self.export_button.pack(side=tk.LEFT)
        self.results_count_label = ttk.Label(export_frame, text="Found: 0 images")
        self.results_count_label.pack(side=tk.RIGHT)

        # Thumbnails canvas
        canvas_frame = ttk.Frame(results_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, borderwidth=0)
        self.preview_frame = ttk.Frame(self.canvas)
        self.vsb = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas_frame_window = self.canvas.create_window((4,4), window=self.preview_frame, anchor="nw", tags="self.preview_frame")

        # Bind canvas events
        self.preview_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

        # Add a separator between top and bottom sections
        ttk.Separator(main_container, orient='horizontal').pack(fill=tk.X, pady=5)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling across platforms."""
        if event.num == 5 or event.delta < 0: # Scroll down
            delta = 1
        elif event.num == 4 or event.delta > 0: # Scroll up
            delta = -1
        else:
            delta = 0
        self.canvas.yview_scroll(delta, "units")

    def on_frame_configure(self, event):
        """Reset the scroll region to encompass the inner frame"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def update_status(self, message):
        """Appends a message to the status text area, thread-safe."""
        self.status_text.configure(state=tk.NORMAL)
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.configure(state=tk.DISABLED)
        self.status_text.see(tk.END) # Scroll to the bottom

    def check_queue(self):
        """Checks the queue for messages from worker threads."""
        try:
            while True:
                message = self.status_queue.get_nowait()
                # Simple parsing of messages from worker threads
                if message.startswith("PROGRESS:"):
                    parts = message.split(":")
                    try:
                        progress_val = float(parts[1])
                        self.progress_bar['value'] = progress_val
                        # Optionally display file name too
                        # self.update_status(f"Processing: {parts[2]}")
                    except:
                        self.update_status(message) # Show raw message on error
                elif message.startswith("RESULT:"):
                    self.current_results.append(message.split(":", 1)[1])
                elif message == "PREVIEW_START":
                    self.update_status("Generating result previews...")
                    self.progress_bar['mode'] = 'indeterminate'
                    self.progress_bar.start()
                elif message == "PREVIEW_DONE":
                    self.update_status("Preview generation complete.")
                    self.progress_bar.stop()
                    self.progress_bar['mode'] = 'determinate'
                    self.progress_bar['value'] = 0
                    self.enable_buttons()
                    if self.current_results:
                        self.export_button.config(state=tk.NORMAL)
                    self.results_count_label.config(text=f"Found: {len(self.current_results)} images")
                elif message in ["DONE", "ABORTED"]:
                    self.update_status(f"Operation finished: {message}")
                    if message == "DONE" and self.current_results:
                        self.disable_buttons()
                        self.export_button.config(state=tk.DISABLED)
                        self.results_count_label.config(text=f"Found: {len(self.current_results)} images")
                        thread = threading.Thread(target=self.populate_preview_area, args=(self.current_results,), daemon=True)
                        thread.start()
                    else:
                        self.enable_buttons()
                        self.progress_bar['value'] = 0
                        self.results_count_label.config(text=f"Found: {len(self.current_results)} images")
                        if not self.current_results:
                            self.export_button.config(state=tk.DISABLED)
                else:
                    # Default: treat as status message (handles INFO, WARN, ERROR prefixes)
                    prefix = message.split(":", 1)[0]
                    if prefix in ["INFO", "WARN", "ERROR", "FATAL ERROR"]:
                         self.update_status(message.split(":", 1)[1].strip())
                    else:
                         self.update_status(message) # Show unknown messages raw

        except queue.Empty:
            pass # No messages currently
        finally:
            # Reschedule the check
            self.root.after(100, self.check_queue)

    def disable_buttons(self):
        self.is_processing = True
        self.index_button.config(state=tk.DISABLED)
        self.search_button.config(state=tk.DISABLED)
        self.index_dir_button.config(state=tk.DISABLED)
        self.ref_img_button.config(state=tk.DISABLED)
        self.export_button.config(state=tk.DISABLED)

    def enable_buttons(self):
        self.is_processing = False
        self.index_button.config(state=tk.NORMAL)
        self.search_button.config(state=tk.NORMAL)
        self.index_dir_button.config(state=tk.NORMAL)
        self.ref_img_button.config(state=tk.NORMAL)
        if self.current_results:
            self.export_button.config(state=tk.NORMAL)

    def select_index_dirs(self):
        if self.is_processing: return
        # Allow selecting multiple directories
        # On macOS, askdirectory doesn't directly support multi-select.
        # Workaround: Ask repeatedly or inform user to select one top-level folder.
        # Let's just ask for one for simplicity now.
        directory = filedialog.askdirectory(title="Select Directory to Index")
        if directory:
             # Replace list with the single selected dir for now
             self.index_dirs = [directory]
             self.index_dir_label.config(text=directory)
             self.update_status(f"Selected directory for indexing: {directory}")
        # TODO: Implement better multi-directory selection if needed

    def select_ref_image(self):
        if self.is_processing: return
        filepath = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.heic *.heif *.bmp *.gif"), ("All Files", "*.*")]
        )
        if filepath:
            self.reference_image_path.set(filepath)
            self.ref_img_label.config(text=os.path.basename(filepath)) # Show only filename
            self.update_status(f"Selected reference image: {filepath}")

    def select_db_path(self):
         if self.is_processing: return
         # Ask to save file to allow creating a new one or selecting existing
         filepath = filedialog.asksaveasfilename(
             title="Select or Create Database File",
             initialfile=os.path.basename(self.db_path.get()),
             defaultextension=".db",
             filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")]
             )
         if filepath:
             self.db_path.set(filepath)
             self.update_status(f"Set database path to: {filepath}")

    def start_indexing(self):
        if self.is_processing:
            messagebox.showwarning("Busy", "Another operation is already in progress.")
            return
        if not self.index_dirs:
            messagebox.showerror("Error", "Please select at least one directory to index.")
            return

        db = self.db_path.get()
        if not db:
             messagebox.showerror("Error", "Please specify a database file path.")
             return

        force = self.force_rescan_var.get()
        # Use default settings from indexer for now
        detection_model = indexer.DETECTION_MODEL
        num_workers = multiprocessing.cpu_count()

        self.disable_buttons()
        self.status_text.configure(state=tk.NORMAL)
        self.status_text.delete('1.0', tk.END) # Clear previous status/results
        self.status_text.configure(state=tk.DISABLED)
        self.update_status("Starting indexing process...")
        self.progress_bar['value'] = 0

        # Run indexing in a separate thread
        thread = threading.Thread(
            target=indexer.run_indexing,
            args=(self.index_dirs, db, force, detection_model, num_workers, self.status_queue),
            daemon=True # Allows app to exit even if thread hangs
        )
        thread.start()

    def start_search(self):
        if self.is_processing:
            messagebox.showwarning("Busy", "Another operation is already in progress.")
            return

        ref_image = self.reference_image_path.get()
        if not ref_image:
            messagebox.showerror("Error", "Please select a reference image.")
            return

        db = self.db_path.get()
        if not db:
             messagebox.showerror("Error", "Please specify a database file path.")
             return
        if not Path(db).exists():
             messagebox.showerror("Error", f"Database file not found:\n{db}\nPlease run indexing first or check path.")
             return

        try:
            tolerance = float(self.tolerance_var.get())
            if not (0 < tolerance < 1): raise ValueError()
        except ValueError:
            messagebox.showerror("Error", "Invalid tolerance value. Must be a number between 0 and 1 (e.g., 0.6).")
            return

        self.disable_buttons()
        self.status_text.configure(state=tk.NORMAL)
        self.status_text.delete('1.0', tk.END) # Clear previous status/results
        self.status_text.configure(state=tk.DISABLED)
        self.update_status(f"Starting search for: {os.path.basename(ref_image)}")
        self.progress_bar['value'] = 0 # Reset progress for search phase

        # Run search in a separate thread
        thread = threading.Thread(
            target=searcher.run_search,
            args=(ref_image, db, tolerance, self.status_queue),
            daemon=True
        )
        thread.start()

    def export_results(self):
        if not self.current_results:
            messagebox.showinfo("Export", "No results to export.")
            return
        if self.is_processing:
            messagebox.showwarning("Busy", "Cannot export while another operation is running.")
            return

        dest_dir = filedialog.askdirectory(title="Select Destination Folder for Export")
        if not dest_dir:
            return

        self.update_status(f"Starting export to: {dest_dir}")
        self.disable_buttons()

        thread = threading.Thread(target=self._perform_export, args=(list(self.current_results), dest_dir), daemon=True)
        thread.start()

    def _perform_export(self, paths_to_export, destination_folder):
        """Actual file copying logic (runs in thread)."""
        export_count = 0
        export_errors = 0
        start_time = time.time()

        dest_path = Path(destination_folder)
        if not dest_path.exists():
            try:
                dest_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.status_queue.put(f"ERROR: Could not create destination folder: {e}")
                self.status_queue.put("ABORTED")
                return

        total_files = len(paths_to_export)
        for i, src_path_str in enumerate(paths_to_export):
            src_path = Path(src_path_str)
            dest_file_path = dest_path / src_path.name

            percent = ((i + 1) / total_files) * 100
            self.status_queue.put(f"PROGRESS:{percent:.1f}:Exporting {src_path.name}")

            if not src_path.exists():
                self.status_queue.put(f"WARN: Skipping export, source file not found: {src_path.name}")
                export_errors += 1
                continue

            try:
                shutil.copy2(src_path, dest_file_path)
                export_count += 1
            except Exception as e:
                self.status_queue.put(f"ERROR: Failed to copy {src_path.name}: {e}")
                export_errors += 1

        duration = time.time() - start_time
        self.status_queue.put(f"INFO: Export finished in {duration:.2f} seconds.")
        self.status_queue.put(f"INFO: Successfully exported {export_count} files.")
        if export_errors > 0:
            self.status_queue.put(f"WARN: Failed to export {export_errors} files.")
        self.status_queue.put("DONE")

    def clear_preview_area(self):
        """Clears the thumbnail previews."""
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        self.thumbnail_cache.clear()

    def populate_preview_area(self, file_paths):
        """Loads thumbnails and displays them. Runs in a thread."""
        self.status_queue.put("PREVIEW_START")
        self.clear_preview_area()

        cols = 5
        temp_cache = []
        valid_paths_for_preview = []

        for idx, file_path_str in enumerate(file_paths):
            try:
                file_path = Path(file_path_str)
                if not file_path.is_file():
                    self.status_queue.put(f"WARN: Preview skipped, file not found: {file_path.name}")
                    continue

                img = Image.open(file_path)
                img.thumbnail(THUMBNAIL_SIZE)
                photo_img = ImageTk.PhotoImage(img)
                temp_cache.append(photo_img)
                valid_paths_for_preview.append(file_path_str)

            except UnidentifiedImageError:
                self.status_queue.put(f"WARN: Preview skipped, cannot identify image: {file_path.name}")
            except Exception as e:
                self.status_queue.put(f"ERROR: Preview failed for {file_path.name}: {e}")

        self.root.after(0, self._update_gui_with_previews, temp_cache, valid_paths_for_preview, cols)

    def _update_gui_with_previews(self, photo_images, file_paths, cols):
        """This function runs in the main GUI thread to update widgets."""
        self.thumbnail_cache = photo_images

        for idx, (photo_img, file_path_str) in enumerate(zip(photo_images, file_paths)):
            row = idx // cols
            col = idx % cols

            tile_frame = ttk.Frame(self.preview_frame, padding=2, relief=tk.RIDGE, borderwidth=1)
            tile_frame.grid(row=row, column=col, padx=3, pady=3)

            lbl = ttk.Label(tile_frame, image=photo_img)
            lbl.image = photo_img
            lbl.pack()

            fname_lbl = ttk.Label(tile_frame, text=Path(file_path_str).name, wraplength=THUMBNAIL_SIZE[0])
            fname_lbl.pack()

        self.preview_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.status_queue.put("PREVIEW_DONE")

# --- Main Execution ---
if __name__ == "__main__":
    # freeze_support is needed for multiprocessing when bundled (e.g., PyInstaller)
    multiprocessing.freeze_support()

    root = tk.Tk()
    app = FaceFinderApp(root)
    root.mainloop() 