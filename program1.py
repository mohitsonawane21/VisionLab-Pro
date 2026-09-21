# ============================================================
# VISIONLAB PRO
# Professional Interactive Computer Vision & Image Processing
# Python 3.11+
#
# Install:
#   py -m pip install opencv-python pillow numpy
#
# Run:
#   py visionlab_pro.py
# ============================================================

import os
import io
import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
import urllib.request
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk


class VisionLabPro:
    def __init__(self, root):
        self.root = root
        self.root.title("VisionLab Pro — Computer Vision Studio")
        self.root.geometry("1450x900")
        self.root.minsize(1180, 760)

        self.original_image = None
        self.current_image = None
        self.image_path = None
        self.history = []
        self.history_index = -1
        self.last_operation = "Ready"
        self.preview_refs = []
        self.web_cache = {}

        self.build_style()
        self.build_header()
        self.build_toolbar()
        self.build_body()
        self.build_status_bar()
        self.bind_shortcuts()

        self.root.after(200, self.refresh_dashboard)

    # ========================================================
    # STYLE
    # ========================================================

    def build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 7))
        style.configure("Small.TButton", font=("Segoe UI", 9), padding=(7, 5))
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10))
        style.configure("Section.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Treeview", rowheight=25)

    # ========================================================
    # HEADER
    # ========================================================

    def build_header(self):
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=16, pady=(12, 6))

        left = ttk.Frame(header)
        left.pack(side="left", fill="x", expand=True)

        ttk.Label(left, text="VISIONLAB PRO", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            left,
            text="Interactive Computer Vision & Image Processing Studio"
        ).pack(anchor="w", pady=(2, 0))

        self.header_state = ttk.Label(
            header,
            text="NO IMAGE LOADED",
            font=("Segoe UI", 10, "bold")
        )
        self.header_state.pack(side="right", padx=5)

    # ========================================================
    # TOP TOOLBAR
    # ========================================================

    def build_toolbar(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill="x", padx=16, pady=(2, 8))

        groups = [
            ("Open", self.open_image),
            ("Web Image", self.web_image_search),
            ("Image URL", self.load_image_from_url_dialog),
            ("Save", self.save_direct),
            ("Save As", self.save_as),
            ("Undo", self.undo),
            ("Redo", self.redo),
            ("Reset", self.reset_image),
            ("Image Info", self.show_image_info),
        ]

        for text, command in groups:
            ttk.Button(toolbar, text=text, command=command).pack(
                side="left", padx=3
            )

        ttk.Separator(toolbar, orient="vertical").pack(
            side="left", fill="y", padx=8
        )

        ttk.Label(toolbar, text="Preview:").pack(side="left", padx=(2, 4))

        self.preview_mode = tk.StringVar(value="Fit")
        preview_box = ttk.Combobox(
            toolbar,
            textvariable=self.preview_mode,
            values=["Fit", "50%", "75%", "100%"],
            width=8,
            state="readonly"
        )
        preview_box.pack(side="left")
        preview_box.bind("<<ComboboxSelected>>", lambda e: self.update_preview())

        ttk.Button(
            toolbar,
            text="About",
            command=self.show_about
        ).pack(side="right", padx=3)

    # ========================================================
    # BODY
    # ========================================================

    def build_body(self):
        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=16, pady=4)

        # LEFT: operations
        left = ttk.LabelFrame(body, text="TOOLS")
        left.pack(side="left", fill="y", padx=(0, 10))

        tools_canvas = tk.Canvas(left, width=245, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=tools_canvas.yview)
        tools_frame = ttk.Frame(tools_canvas)

        tools_frame.bind(
            "<Configure>",
            lambda e: tools_canvas.configure(
                scrollregion=tools_canvas.bbox("all")
            )
        )

        tools_canvas.create_window((0, 0), window=tools_frame, anchor="nw")
        tools_canvas.configure(yscrollcommand=scrollbar.set)

        tools_canvas.pack(side="left", fill="y", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.create_tool_sections(tools_frame)

        # CENTER: preview
        center = ttk.LabelFrame(body, text="IMAGE WORKSPACE")
        center.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.preview_container = ttk.Frame(center)
        self.preview_container.pack(fill="both", expand=True, padx=8, pady=8)

        self.original_panel = self.create_preview_panel(
            self.preview_container, "ORIGINAL"
        )
        self.processed_panel = self.create_preview_panel(
            self.preview_container, "PROCESSED"
        )

        # RIGHT: dashboard
        right = ttk.LabelFrame(body, text="ANALYSIS & STATUS DASHBOARD")
        right.pack(side="right", fill="both", padx=(0, 0))
        right.configure(width=350)
        right.pack_propagate(False)

        self.dashboard = tk.Text(
            right,
            width=42,
            height=32,
            font=("Consolas", 10),
            wrap="word",
            padx=12,
            pady=12,
            state="disabled"
        )
        self.dashboard.pack(fill="both", expand=True, padx=8, pady=8)

        self.refresh_dashboard()

    # ========================================================
    # TOOL SECTIONS
    # ========================================================

    def create_tool_sections(self, parent):
        sections = [
            ("BASIC", [
                ("Grayscale", self.grayscale),
                ("Resize", self.resize_image),
                ("Crop Center", self.crop_center),
                ("Rotate 90°", self.rotate_90),
                ("Rotate 180°", self.rotate_180),
                ("Flip Horizontal", self.flip_horizontal),
                ("Flip Vertical", self.flip_vertical),
            ]),
            ("LIGHT & COLOR", [
                ("Colorize Image", self.colorize_image),
                ("Brightness", self.brightness),
                ("Contrast", self.contrast),
                ("Auto Contrast", self.auto_contrast),
                ("Saturation", self.saturation),
                ("Sepia", self.sepia),
                ("Negative", self.negative),
            ]),
            ("FILTERS", [
                ("Gaussian Blur", self.gaussian_blur),
                ("Median Blur", self.median_blur),
                ("Sharpen", self.sharpen),
                ("Noise Reduction", self.noise_reduction),
            ]),
            ("CV ANALYSIS", [
                ("Edge Detection", self.edge_detection),
                ("Binary Threshold", self.threshold),
                ("Adaptive Threshold", self.adaptive_threshold),
                ("HSV Analysis", self.color_analysis),
                ("Histogram", self.histogram),
                ("Face Detection", self.face_detection),
            ]),
        ]

        for section_name, buttons in sections:
            ttk.Label(
                parent, text=section_name, style="Section.TLabel"
            ).pack(anchor="w", padx=10, pady=(12, 4))

            for text, command in buttons:
                ttk.Button(
                    parent,
                    text=text,
                    command=command,
                    width=25
                ).pack(padx=8, pady=2)

    # ========================================================
    # PREVIEW PANEL
    # ========================================================

    def create_preview_panel(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(side="left", fill="both", expand=True, padx=5)

        label = tk.Label(
            frame,
            text="No image",
            bg="#eeeeee",
            fg="#555555",
            font=("Segoe UI", 11),
            relief="solid",
            bd=1
        )
        label.pack(fill="both", expand=True, padx=5, pady=5)

        meta = ttk.Label(frame, text="—", anchor="center")
        meta.pack(fill="x", padx=5, pady=(0, 5))

        return {"frame": frame, "label": label, "meta": meta}

    # ========================================================
    # STATUS BAR
    # ========================================================

    def build_status_bar(self):
        self.status_var = tk.StringVar(value="Ready — Open an image to begin.")
        status = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padding=(10, 6)
        )
        status.pack(fill="x", side="bottom", padx=0, pady=0)

    def set_status(self, text):
        self.status_var.set(text)
        self.last_operation = text
        self.refresh_dashboard()

    # ========================================================
    # SHORTCUTS
    # ========================================================

    def bind_shortcuts(self):
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-s>", lambda e: self.save_direct())
        self.root.bind("<Control-Shift-S>", lambda e: self.save_as())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Control-r>", lambda e: self.reset_image())

    # ========================================================
    # FILE OPERATIONS
    # ========================================================

    # ========================================================
    # WEB IMAGE SEARCH
    # ========================================================

    def open_image(self):
        """Open a local image safely, including Windows/Unicode paths."""
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Bitmap", "*.bmp"),
                ("TIFF", "*.tif *.tiff"),
                ("WebP", "*.webp"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            # cv2.imread can fail on some Windows Unicode paths, so use
            # np.fromfile + imdecode for reliable path handling.
            raw = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("OpenCV could not decode this image.")

            if len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            self.original_image = image.copy()
            self.current_image = image.copy()
            self.image_path = path
            self.history = []
            self.history_index = -1
            self.last_operation = "Image opened"
            self.add_history("Image opened")
            self.update_preview()
            self.refresh_dashboard()
            self.set_status(f"Opened: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror(
                "Open Image",
                f"Could not open the selected image.\n\n{exc}"
            )

    def _load_image_bytes(self, data):
        """Decode downloaded image bytes into a BGR OpenCV image."""
        if not data:
            raise ValueError("The server returned an empty file.")
        arr = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Downloaded content is not a supported image file.")
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        return image

    def _download_url(self, url, timeout=20, retries=3, referer=None):
        """Download a URL with retry/backoff handling for temporary 429/5xx errors."""
        if not url or not re.match(r"^https?://", url.strip(), re.I):
            raise ValueError("Please provide a valid http:// or https:// URL.")

        url = url.strip()
        last_error = None
        for attempt in range(max(1, retries)):
            headers = {
                "User-Agent": (
                    "VisionLab-Pro/2.0 (educational desktop image tool) "
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/140.0 Safari/537.36"
                ),
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.8",
                "Cache-Control": "no-cache",
            }
            if referer:
                headers["Referer"] = referer
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    return response.read(), response.headers.get("Content-Type", "")
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in (429, 500, 502, 503, 504) or attempt >= retries - 1:
                    raise
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    wait = min(8.0, max(1.0, float(retry_after))) if retry_after else min(8.0, 1.5 * (2 ** attempt))
                except (TypeError, ValueError):
                    wait = min(8.0, 1.5 * (2 ** attempt))
                time.sleep(wait)
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt >= retries - 1:
                    raise
                time.sleep(min(5.0, 1.0 + attempt))

        if last_error:
            raise last_error
        raise RuntimeError("Download failed.")

    def _load_web_image(self, image_url, source_name="Web image", referer=None):
        """Download and make a web image the current VisionLab image."""
        data, content_type = self._download_url(image_url, referer=referer)
        image = self._load_image_bytes(data)
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.image_path = None
        self.history = []
        self.history_index = -1
        self.add_history(f"{source_name} loaded")
        self.update_preview()
        self.refresh_dashboard()
        self.set_status(f"{source_name} loaded successfully.")
        return image

    def load_image_from_url_dialog(self):
        """Fast, non-blocking image/webpage URL loader."""
        win = tk.Toplevel(self.root)
        win.title("VisionLab Pro — Load Image From Website")
        win.geometry("760x280")
        win.minsize(650, 230)
        win.transient(self.root)
        win.grab_set()

        outer = ttk.Frame(win, padding=18)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Paste an image URL or webpage URL",
                  font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(
            outer,
            text="Direct image links are fastest. Webpage URLs are supported when the page exposes og:image, twitter:image, or an <img> URL.",
            wraplength=700,
        ).pack(anchor="w", pady=(5, 12))

        url_var = tk.StringVar()
        entry = ttk.Entry(outer, textvariable=url_var)
        entry.pack(fill="x", ipady=5)
        entry.focus_set()

        status = ttk.Label(outer, text="Ready")
        status.pack(anchor="w", pady=(10, 10))

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(4, 0))
        load_btn = ttk.Button(buttons, text="Load Image")
        load_btn.pack(side="left")
        ttk.Button(buttons, text="Cancel", command=win.destroy).pack(side="right")

        def finish_image(image, message):
            self.original_image = image.copy()
            self.current_image = image.copy()
            self.image_path = None
            self.history = []
            self.history_index = -1
            self.add_history(message)
            self.update_preview()
            self.refresh_dashboard()
            self.set_status(message + ".")
            if win.winfo_exists():
                win.destroy()

        def worker(url):
            try:
                # Direct image request first. One retry only keeps the UI fast.
                data, content_type = self._download_url(url, timeout=12, retries=1)
                try:
                    image = self._load_image_bytes(data)
                    self.root.after(0, lambda: finish_image(image, "Image loaded from URL"))
                    return
                except Exception:
                    pass

                # The response was probably HTML. Extract a useful image URL.
                html = data.decode("utf-8", errors="ignore")
                candidates = []
                patterns = [
                    r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)',
                    r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)',
                    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
                    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image["\']',
                    r'<img[^>]+src=["\']([^"\']+)',
                ]
                for pattern in patterns:
                    candidates.extend(re.findall(pattern, html, re.I))

                # Prefer the first image URL that looks like an actual image.
                image_url = None
                for candidate in candidates:
                    candidate = urllib.parse.urljoin(url, candidate)
                    if re.search(r'\.(?:jpe?g|png|webp|bmp|gif|tiff?)(?:[?#].*)?$', candidate, re.I):
                        image_url = candidate
                        break
                if image_url is None and candidates:
                    image_url = urllib.parse.urljoin(url, candidates[0])
                if not image_url:
                    raise ValueError("This webpage did not expose a downloadable image.")

                image_data, _ = self._download_url(image_url, timeout=12, retries=1, referer=url)
                image = self._load_image_bytes(image_data)
                self.root.after(0, lambda: finish_image(image, "Website image loaded"))
            except Exception as exc:
                self.root.after(0, lambda error=exc: self._url_load_error(win, status, load_btn, error))

        def load():
            url = url_var.get().strip()
            if not url:
                messagebox.showwarning("Image URL", "Paste a URL first.", parent=win)
                return
            if not re.match(r"^https?://", url, re.I):
                messagebox.showwarning("Image URL", "URL must start with http:// or https://", parent=win)
                return
            load_btn.configure(state="disabled")
            status.configure(text="Downloading in background…")
            threading.Thread(target=worker, args=(url,), daemon=True).start()

        load_btn.configure(command=load)
        entry.bind("<Return>", lambda e: load())

    def _url_load_error(self, win, status, load_btn, exc):
        try:
            status.configure(text="Load failed — try another URL.")
            load_btn.configure(state="normal")
            if win.winfo_exists():
                messagebox.showerror(
                    "Image URL",
                    "Could not load an image from this URL.\n\n"
                    f"{exc}\n\n"
                    "For best results, paste the direct image URL.",
                    parent=win,
                )
        except tk.TclError:
            pass

    def web_image_search(self):
        """Fast web image search: network work runs off the Tkinter UI thread."""
        win = tk.Toplevel(self.root)
        win.title("VisionLab Pro — Web Image Search")
        win.geometry("1100x760")
        win.minsize(900, 620)
        win.transient(self.root)

        top = ttk.Frame(win, padding=12)
        top.pack(fill="x")
        ttk.Label(top, text="Web Image Search", font=("Segoe UI", 14, "bold")).pack(side="left", padx=(0, 10))
        query_var = tk.StringVar()
        entry = ttk.Entry(top, textvariable=query_var)
        entry.pack(side="left", fill="x", expand=True)
        source_var = tk.StringVar(value="Auto")
        ttk.Combobox(top, textvariable=source_var,
                     values=["Auto", "Openverse", "Wikimedia Commons"],
                     state="readonly", width=18).pack(side="left", padx=8)

        results_frame = ttk.Frame(win, padding=(12, 0, 12, 8))
        results_frame.pack(fill="both", expand=True)
        canvas = tk.Canvas(results_frame, highlightthickness=0)
        scroll = ttk.Scrollbar(results_frame, orient="vertical", command=canvas.yview)
        grid = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=grid, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        grid.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))

        bottom = ttk.Frame(win, padding=12)
        bottom.pack(fill="x")
        status = ttk.Label(bottom, text="Search for an image.")
        status.pack(side="left", fill="x", expand=True)
        selected = {"item": None, "button": None}
        refs = []
        thumbnail_cache = {}
        thumb_pool = ThreadPoolExecutor(max_workers=4)
        search_token = {"value": 0}

        def close_window():
            search_token["value"] += 1
            try:
                thumb_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            try:
                win.destroy()
            except tk.TclError:
                pass

        win.protocol("WM_DELETE_WINDOW", close_window)

        def clear():
            for child in grid.winfo_children():
                child.destroy()
            refs.clear()
            selected["item"] = None
            selected["button"] = None

        def choose(item, button):
            if selected["button"] is not None:
                try:
                    selected["button"].configure(relief="raised")
                except tk.TclError:
                    pass
            selected["item"] = item
            selected["button"] = button
            button.configure(relief="sunken")
            status.configure(text=f"Selected: {(item.get('title') or 'image')[:100]}")

        def search_openverse(q):
            params = urllib.parse.urlencode({"q": q, "page_size": 12})
            req = urllib.request.Request(
                f"https://api.openverse.org/v1/images/?{params}",
                headers={
                    "User-Agent": "VisionLab-Pro/2.0",
                    "Accept": "application/json",
                    "Accept-Language": "en-US,en;q=0.8",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                payload = json.loads(r.read().decode("utf-8"))
            results = payload.get("results", [])
            return [x for x in results if x.get("url") or x.get("thumbnail")]

        def search_wikimedia(q):
            params = urllib.parse.urlencode({
                "action": "query", "generator": "search", "gsrsearch": q,
                "gsrnamespace": 6, "gsrlimit": 12,
                "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": 360,
                "format": "json", "origin": "*"
            })
            req = urllib.request.Request(
                f"https://commons.wikimedia.org/w/api.php?{params}",
                headers={
                    "User-Agent": "VisionLab-Pro/2.0 (educational desktop image tool)",
                    "Accept": "application/json",
                    "Accept-Language": "en-US,en;q=0.8",
                    "Referer": "https://commons.wikimedia.org/",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                payload = json.loads(r.read().decode("utf-8"))
            results = []
            for page in payload.get("query", {}).get("pages", {}).values():
                info = (page.get("imageinfo") or [{}])[0]
                title = page.get("title", "")
                results.append({
                    "title": title.replace("File:", "", 1),
                    "url": info.get("url"),
                    "thumbnail": info.get("thumburl") or info.get("url"),
                    "source": "Wikimedia Commons",
                })
            return [x for x in results if x.get("url")]

        def make_card(item, index, token):
            row, col = divmod(index, 4)
            card = ttk.Frame(grid, padding=6, relief="ridge")
            card.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            thumb_url = item.get("thumbnail") or item.get("url")
            b = tk.Button(card, text="Loading preview…", width=25, height=8,
                          relief="raised", cursor="hand2")
            b.pack(fill="both", expand=True)
            b.configure(command=lambda it=item, bb=b: choose(it, bb))
            ttk.Label(card, text=(item.get("title") or "Untitled")[:55],
                      wraplength=210, justify="center").pack(pady=(5, 2))
            ttk.Label(card, text=str(item.get("source") or item.get("provider") or "Web result")[:45],
                      font=("Segoe UI", 8)).pack()

            def thumb_worker():
                try:
                    if thumb_url in self.web_cache:
                        data = self.web_cache[thumb_url]
                    else:
                        data, _ = self._download_url(
                            thumb_url, timeout=6, retries=1,
                            referer="https://commons.wikimedia.org/" if "wikimedia.org" in thumb_url else None,
                        )
                        # Small thumbnails only; keeps memory use reasonable.
                        if len(data) <= 2_000_000:
                            self.web_cache[thumb_url] = data
                    thumbnail_cache[id(item)] = data
                    pil = Image.open(io.BytesIO(data)).convert("RGB")
                    pil.thumbnail((200, 145), Image.LANCZOS)
                    return ImageTk.PhotoImage(pil)
                except Exception:
                    return None

            def thumb_done(future):
                if token != search_token["value"] or not win.winfo_exists():
                    return
                try:
                    photo = future.result()
                except Exception:
                    photo = None
                if photo is not None:
                    refs.append(photo)
                    try:
                        b.configure(image=photo, text="")
                    except tk.TclError:
                        pass
                else:
                    try:
                        b.configure(text="Preview unavailable")
                    except tk.TclError:
                        pass

            future = thumb_pool.submit(thumb_worker)
            future.add_done_callback(lambda f: self.root.after(0, thumb_done, f))

        def render(results, token):
            if token != search_token["value"] or not win.winfo_exists():
                return
            clear()
            for i, item in enumerate(results):
                make_card(item, i, token)
            status.configure(text=f"{len(results)} results. Select one and click Load Selected.")

        def search_worker(q, source, token):
            results = []
            errors = []
            if source in ("Openverse", "Auto"):
                try:
                    results = search_openverse(q)
                except Exception as exc:
                    errors.append(f"Openverse: {exc}")
            if not results and source in ("Wikimedia Commons", "Auto"):
                try:
                    results = search_wikimedia(q)
                except Exception as exc:
                    errors.append(f"Wikimedia Commons: {exc}")
            self.root.after(0, lambda: search_done(results, errors, token))

        def search_done(results, errors, token):
            if token != search_token["value"] or not win.winfo_exists():
                return
            search_btn.configure(state="normal")
            if not results:
                clear()
                status.configure(text="Search failed — try another query or source.")
                messagebox.showerror(
                    "Web Search",
                    "No web images could be loaded.\n\n" +
                    ("\n".join(errors) if errors else "No results found.") +
                    "\n\nYou can also use Image URL for a direct image link.",
                    parent=win,
                )
                return
            render(results, token)

        def search():
            q = query_var.get().strip()
            if not q:
                messagebox.showwarning("Web Search", "Enter a search term.", parent=win)
                return
            search_btn.configure(state="disabled")
            status.configure(text="Searching in background…")
            search_token["value"] += 1
            token = search_token["value"]
            threading.Thread(
                target=search_worker,
                args=(q, source_var.get(), token),
                daemon=True,
            ).start()

        def load_selected():
            item = selected["item"]
            if not item:
                messagebox.showwarning("Web Search", "Select an image first.", parent=win)
                return
            url = item.get("url") or item.get("thumbnail")
            if not url:
                messagebox.showerror("Web Search", "Selected result has no image URL.", parent=win)
                return

            status.configure(text="Downloading selected image in background…")
            load_btn.configure(state="disabled")
            cached = thumbnail_cache.get(id(item))

            def worker():
                try:
                    referer = "https://commons.wikimedia.org/" if "wikimedia.org" in url else None
                    try:
                        data, _ = self._download_url(url, timeout=15, retries=2, referer=referer)
                        image = self._load_image_bytes(data)
                        message = "Web image loaded successfully"
                    except Exception as full_exc:
                        if not cached:
                            raise full_exc
                        image = self._load_image_bytes(cached)
                        message = "Web image loaded using fast thumbnail fallback"
                    self.root.after(0, lambda: loaded(image, message))
                except Exception as exc:
                    self.root.after(0, lambda error=exc: load_failed(error))

            def loaded(image, message):
                if not win.winfo_exists():
                    return
                self.original_image = image.copy()
                self.current_image = image.copy()
                self.image_path = None
                self.history = []
                self.history_index = -1
                self.add_history(message)
                self.update_preview()
                self.refresh_dashboard()
                self.set_status(message + ".")
                win.destroy()

            def load_failed(exc):
                if not win.winfo_exists():
                    return
                load_btn.configure(state="normal")
                status.configure(text="Download failed — choose another result.")
                messagebox.showerror(
                    "Web Search",
                    "The selected image could not be downloaded.\n\n"
                    f"{exc}\n\nTry another result or use Image URL.",
                    parent=win,
                )

            threading.Thread(target=worker, daemon=True).start()

        search_btn = ttk.Button(top, text="Search", command=search)
        search_btn.pack(side="left", padx=(8, 0))
        entry.bind("<Return>", lambda e: search())
        load_btn = ttk.Button(bottom, text="Load Selected", command=load_selected)
        load_btn.pack(side="right")
        ttk.Button(bottom, text="Close", command=close_window).pack(side="right", padx=(0, 8))
        entry.focus_set()

    def save_direct(self):
        if not self.check_image():
            return

        # Ctrl+S always saves into the application's dedicated Screenshots folder.
        # The folder is created automatically if it does not exist.
        app_dir = os.path.dirname(os.path.abspath(__file__))
        screenshot_dir = os.path.join(app_dir, "Screenshots")
        os.makedirs(screenshot_dir, exist_ok=True)

        # Keep the source image name when available; web-loaded images use a clean default.
        if self.image_path:
            base = os.path.splitext(os.path.basename(self.image_path))[0]
            ext = os.path.splitext(self.image_path)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
                ext = ".png"
        else:
            base = "visionlab_result"
            ext = ".png"

        output = os.path.join(screenshot_dir, f"{base}_edited{ext}")

        # Never overwrite an existing screenshot.
        counter = 2
        candidate = output
        while os.path.exists(candidate):
            candidate = os.path.join(
                screenshot_dir, f"{base}_edited_{counter}{ext}"
            )
            counter += 1
        output = candidate

        if cv2.imwrite(output, self.current_image):
            self.set_status(f"Saved to Screenshots: {os.path.basename(output)}")
            messagebox.showinfo(
                "Saved",
                f"Image saved successfully.\n\nScreenshots folder:\n{output}"
            )
        else:
            messagebox.showerror("Save Error", "Could not save the image.")

    def save_as(self):
        if not self.check_image():
            return

        path = filedialog.asksaveasfilename(
            title="Save Image As",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tif"),
                ("WEBP", "*.webp")
            ]
        )

        if not path:
            return

        if cv2.imwrite(path, self.current_image):
            self.set_status(f"Saved as: {os.path.basename(path)}")
            messagebox.showinfo("Saved", "Image saved successfully.")
        else:
            messagebox.showerror("Save Error", "Could not save the image.")

    # ========================================================
    # HISTORY
    # ========================================================

    def add_history(self, operation):
        if self.current_image is None:
            return

        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]

        self.history.append(self.current_image.copy())
        self.history_index += 1
        self.last_operation = operation

        self.update_preview()
        self.refresh_dashboard()
        self.set_status(operation)

    def undo(self):
        if self.history_index <= 0:
            self.set_status("Nothing to undo.")
            return

        self.history_index -= 1
        self.current_image = self.history[self.history_index].copy()
        self.update_preview()
        self.set_status("Undo completed.")

    def redo(self):
        if self.history_index >= len(self.history) - 1:
            self.set_status("Nothing to redo.")
            return

        self.history_index += 1
        self.current_image = self.history[self.history_index].copy()
        self.update_preview()
        self.set_status("Redo completed.")

    def reset_image(self):
        if not self.check_image():
            return

        self.current_image = self.original_image.copy()
        self.history = [self.original_image.copy()]
        self.history_index = 0
        self.update_preview()
        self.set_status("Reset completed — original image restored.")

    # ========================================================
    # PREVIEW
    # ========================================================

    def update_preview(self):
        if self.original_image is None:
            return

        self.display_on_panel(
            self.original_image,
            self.original_panel,
            "Original resolution"
        )
        self.display_on_panel(
            self.current_image,
            self.processed_panel,
            self.processed_meta_text()
        )

    def display_on_panel(self, image, panel, meta_text):
        if image is None:
            return

        display = image.copy()

        if len(display.shape) == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)
        elif display.shape[2] == 3:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

        h, w = display.shape[:2]

        mode = self.preview_mode.get()

        if mode == "Fit":
            max_w = max(250, panel["label"].winfo_width() - 20)
            max_h = max(250, panel["label"].winfo_height() - 20)
            scale = min(max_w / w, max_h / h, 1.0)
        else:
            scale = int(mode[:-1]) / 100.0

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        display = cv2.resize(
            display,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        pil = Image.fromarray(display)
        photo = ImageTk.PhotoImage(pil)

        panel["label"].configure(image=photo, text="")
        panel["label"].image = photo
        panel["meta"].configure(text=meta_text)

    def processed_meta_text(self):
        if self.current_image is None:
            return "—"

        h, w = self.current_image.shape[:2]
        original_h, original_w = self.original_image.shape[:2]

        if (w, h) != (original_w, original_h):
            return (
                f"Actual: {w} × {h} px  |  "
                f"RESIZED from {original_w} × {original_h} px"
            )

        return f"Actual: {w} × {h} px  |  Preview scaling does not modify image"

    # ========================================================
    # DASHBOARD
    # ========================================================

    def refresh_dashboard(self):
        if not hasattr(self, "dashboard"):
            return

        text = "VISIONLAB PRO — DASHBOARD\n"
        text += "=" * 40 + "\n\n"

        if self.current_image is None:
            text += (
                "STATUS\n"
                "------\n"
                "No image loaded.\n\n"
                "Open an image to start.\n\n"
                "The preview is display-scaled only.\n"
                "Your actual image resolution is not\n"
                "changed unless you use Resize.\n\n"
                "SAVE BEHAVIOR\n"
                "-------------\n"
                "Save automatically stores files in\n"
                "the app's Screenshots folder.\n"
                "Your original is never overwritten.\n"
            )
        else:
            h, w = self.current_image.shape[:2]
            oh, ow = self.original_image.shape[:2]

            channels = (
                self.current_image.shape[2]
                if len(self.current_image.shape) == 3
                else 1
            )

            changed = (w, h) != (ow, oh)

            text += (
                f"FILE\n"
                f"----\n"
                f"{os.path.basename(self.image_path) if self.image_path else 'Untitled'}\n\n"
                f"CURRENT IMAGE\n"
                f"-------------\n"
                f"Resolution : {w} × {h} px\n"
                f"Channels   : {channels}\n"
                f"Data type  : {self.current_image.dtype}\n"
                f"Memory     : {self.current_image.nbytes:,} bytes\n"
                f"Pixels     : {w * h:,}\n\n"
                f"ORIGINAL\n"
                f"--------\n"
                f"{ow} × {oh} px\n\n"
                f"RESIZE STATE\n"
                f"------------\n"
                f"{'RESIZED' if changed else 'Original dimensions'}\n\n"
                f"HISTORY\n"
                f"-------\n"
                f"Step {self.history_index + 1} / {len(self.history)}\n\n"
                f"LAST OPERATION\n"
                f"--------------\n"
                f"{self.last_operation}\n\n"
                f"PREVIEW\n"
                f"-------\n"
                f"Display scaling only.\n"
                f"Preview mode: {self.preview_mode.get()}\n\n"
                f"SAVE\n"
                f"----\n"
                f"Ctrl+S = save to Screenshots\n"
                f"Ctrl+Shift+S = Save As\n"
            )

        self.dashboard.configure(state="normal")
        self.dashboard.delete("1.0", "end")
        self.dashboard.insert("end", text)
        self.dashboard.configure(state="disabled")

        if hasattr(self, "header_state"):
            if self.current_image is None:
                self.header_state.configure(text="NO IMAGE LOADED")
            else:
                self.header_state.configure(
                    text=f"READY • STEP {self.history_index + 1}"
                )

    # ========================================================
    # IMAGE INFORMATION
    # ========================================================

    def show_image_info(self):
        if not self.check_image():
            return

        h, w = self.current_image.shape[:2]
        channels = self.current_image.shape[2] if len(self.current_image.shape) == 3 else 1

        messagebox.showinfo(
            "Image Information",
            f"File: {os.path.basename(self.image_path) if self.image_path else 'Untitled'}\n\n"
            f"Current size: {w} × {h} pixels\n"
            f"Original size: {self.original_image.shape[1]} × {self.original_image.shape[0]} pixels\n"
            f"Channels: {channels}\n"
            f"Data type: {self.current_image.dtype}\n"
            f"Memory: {self.current_image.nbytes:,} bytes\n"
            f"Pixels: {w * h:,}\n\n"
            f"Last operation:\n{self.last_operation}"
        )

    # ========================================================
    # BASIC OPERATIONS
    # ========================================================

    def grayscale(self):
        if not self.check_image():
            return
        if len(self.current_image.shape) == 2:
            self.set_status("Image is already grayscale.")
            return
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        self.add_history("Grayscale conversion applied.")

    def resize_image(self):
        if not self.check_image():
            return

        value = simpledialog.askfloat(
            "Resize",
            "Enter percentage (e.g. 50 = half size, 150 = 1.5×):",
            parent=self.root,
            initialvalue=100,
            minvalue=1,
            maxvalue=1000
        )
        if value is None:
            return

        h, w = self.current_image.shape[:2]
        nw = max(1, int(w * value / 100))
        nh = max(1, int(h * value / 100))

        self.current_image = cv2.resize(
            self.current_image,
            (nw, nh),
            interpolation=cv2.INTER_AREA if value < 100 else cv2.INTER_CUBIC
        )
        self.add_history(
            f"RESIZED image to {nw} × {nh} px ({value:.0f}%)."
        )

    def crop_center(self):
        if not self.check_image():
            return

        value = simpledialog.askfloat(
            "Center Crop",
            "Keep what percentage of the image? (10–100)",
            parent=self.root,
            initialvalue=70,
            minvalue=10,
            maxvalue=100
        )
        if value is None:
            return

        h, w = self.current_image.shape[:2]
        cw = int(w * value / 100)
        ch = int(h * value / 100)

        x1 = (w - cw) // 2
        y1 = (h - ch) // 2

        self.current_image = self.current_image[
            y1:y1 + ch, x1:x1 + cw
        ]
        self.add_history(f"Center crop applied — {cw} × {ch} px.")

    def rotate_90(self):
        if not self.check_image():
            return
        self.current_image = cv2.rotate(
            self.current_image, cv2.ROTATE_90_CLOCKWISE
        )
        self.add_history("Rotated 90° clockwise.")

    def rotate_180(self):
        if not self.check_image():
            return
        self.current_image = cv2.rotate(
            self.current_image, cv2.ROTATE_180
        )
        self.add_history("Rotated 180°.")

    def flip_horizontal(self):
        if not self.check_image():
            return
        self.current_image = cv2.flip(self.current_image, 1)
        self.add_history("Horizontal flip applied.")

    def flip_vertical(self):
        if not self.check_image():
            return
        self.current_image = cv2.flip(self.current_image, 0)
        self.add_history("Vertical flip applied.")

    # ========================================================
    # LIGHT & COLOR
    # ========================================================

    def brightness(self):
        if not self.check_image():
            return

        value = simpledialog.askinteger(
            "Brightness",
            "Brightness (-100 to +100):",
            parent=self.root,
            initialvalue=20,
            minvalue=-100,
            maxvalue=100
        )
        if value is None:
            return

        self.current_image = cv2.convertScaleAbs(
            self.current_image, alpha=1.0, beta=value
        )
        self.add_history(f"Brightness changed by {value:+d}.")

    def contrast(self):
        if not self.check_image():
            return

        value = simpledialog.askfloat(
            "Contrast",
            "Contrast (0.2 to 3.0):",
            parent=self.root,
            initialvalue=1.25,
            minvalue=0.2,
            maxvalue=3.0
        )
        if value is None:
            return

        self.current_image = cv2.convertScaleAbs(
            self.current_image, alpha=value, beta=0
        )
        self.add_history(f"Contrast set to {value:.2f}×.")

    def auto_contrast(self):
        if not self.check_image():
            return

        if len(self.current_image.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            self.current_image = clahe.apply(self.current_image)
        else:
            lab = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            self.current_image = cv2.cvtColor(
                cv2.merge((l, a, b)),
                cv2.COLOR_LAB2BGR
            )

        self.add_history("Auto contrast / CLAHE applied.")

    def saturation(self):
        if not self.check_image():
            return

        if len(self.current_image.shape) != 3:
            messagebox.showinfo(
                "Saturation",
                "Saturation requires a color image."
            )
            return

        value = simpledialog.askfloat(
            "Saturation",
            "Saturation multiplier (0 = gray, 1 = original, 2 = double):",
            parent=self.root,
            initialvalue=1.3,
            minvalue=0,
            maxvalue=4
        )
        if value is None:
            return

        hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * value, 0, 255)
        hsv = hsv.astype(np.uint8)
        self.current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        self.add_history(f"Saturation adjusted to {value:.2f}×.")

    def sepia(self):
        if not self.check_image():
            return

        if len(self.current_image.shape) != 3:
            messagebox.showinfo("Sepia", "Sepia requires a color image.")
            return

        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])

        self.current_image = cv2.transform(
            self.current_image, kernel
        )
        self.current_image = np.clip(
            self.current_image, 0, 255
        ).astype(np.uint8)

        self.add_history("Sepia tone applied.")

    def negative(self):
        if not self.check_image():
            return

        self.current_image = cv2.bitwise_not(self.current_image)
        self.add_history("Negative / color inversion applied.")

    # ========================================================
    # FILTERS
    # ========================================================

    def gaussian_blur(self):
        if not self.check_image():
            return

        self.current_image = cv2.GaussianBlur(
            self.current_image, (7, 7), 0
        )
        self.add_history("Gaussian blur applied.")

    def median_blur(self):
        if not self.check_image():
            return

        self.current_image = cv2.medianBlur(
            self.current_image, 5
        )
        self.add_history("Median blur applied.")

    def sharpen(self):
        if not self.check_image():
            return

        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)

        self.current_image = cv2.filter2D(
            self.current_image, -1, kernel
        )
        self.add_history("Sharpening filter applied.")

    def noise_reduction(self):
        if not self.check_image():
            return

        if len(self.current_image.shape) == 3:
            self.current_image = cv2.fastNlMeansDenoisingColored(
                self.current_image, None, 10, 10, 7, 21
            )
        else:
            self.current_image = cv2.fastNlMeansDenoising(
                self.current_image, None, 10, 7, 21
            )

        self.add_history("Noise reduction applied.")

    # ========================================================
    # CV ANALYSIS
    # ========================================================

    def get_gray(self):
        if len(self.current_image.shape) == 2:
            return self.current_image
        return cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

    def edge_detection(self):
        if not self.check_image():
            return

        gray = self.get_gray()
        self.current_image = cv2.Canny(gray, 100, 200)
        self.add_history("Canny edge detection applied.")

    def threshold(self):
        if not self.check_image():
            return

        gray = self.get_gray()
        _, result = cv2.threshold(
            gray, 127, 255, cv2.THRESH_BINARY
        )
        self.current_image = result
        self.add_history("Binary threshold applied at 127.")

    def adaptive_threshold(self):
        if not self.check_image():
            return

        gray = self.get_gray()
        result = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        self.current_image = result
        self.add_history("Adaptive threshold applied.")

    def color_analysis(self):
        if not self.check_image():
            return

        if len(self.current_image.shape) != 3:
            messagebox.showinfo(
                "HSV Analysis",
                "HSV analysis requires a color image."
            )
            return

        hsv = cv2.cvtColor(
            self.current_image, cv2.COLOR_BGR2HSV
        )

        h, s, v = cv2.split(hsv)

        messagebox.showinfo(
            "HSV Color Analysis",
            f"Average Hue        : {np.mean(h):.2f}\n"
            f"Average Saturation : {np.mean(s):.2f}\n"
            f"Average Value      : {np.mean(v):.2f}\n\n"
            f"Minimum Hue        : {np.min(h)}\n"
            f"Maximum Hue        : {np.max(h)}"
        )

        self.set_status("HSV color analysis completed.")

    def histogram(self):
        if not self.check_image():
            return

        win = tk.Toplevel(self.root)
        win.title("Histogram Analysis")
        win.geometry("900x620")
        win.minsize(700, 500)

        canvas = tk.Canvas(win, bg="white")
        canvas.pack(fill="both", expand=True, padx=10, pady=10)

        image = self.current_image

        if len(image.shape) == 2:
            histograms = [
                ("Gray", cv2.calcHist([image], [0], None, [256], [0, 256]))
            ]
        else:
            b, g, r = cv2.split(image)
            histograms = [
                ("Blue", cv2.calcHist([b], [0], None, [256], [0, 256])),
                ("Green", cv2.calcHist([g], [0], None, [256], [0, 256])),
                ("Red", cv2.calcHist([r], [0], None, [256], [0, 256]))
            ]

        def redraw(event=None):
            canvas.delete("all")

            width = max(canvas.winfo_width(), 700)
            height = max(canvas.winfo_height(), 450)

            left = 55
            bottom = height - 45
            right = width - 25
            top = 25

            canvas.create_line(left, bottom, right, bottom)
            canvas.create_line(left, top, left, bottom)

            max_value = max(float(np.max(h)) for _, h in histograms)
            max_value = max(max_value, 1.0)

            for name, hist in histograms:
                points = []
                for x in range(256):
                    px = left + x * (right - left) / 255
                    py = bottom - (
                        float(hist[x][0]) / max_value
                    ) * (bottom - top)
                    points.extend([px, py])

                canvas.create_line(*points, smooth=False)

            for tick in [0, 64, 128, 192, 255]:
                x = left + tick * (right - left) / 255
                canvas.create_line(x, bottom, x, bottom + 5)
                canvas.create_text(x, bottom + 18, text=str(tick))

            canvas.create_text(
                (left + right) / 2,
                height - 12,
                text="Pixel Intensity (0–255)"
            )
            canvas.create_text(
                15,
                (top + bottom) / 2,
                text="Frequency",
                angle=90
            )

            legend = "   ".join(name for name, _ in histograms)
            canvas.create_text(
                width - 120,
                20,
                text=legend
            )

        canvas.bind("<Configure>", redraw)
        redraw()

        self.set_status("Histogram analysis opened.")

    def face_detection(self):
        """Detect faces without permanently modifying the processed image.

        Green detection boxes are shown only temporarily while the result
        dialog is displayed. After the dialog closes, the exact image that
        existed before Face Detection is restored, so no green marks are
        saved, added to history, or carried into later operations.
        """
        if not self.check_image():
            return

        # Keep the user's current image completely untouched.
        source_image = self.current_image.copy()

        if len(source_image.shape) == 2:
            gray = source_image.copy()
            display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            if source_image.shape[2] == 4:
                display_img = cv2.cvtColor(source_image, cv2.COLOR_BGRA2BGR)
            else:
                display_img = source_image.copy()
            gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)

        gray = cv2.equalizeHist(gray)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)

        if face_cascade.empty():
            messagebox.showerror("Face Detection", "OpenCV face detector could not be loaded.")
            return

        h, w = gray.shape[:2]
        min_side = max(45, int(min(w, h) * 0.09))
        detections = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=10,
            minSize=(min_side, min_side),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        def iou(a, b):
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            x1, y1 = max(ax, bx), max(ay, by)
            x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            union = aw * ah + bw * bh - inter
            return inter / union if union else 0.0

        detections = sorted(
            [tuple(map(int, f)) for f in detections],
            key=lambda f: f[2] * f[3],
            reverse=True
        )

        filtered = []
        for face in detections:
            if any(iou(face, kept) > 0.35 for kept in filtered):
                continue
            fw, fh = face[2], face[3]
            ratio = fw / max(fh, 1)
            if 0.65 <= ratio <= 1.45:
                filtered.append(face)

        # Draw boxes only on a temporary preview copy.
        preview_img = display_img.copy()
        for index, (x, y, fw, fh) in enumerate(filtered, 1):
            cv2.rectangle(
                preview_img,
                (x, y),
                (x + fw, y + fh),
                (0, 255, 0),
                3
            )
            cv2.putText(
                preview_img,
                f"Face {index}",
                (x, max(28, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        count = len(filtered)
        self.set_status(f"Face detection completed — {count} face(s) detected.")

        # Show the detection result temporarily, but DO NOT add it to history.
        self.current_image = preview_img
        self.update_preview()

        if count == 0:
            msg = "No face detected.\n\nTry a clear, front-facing photo with good lighting."
        elif count == 1:
            msg = "1 face detected successfully."
        else:
            msg = f"{count} faces detected."

        messagebox.showinfo("Face Detection", msg)

        # Restore the exact pre-detection image and remove all green marks.
        self.current_image = source_image
        self.update_preview()
        self.set_status(
            f"Face detection completed — {count} face(s) detected. Green marks removed."
        )

    def colorize_image(self):
        """Apply stable pseudo-colorization to the current image.

        Works after ANY previous operation, even when the image is still BGR.
        OpenCV colormaps operate on an 8-bit grayscale intensity image; for a
        color input we first convert it to luminance, then map that luminance
        to color. This is false/pseudo color, not recovery of the original
        real-world colors.
        """
        if not self.check_image():
            return

        try:
            if len(self.current_image.shape) == 2:
                gray = self.current_image.copy()
            else:
                if self.current_image.shape[2] == 4:
                    bgr = cv2.cvtColor(self.current_image, cv2.COLOR_BGRA2BGR)
                else:
                    bgr = self.current_image
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            if gray.dtype != np.uint8:
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Improve tonal separation before mapping, especially after
            # brightness/contrast/filter operations.
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

            self.current_image = colored
            self.add_history("Colorize applied (Turbo pseudo-color).")
            self.set_status("Colorized successfully — luminance mapped to Turbo colors.")
        except Exception as exc:
            messagebox.showerror("Colorize Image", f"Could not colorize the image.\n\n{exc}")


    # ========================================================
    # UTILITY
    # ========================================================

    def check_image(self):
        if self.current_image is None:
            messagebox.showwarning(
                "No Image",
                "Please open an image first."
            )
            return False
        return True

    def show_about(self):
        messagebox.showinfo(
            "About VisionLab Pro",
            "VISIONLAB PRO\n\n"
            "Interactive Computer Vision & Image Processing Studio\n\n"
            "Features include:\n"
            "• Non-destructive original preview\n"
            "• Direct safe save\n"
            "• Undo / Redo history\n"
            "• 25+ image processing operations\n"
            "• Histogram analysis\n"
            "• HSV analysis\n"
            "• Face detection\n"
            "• Live image dashboard\n"
            "• Keyboard shortcuts\n\n"
            "Preview scaling never changes the actual image.\n"
            "Only the Resize tool changes image dimensions."
        )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = VisionLabPro(root)
    root.mainloop()
