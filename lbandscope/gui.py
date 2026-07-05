"""Desktop application window.

A worker thread runs the decoder and computes spectra, passing results to the UI
through a queue. The window presents three views: decoded messages, a spectrum
waterfall, and a map of reported positions.
"""
from __future__ import annotations

import collections
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from . import __version__, demo, dsp, frontend, messages, presets, sdr, spectrum

DEMO_RADIO = "Demo (no radio required)"
NFFT = 256
WATERFALL_ROWS = 220

SETUP_HELP = """\
Using a real receiver

Hardware
  - An RTL-SDR USB receiver (the RTL-SDR Blog V3 is a common choice).
  - An L-band patch antenna, often sold together with the receiver.

One-time setup (about ten minutes)
  1. Connect the receiver to a USB port.
  2. Install the USB driver with Zadig (https://zadig.akeo.ie):
     Options > List All Devices, select "Bulk-In, Interface (Interface 0)",
     choose WinUSB, and click Replace Driver.
  3. Install receiver support so this application can see the device:
        pip install pyrtlsdr
  4. Place the patch antenna with a clear view of the sky toward the Inmarsat
     satellite for your region. It is geostationary and does not need tracking.

Then
  - Click Refresh, select the receiver, choose a channel, and click Start.

Until a receiver is connected, Demo mode reproduces the full experience.
"""

ABOUT = f"""\
LBandScope {__version__}

A decoder and monitor for Inmarsat L-band signals.

Free, open-source software under the GNU General Public License v3.

An independent implementation. For a more complete decoder see InmarScope by
Sarah Rose Lives, and JAERO by Jonti, whose work in this field this project
builds on in spirit.
"""


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"LBandScope {__version__}")
        self.geometry("880x620")
        self.minsize(760, 540)

        self.q: queue.Queue = queue.Queue()
        self.worker = None
        self.stop_flag = threading.Event()
        self.devices = []
        self.count = 0
        self.records = []
        self.wf_rows = collections.deque(maxlen=WATERFALL_ROWS)
        self._wf_img = None
        self._rec_file = None
        self._last_row = None
        self._find_fs = 2.048e6

        self._build_menu()
        self._build()
        self.refresh_devices()
        self.after(100, self._drain)

    # -- construction -----------------------------------------------------
    def _build_menu(self):
        m = tk.Menu(self)
        filem = tk.Menu(m, tearoff=0)
        filem.add_command(label="Export data (CSV)...", command=lambda: self.export("csv"))
        filem.add_command(label="Export map (KML)...", command=lambda: self.export("kml"))
        filem.add_command(label="Export map (GeoJSON)...", command=lambda: self.export("geojson"))
        filem.add_separator()
        filem.add_command(label="Exit", command=self.destroy)
        m.add_cascade(label="File", menu=filem)
        helpm = tk.Menu(m, tearoff=0)
        helpm.add_command(label="Setup a receiver", command=self.help)
        helpm.add_command(label="About", command=self.about)
        m.add_cascade(label="Help", menu=helpm)
        self.config(menu=m)

    def _build(self):
        top = ttk.Frame(self, padding=(12, 8))
        top.pack(fill="x")
        ttk.Label(top, text="LBandScope", font=("Segoe UI", 16, "bold")).pack(side="left")
        self.state_dot = tk.Label(top, text="●", fg="#c0392b")
        self.state_dot.pack(side="right")
        self.status = tk.StringVar(value="Checking for receivers...")
        ttk.Label(top, textvariable=self.status).pack(side="right", padx=8)

        ctrl = ttk.Frame(self, padding=(12, 2))
        ctrl.pack(fill="x")
        ttk.Label(ctrl, text="Channel").grid(row=0, column=0, sticky="w", pady=3)
        self.channel = ttk.Combobox(ctrl, values=presets.names(), state="readonly", width=34)
        self.channel.current(0)
        self.channel.grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(ctrl, text="Receiver").grid(row=1, column=0, sticky="w", pady=3)
        self.device = ttk.Combobox(ctrl, state="readonly", width=34)
        self.device.grid(row=1, column=1, sticky="w", padx=8)
        ttk.Button(ctrl, text="Refresh", command=self.refresh_devices).grid(row=1, column=2, padx=4)

        self.clean = tk.BooleanVar(value=True)
        self.bias = tk.BooleanVar(value=True)
        adv = ttk.Frame(ctrl)
        adv.grid(row=0, column=2, rowspan=2, padx=(16, 0), sticky="w")
        ttk.Checkbutton(adv, text="Signal cleanup", variable=self.clean).pack(anchor="w")
        ttk.Checkbutton(adv, text="Antenna power (bias-tee)", variable=self.bias).pack(anchor="w")

        self.start_btn = tk.Button(self, text="Start", height=2, bg="#1e7e34", fg="white",
                                   font=("Segoe UI", 12, "bold"), command=self.toggle)
        self.start_btn.pack(fill="x", padx=12, pady=(6, 4))

        meter = ttk.Frame(self, padding=(12, 0))
        meter.pack(fill="x")
        ttk.Label(meter, text="Signal", width=8).grid(row=0, column=0, sticky="w")
        self.level_bar = ttk.Progressbar(meter, length=180, maximum=100)
        self.level_bar.grid(row=0, column=1, padx=6)
        self.level_txt = tk.StringVar(value="--")
        ttk.Label(meter, textvariable=self.level_txt, width=10).grid(row=0, column=2)
        ttk.Label(meter, text="Quality", width=8).grid(row=0, column=3, sticky="w", padx=(16, 0))
        self.qual_bar = ttk.Progressbar(meter, length=180, maximum=100)
        self.qual_bar.grid(row=0, column=4, padx=6)
        self.qual_txt = tk.StringVar(value="--")
        ttk.Label(meter, textvariable=self.qual_txt, width=10).grid(row=0, column=5)

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=12, pady=4)
        self._build_messages(nb)
        self._build_spectrum(nb)
        self._build_constellation(nb)
        self._build_map(nb)

        foot = ttk.Frame(self, padding=(12, 6))
        foot.pack(fill="x")
        ttk.Button(foot, text="Clear", command=self.clear).pack(side="left")
        ttk.Button(foot, text="Find signal", command=self.find_signal).pack(side="left", padx=6)
        self.rec_btn = ttk.Button(foot, text="Record IQ", command=self.toggle_record)
        self.rec_btn.pack(side="left")
        self.counter = tk.StringVar(value="0 messages")
        ttk.Label(foot, textvariable=self.counter).pack(side="right")

    def _build_messages(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Messages")
        cols = ("time", "type", "id", "message")
        self.tree = ttk.Treeview(tab, columns=cols, show="headings")
        for c, w in (("time", 80), ("type", 70), ("id", 90), ("message", 560)):
            self.tree.heading(c, text=c.title())
            self.tree.column(c, width=w, anchor="w", stretch=(c == "message"))
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(tab, command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)

    def _build_spectrum(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Spectrum")
        self.wf_canvas = tk.Canvas(tab, bg="#101014", highlightthickness=0)
        self.wf_canvas.pack(fill="both", expand=True)

    def _build_constellation(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Constellation")
        self.const_canvas = tk.Canvas(tab, bg="#0a0e12", highlightthickness=0)
        self.const_canvas.pack(fill="both", expand=True)

    def _build_map(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Map")
        self.map_canvas = tk.Canvas(tab, bg="#0b1a2b", highlightthickness=0)
        self.map_canvas.pack(fill="both", expand=True)
        self.map_canvas.bind("<Configure>", lambda e: self._render_map())

    # -- devices ----------------------------------------------------------
    def refresh_devices(self):
        env = sdr.check_environment()
        self.status.set(env["summary"])
        self.state_dot.config(fg="#1e7e34" if env["state"] == "ready" else "#c0392b")
        self.devices = env["devices"]
        self.device["values"] = [DEMO_RADIO] + [d["label"] for d in self.devices]
        self.device.current(0)

    def _selected_device(self):
        i = self.device.current()
        return None if i <= 0 else self.devices[i - 1]

    # -- run control ------------------------------------------------------
    def toggle(self):
        if self.worker and self.worker.is_alive():
            self.stop_flag.set()
            self.start_btn.config(text="Stopping...", state="disabled")
            return
        preset = presets.by_name(self.channel.get())
        device = None if preset["kind"] == "demo" else self._selected_device()
        clean, bias = self.clean.get(), self.bias.get()
        self._find_fs = preset.get("rate", 2.048e6)
        self.stop_flag.clear()
        self.worker = threading.Thread(target=self._run, args=(preset, device, clean, bias),
                                       daemon=True)
        self.worker.start()
        self.start_btn.config(text="Stop", bg="#a11")

    def _run(self, preset, device, clean, bias):
        try:
            if device is None:
                self.q.put(("log", "Demo mode."))
                source, sps, live = demo.demo_iq_blocks(n_blocks=10 ** 9), 8, True
            else:
                self.q.put(("log", f"Receiving on {device['label']}."))
                cfg = {"freq": preset["freq"], "rate": preset["rate"], "bias_tee": bias}
                source, sps, live = sdr.stream(device, cfg), preset.get("sps", 8), False
            for block in source:
                if self.stop_flag.is_set():
                    break
                if clean:
                    block = frontend.precondition(block)
                rf = self._rec_file
                if rf is not None:
                    inter = np.empty(block.size * 2, np.float32)
                    inter[0::2], inter[1::2] = block.real, block.imag
                    try:
                        rf.write(inter.tobytes())
                    except Exception:
                        pass
                row = spectrum.spectrum_db(block, NFFT)
                self.q.put(("spectrum", row))
                self.q.put(("meter", (frontend.signal_level_db(block),
                                      frontend.quality_db(row))))
                for f in dsp.decode_frames(block, sps, with_symbols=True):
                    self.q.put(("msg", f["payload"]))
                    if "symbols" in f:
                        s = f["symbols"]
                        if len(s) > 500:
                            s = s[np.linspace(0, len(s) - 1, 500).astype(int)]
                        self.q.put(("const", s))
                if live:
                    time.sleep(0.7)
        except Exception as e:
            self.q.put(("error", str(e)))
        finally:
            self.q.put(("stopped", None))

    # -- UI pump ----------------------------------------------------------
    def _drain(self):
        dirty_wf = False
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == "msg":
                    self._add_message(data)
                elif kind == "spectrum":
                    self.wf_rows.append(data)
                    self._last_row = data
                    dirty_wf = True
                elif kind == "meter":
                    self._update_meter(*data)
                elif kind == "const":
                    self._render_constellation(data)
                elif kind == "log":
                    self.status.set(data)
                elif kind == "error":
                    self.status.set("Stopped.")
                    messagebox.showerror("Receiver error",
                                         f"{data}\n\nSee Help > Setup a receiver, or use Demo mode.")
                elif kind == "stopped":
                    self.start_btn.config(text="Start", bg="#1e7e34", state="normal")
        except queue.Empty:
            pass
        if dirty_wf:
            self._render_waterfall()
        self.after(100, self._drain)

    def _add_message(self, payload: bytes):
        self.count += 1
        text = payload.decode("latin-1")
        rec = messages.make_record(text, time.strftime("%H:%M:%S"))
        self.records.append(rec)
        self.tree.insert("", "end", values=(rec["time"], rec["kind"], rec["id"], text))
        self.tree.yview_moveto(1.0)
        self.counter.set(f"{self.count} messages")
        if rec["lat"] is not None:
            self._render_map()

    def _update_meter(self, level_db, quality_db):
        self.level_bar["value"] = max(0, min(100, (level_db + 60) / 60 * 100))
        self.level_txt.set(f"{level_db:.0f} dBFS")
        self.qual_bar["value"] = max(0, min(100, quality_db / 40 * 100))
        self.qual_txt.set(f"{quality_db:.0f} dB")

    def find_signal(self):
        if self._last_row is None:
            self.status.set("Start receiving first, then Find signal.")
            return
        off, prom = frontend.find_peak_offset(self._last_row, self._find_fs)
        self.status.set(f"Strongest signal: {off / 1e3:+.1f} kHz from center, "
                        f"{prom:.0f} dB above noise.")

    def toggle_record(self):
        if self._rec_file is not None:
            f, self._rec_file = self._rec_file, None
            try:
                f.close()
            except Exception:
                pass
            self.rec_btn.config(text="Record IQ")
            self.status.set("Recording stopped.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".cf32",
                                            filetypes=[("Complex float32 IQ", "*.cf32")])
        if not path:
            return
        self._rec_file = open(path, "wb")
        self.rec_btn.config(text="Stop recording")
        self.status.set("Recording IQ (cf32)...")

    # -- rendering --------------------------------------------------------
    def _render_constellation(self, pts):
        c = self.const_canvas
        w, h = max(c.winfo_width(), 64), max(c.winfo_height(), 64)
        c.delete("all")
        cx, cy, r = w / 2, h / 2, min(w, h) * 0.4
        c.create_line(0, cy, w, cy, fill="#1b2733")
        c.create_line(cx, 0, cx, h, fill="#1b2733")
        pts = np.asarray(pts)
        if len(pts) == 0:
            return
        m = float(np.median(np.abs(pts))) + 1e-9
        for p in pts:
            x = cx + (p.real / m) * r * 0.5
            y = cy - (p.imag / m) * r * 0.5
            c.create_oval(x - 1.5, y - 1.5, x + 1.5, y + 1.5, fill="#5dade2", outline="")

    def _render_waterfall(self):
        if not self.wf_rows:
            return
        w = max(self.wf_canvas.winfo_width(), 64)
        h = min(len(self.wf_rows), max(self.wf_canvas.winfo_height(), 64))
        stack = np.array(list(self.wf_rows)[-h:])
        xs = np.linspace(0, NFFT - 1, w)
        res = np.stack([np.interp(xs, np.arange(NFFT), row) for row in stack])
        ppm = spectrum.to_ppm(spectrum.colorize(res))
        self._wf_img = tk.PhotoImage(data=ppm, format="ppm")
        self.wf_canvas.delete("all")
        self.wf_canvas.create_image(0, 0, anchor="nw", image=self._wf_img)

    def _render_map(self):
        c = self.map_canvas
        w, h = max(c.winfo_width(), 64), max(c.winfo_height(), 64)
        c.delete("all")
        pts = [(r["lon"], r["lat"], r) for r in self.records if r["lat"] is not None]
        if pts:
            lons = [p[0] for p in pts]
            lats = [p[1] for p in pts]
            lon0, lon1 = min(lons) - 3, max(lons) + 3
            lat0, lat1 = min(lats) - 3, max(lats) + 3
        else:
            lon0, lon1, lat0, lat1 = -180, 180, -80, 80
        if lon1 - lon0 < 1:
            lon0, lon1 = lon0 - 5, lon1 + 5
        if lat1 - lat0 < 1:
            lat0, lat1 = lat0 - 5, lat1 + 5

        def to_xy(lon, lat):
            return ((lon - lon0) / (lon1 - lon0) * w,
                    (lat1 - lat) / (lat1 - lat0) * h)

        step = 10 if (lon1 - lon0) < 60 else 30
        for g in range(-180, 181, step):
            if lon0 <= g <= lon1:
                x, _ = to_xy(g, lat0)
                c.create_line(x, 0, x, h, fill="#16324a")
                c.create_text(x + 2, h - 8, anchor="w", fill="#4f7ba3", text=f"{g}°", font=("Segoe UI", 7))
        for g in range(-90, 91, step):
            if lat0 <= g <= lat1:
                _, y = to_xy(lon0, g)
                c.create_line(0, y, w, y, fill="#16324a")
                c.create_text(4, y + 8, anchor="w", fill="#4f7ba3", text=f"{g}°", font=("Segoe UI", 7))
        for lon, lat, r in pts:
            x, y = to_xy(lon, lat)
            col = "#f4d03f" if r["kind"] == "Aero" else "#5dade2"
            c.create_oval(x - 4, y - 4, x + 4, y + 4, fill=col, outline="white")
            if r["id"]:
                c.create_text(x + 7, y, anchor="w", fill="white", text=r["id"], font=("Segoe UI", 8))

    # -- misc -------------------------------------------------------------
    def clear(self):
        self.tree.delete(*self.tree.get_children())
        self.records.clear()
        self.count = 0
        self.counter.set("0 messages")
        self._render_map()

    def export(self, fmt):
        if not self.records:
            messagebox.showinfo("Export", "No messages to export yet.")
            return
        ext = {"csv": ".csv", "kml": ".kml", "geojson": ".geojson"}[fmt]
        path = filedialog.asksaveasfilename(defaultextension=ext,
                                            filetypes=[(fmt.upper(), "*" + ext)])
        if not path:
            return
        body = {"csv": messages.to_csv, "kml": messages.to_kml,
                "geojson": messages.to_geojson}[fmt](self.records)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)

    def help(self):
        self._text_window("Setup a receiver", SETUP_HELP, 640, 520)

    def about(self):
        self._text_window("About LBandScope", ABOUT, 520, 260)

    def _text_window(self, title, body, w, h):
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry(f"{w}x{h}")
        t = tk.Text(win, wrap="word", font=("Segoe UI", 10), padx=12, pady=12)
        t.insert("1.0", body)
        t.config(state="disabled")
        t.pack(fill="both", expand=True)


def main():
    App().mainloop()
    return 0


if __name__ == "__main__":
    main()
