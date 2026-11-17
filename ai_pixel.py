import sys
from pathlib import Path

import numpy as np
from PIL import (
    Image, ImageEnhance, ImageFilter, ImageOps,
    ImageDraw, ImageFont
)

from PySide6.QtCore import Qt, QRect, Signal, QPoint
from PySide6.QtGui import (
    QPixmap, QPainter, QPen, QColor, QFont, QIcon
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QSlider, QComboBox,
    QFileDialog, QMessageBox, QInputDialog, QColorDialog,
    QScrollArea, QFrame, QStackedWidget, QCheckBox
)

try:
    import cv2
except ImportError:
    cv2 = None


# =========================================================
# IMAGE VIEW
# =========================================================

class ImageView(QLabel):
    selectionFinished = Signal(object)
    brushFinished = Signal(object)
    imageClicked = Signal(object)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(500, 400)
        self.setStyleSheet("background:#202020;")
        self.setMouseTracking(True)

        self.image = None
        self.display_rect = QRect()
        self.mode = "normal"
        self.start_point = None
        self.end_point = None
        self.brush_points = []
        self.selections = []

    def set_image(self, image):
        self.image = image
        self.update()

    def set_mode(self, mode):
        self.mode = mode
        self.start_point = None
        self.end_point = None
        self.brush_points = []
        self.update()

    def image_coordinates(self, point):
        if self.image is None or self.display_rect.isNull():
            return None

        x = point.x() - self.display_rect.x()
        y = point.y() - self.display_rect.y()

        if x < 0 or y < 0:
            return None
        if x >= self.display_rect.width() or y >= self.display_rect.height():
            return None

        ix = int(x * self.image.width / self.display_rect.width())
        iy = int(y * self.image.height / self.display_rect.height())

        return (
            max(0, min(ix, self.image.width - 1)),
            max(0, min(iy, self.image.height - 1))
        )

    def mousePressEvent(self, event):
        if self.image is None:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self.start_point = event.position().toPoint()
            self.end_point = self.start_point

            if self.mode == "healing":
                self.brush_points = [self.image_coordinates(self.start_point)]
            elif self.mode == "normal":
                pos = self.image_coordinates(self.start_point)
                if pos:
                    self.imageClicked.emit(pos)

            self.update()

    def mouseMoveEvent(self, event):
        if self.start_point is None:
            return

        self.end_point = event.position().toPoint()

        if self.mode == "healing":
            pos = self.image_coordinates(self.end_point)
            if pos:
                self.brush_points.append(pos)

        self.update()

    def mouseReleaseEvent(self, event):
        if self.start_point is None:
            return

        self.end_point = event.position().toPoint()

        if self.mode == "selective":
            p1 = self.image_coordinates(self.start_point)
            p2 = self.image_coordinates(self.end_point)

            if p1 and p2:
                x1, y1 = p1
                x2, y2 = p2
                rect = (
                    min(x1, x2), min(y1, y2),
                    max(x1, x2), max(y1, y2)
                )
                if rect[2] - rect[0] > 5 and rect[3] - rect[1] > 5:
                    self.selections.append(rect)
                    self.selectionFinished.emit(rect)

        elif self.mode == "healing":
            self.brushFinished.emit(self.brush_points)

        self.start_point = None
        self.end_point = None
        self.brush_points = []
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.image is None:
            painter = QPainter(self)
            painter.setPen(QColor("white"))
            painter.setFont(QFont("Arial", 16))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                "AI Pixel Studio\n\nClick Add Photo to start"
            )
            return

        qimage = self.image.convert("RGB")
        data = qimage.tobytes("raw", "RGB")

        from PySide6.QtGui import QImage

        qt_image = QImage(
            data, qimage.width, qimage.height,
            qimage.width * 3, QImage.Format.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(qt_image)

        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        self.display_rect = QRect(x, y, scaled.width(), scaled.height())

        painter = QPainter(self)
        painter.drawPixmap(x, y, scaled)

        # Selective selection preview
        if self.selections:
            sx = scaled.width() / self.image.width
            sy = scaled.height() / self.image.height

            painter.setPen(QPen(QColor("#00d9ff"), 2))
            for x1, y1, x2, y2 in self.selections:
                painter.drawRect(QRect(
                    int(x + x1 * sx),
                    int(y + y1 * sy),
                    int((x2 - x1) * sx),
                    int((y2 - y1) * sy)
                ))

        # Current drag selection
        if self.start_point and self.mode == "selective":
            painter.setPen(QPen(QColor("#00ff99"), 2))
            painter.drawRect(QRect(
                self.start_point,
                self.end_point or self.start_point
            ))

        # Healing brush preview
        if self.start_point and self.mode == "healing":
            painter.setPen(QPen(QColor("#ff5555"), 3))
            for i in range(1, len(self.brush_points)):
                p1 = self.brush_points[i - 1]
                p2 = self.brush_points[i]
                if p1 and p2:
                    sx = scaled.width() / self.image.width
                    sy = scaled.height() / self.image.height
                    painter.drawLine(
                        QPoint(int(x + p1[0] * sx), int(y + p1[1] * sy)),
                        QPoint(int(x + p2[0] * sx), int(y + p2[1] * sy))
                    )

        painter.end()


# =========================================================
# MAIN APPLICATION
# =========================================================

class AIPixelStudio(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Pixel Studio")
        self.resize(1400, 850)

        self.original = None
        self.current = None
        self.before = None

        self.undo_stack = []
        self.redo_stack = []

        self.theme = "black"
        self.active_tool = "Adjust"
        self.selective_values = {}
        self.text_color = "#ffffff"
        self.text_position = None

        self.init_ui()
        self.apply_theme()

    # -----------------------------------------------------
    # UI
    # -----------------------------------------------------

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # TOP TOOLBAR
        top = QHBoxLayout()

        self.logo = QLabel("✦ AI PIXEL STUDIO")
        self.logo.setStyleSheet("font-size:20px;font-weight:bold;")

        top.addWidget(self.logo)
        top.addStretch()

        self.add_btn = self.top_button("＋ Add Photo", self.open_image)
        self.undo_btn = self.top_button("↶ Undo", self.undo)
        self.redo_btn = self.top_button("↷ Redo", self.redo)
        self.reset_btn = self.top_button("⟲ Reset", self.reset_image)
        self.save_btn = self.top_button("⬇ Save", self.save_image)
        self.theme_btn = self.top_button("◐ Theme", self.toggle_theme)

        for button in [
            self.add_btn, self.undo_btn, self.redo_btn,
            self.reset_btn, self.save_btn, self.theme_btn
        ]:
            top.addWidget(button)

        main_layout.addLayout(top)

        # MAIN CONTENT
        content = QHBoxLayout()
        content.setSpacing(10)

        # LEFT PANEL
        left = QFrame()
        left.setFixedWidth(205)
        self.left_layout = QVBoxLayout(left)

        self.category_buttons = {}
        categories = ["ALL", "REFINE", "FIX", "STYLE"]

        for category in categories:
            btn = QPushButton(category)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(
                lambda checked=False, c=category: self.show_category(c)
            )
            self.left_layout.addWidget(btn)
            self.category_buttons[category] = btn

        self.left_layout.addWidget(self.separator())

        self.tool_area = QWidget()
        self.tool_layout = QVBoxLayout(self.tool_area)
        self.tool_layout.setContentsMargins(0, 0, 0, 0)
        self.tool_layout.setSpacing(5)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.tool_area)

        self.left_layout.addWidget(scroll)

        content.addWidget(left)

        # CENTER IMAGE PREVIEW
        center = QVBoxLayout()

        self.view = ImageView()
        self.view.selectionFinished.connect(self.on_selection)
        self.view.brushFinished.connect(self.on_healing)
        self.view.imageClicked.connect(self.on_image_click)

        center.addWidget(self.view)

        self.preview_label = QLabel("Click the photo to toggle Before / After")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(self.preview_label)

        content.addLayout(center, 1)

        # RIGHT PANEL
        right = QFrame()
        right.setFixedWidth(285)

        right_layout = QVBoxLayout(right)

        self.tool_title = QLabel("Adjust")
        self.tool_title.setStyleSheet("font-size:19px;font-weight:bold;")
        right_layout.addWidget(self.tool_title)

        self.settings_area = QWidget()
        self.settings_layout = QVBoxLayout(self.settings_area)
        self.settings_layout.setContentsMargins(0, 0, 0, 0)

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setWidget(self.settings_area)

        right_layout.addWidget(settings_scroll)

        content.addWidget(right)
        main_layout.addLayout(content, 1)

        self.view.mousePressEvent = self.view_mouse_press
        self.show_category("ALL")
        self.show_tool("Adjust")

    def top_button(self, text, function):
        btn = QPushButton(text)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(function)
        btn.setMinimumHeight(36)
        return btn

    def separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    # -----------------------------------------------------
    # CATEGORY / TOOL MENU
    # -----------------------------------------------------

    def show_category(self, category):
        self.clear_layout(self.tool_layout)

        tools = {
            "ALL": [
                "Adjust", "Details", "Curves", "White Balance",
                "Colour Grade", "Selective", "Healing", "Crop",
                "Vintage", "Black & White", "Text", "Filter"
            ],
            "REFINE": [
                "Adjust", "Details", "Curves",
                "White Balance", "Colour Grade", "Selective"
            ],
            "FIX": ["Healing", "Crop"],
            "STYLE": [
                "Vintage", "Black & White", "Text", "Filter"
            ]
        }

        icons = {
            "Adjust": "☼",
            "Details": "✧",
            "Curves": "〰",
            "White Balance": "◉",
            "Colour Grade": "◌",
            "Selective": "◎",
            "Healing": "✚",
            "Crop": "▣",
            "Vintage": "◷",
            "Black & White": "◐",
            "Text": "T",
            "Filter": "◈"
        }

        for tool in tools[category]:
            btn = QPushButton(f"{icons[tool]}   {tool}")
            btn.setMinimumHeight(38)
            btn.clicked.connect(
                lambda checked=False, t=tool: self.show_tool(t)
            )
            self.tool_layout.addWidget(btn)

        self.tool_layout.addStretch()

    def show_tool(self, tool):
        self.active_tool = tool
        self.tool_title.setText(tool)
        self.clear_layout(self.settings_layout)

        self.view.set_mode("normal")

        if tool == "Adjust":
            self.adjust_panel()

        elif tool == "Details":
            self.details_panel()

        elif tool == "Curves":
            self.curves_panel()

        elif tool == "White Balance":
            self.white_balance_panel()

        elif tool == "Colour Grade":
            self.colour_grade_panel()

        elif tool == "Selective":
            self.selective_panel()
            self.view.set_mode("selective")

        elif tool == "Healing":
            self.healing_panel()
            self.view.set_mode("healing")

        elif tool == "Crop":
            self.crop_panel()

        elif tool == "Vintage":
            self.vintage_panel()

        elif tool == "Black & White":
            self.bw_panel()

        elif tool == "Text":
            self.text_panel()

        elif tool == "Filter":
            self.filter_panel()

    # -----------------------------------------------------
    # COMMON CONTROLS
    # -----------------------------------------------------

    def add_label(self, text):
        label = QLabel(text)
        label.setStyleSheet("font-weight:bold;margin-top:8px;")
        self.settings_layout.addWidget(label)
        return label

    def add_button(self, text, function):
        btn = QPushButton(text)
        btn.setMinimumHeight(34)
        btn.clicked.connect(function)
        self.settings_layout.addWidget(btn)
        return btn

    def add_slider(self, name, minimum, maximum, default, callback):
        self.add_label(name)

        row = QHBoxLayout()
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(default)

        value_label = QLabel(str(default))
        value_label.setFixedWidth(35)

        slider.valueChanged.connect(
            lambda value: value_label.setText(str(value))
        )
        slider.valueChanged.connect(callback)

        row.addWidget(slider)
        row.addWidget(value_label)

        self.settings_layout.addLayout(row)
        return slider

    def add_combo(self, name, options, callback):
        self.add_label(name)
        combo = QComboBox()
        combo.addItems(options)
        combo.currentTextChanged.connect(callback)
        self.settings_layout.addWidget(combo)
        return combo

    def require_image(self):
        if self.current is None:
            QMessageBox.information(
                self, "No Image", "Please add a photo first."
            )
            return False
        return True

    # -----------------------------------------------------
    # IMAGE HISTORY
    # -----------------------------------------------------

    def commit(self, image):
        if image is None:
            return

        if self.current is not None:
            self.undo_stack.append(self.current.copy())

        self.redo_stack.clear()
        self.current = image.convert("RGB")
        self.view.set_image(self.current)
        self.before = None

    def undo(self):
        if not self.undo_stack:
            return

        self.redo_stack.append(self.current.copy())
        self.current = self.undo_stack.pop()
        self.view.set_image(self.current)

    def redo(self):
        if not self.redo_stack:
            return

        self.undo_stack.append(self.current.copy())
        self.current = self.redo_stack.pop()
        self.view.set_image(self.current)

    def reset_image(self):
        if self.original is not None:
            self.commit(self.original.copy())
            self.view.selections.clear()
            self.view.update()

    # -----------------------------------------------------
    # OPEN / SAVE
    # -----------------------------------------------------

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )

        if not path:
            return

        try:
            image = Image.open(path).convert("RGB")
            self.original = image.copy()
            self.current = image.copy()

            self.undo_stack.clear()
            self.redo_stack.clear()

            self.view.selections.clear()
            self.view.set_image(self.current)

        except Exception as error:
            QMessageBox.critical(self, "Error", str(error))

    def save_image(self):
        if not self.require_image():
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "edited_photo.jpg",
            "JPEG (*.jpg);;PNG (*.png)"
        )

        if path:
            try:
                self.current.save(path, quality=95)
                QMessageBox.information(
                    self, "Saved", "Image saved successfully!"
                )
            except Exception as error:
                QMessageBox.critical(self, "Save Error", str(error))

    # -----------------------------------------------------
    # BEFORE / AFTER
    # -----------------------------------------------------

    def view_mouse_press(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self.current is not None and self.original is not None:
                if self.before is None:
                    self.before = self.current.copy()
                    self.view.set_image(self.original)
                    self.preview_label.setText("BEFORE — click again")
                else:
                    self.view.set_image(self.current)
                    self.preview_label.setText(
                        "AFTER — click to see Before"
                    )
                    self.before = None

        ImageView.mousePressEvent(self.view, event)

    # -----------------------------------------------------
    # ADJUST
    # -----------------------------------------------------

    def adjust_panel(self):
        values = {
            "Brightness": 0,
            "Contrast": 0,
            "Saturation": 0,
            "Highlights": 0,
            "Shadows": 0,
            "Ambiance": 0,
            "Warmth": 0,
            "Sharpness": 0
        }

        for name in values:
            self.add_slider(
                name, -100, 100, 0,
                lambda value, n=name: self.adjust_value(n, value)
            )

        self.add_button("Apply Adjustments", self.apply_adjustments)

        self.adjust_values = values
        self.adjust_base = None

    def adjust_value(self, name, value):
        if self.current is None:
            return

        if self.adjust_base is None:
            self.adjust_base = self.current.copy()

        self.adjust_values[name] = value
        self.current = self.process_adjustments(
            self.adjust_base, self.adjust_values
        )
        self.view.set_image(self.current)

    def process_adjustments(self, image, values):
        img = image.copy()

        brightness = values["Brightness"]
        contrast = values["Contrast"]
        saturation = values["Saturation"]
        sharpness = values["Sharpness"]

        if brightness:
            img = ImageEnhance.Brightness(img).enhance(
                max(0, 1 + brightness / 100)
            )

        if contrast:
            img = ImageEnhance.Contrast(img).enhance(
                max(0, 1 + contrast / 100)
            )

        if saturation:
            img = ImageEnhance.Color(img).enhance(
                max(0, 1 + saturation / 100)
            )

        if sharpness:
            img = ImageEnhance.Sharpness(img).enhance(
                max(0, 1 + sharpness / 100)
            )

        arr = np.array(img).astype(np.float32)

        # Shadows
        shadow = values["Shadows"] / 100
        if shadow:
            dark_mask = 1 - (arr.mean(axis=2) / 255)
            arr += (shadow * 75 * dark_mask)[..., None]

        # Highlights
        highlight = values["Highlights"] / 100
        if highlight:
            light_mask = arr.mean(axis=2) / 255
            arr += (highlight * 65 * light_mask)[..., None]

        # Ambiance
        ambiance = values["Ambiance"] / 100
        if ambiance:
            arr += ambiance * (arr - 128) * 0.25

        # Warmth
        warmth = values["Warmth"] / 100
        if warmth:
            arr[:, :, 0] += warmth * 30
            arr[:, :, 2] -= warmth * 30

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def apply_adjustments(self):
        if self.current is None:
            return

        self.commit(self.current.copy())
        self.adjust_base = None

    # -----------------------------------------------------
    # DETAILS
    # -----------------------------------------------------

    def details_panel(self):
        self.add_slider(
            "Texture", -100, 100, 0,
            self.change_texture
        )

    def change_texture(self, value):
        if self.current is None:
            return

        if not hasattr(self, "texture_base") or self.texture_base is None:
            self.texture_base = self.current.copy()

        img = self.texture_base.copy()

        if value < 0:
            img = img.filter(
                ImageFilter.GaussianBlur(abs(value) / 35)
            )
        elif value > 0:
            img = ImageEnhance.Sharpness(img).enhance(
                1 + value / 35
            )

        self.current = img
        self.view.set_image(img)

    # -----------------------------------------------------
    # CURVES
    # -----------------------------------------------------

    def curves_panel(self):
        self.add_combo(
            "Channel",
            ["RGB", "Red", "Green", "Blue", "Luminance"],
            self.set_curve_channel
        )

        self.curve_channel = "RGB"

        self.add_combo(
            "Presets",
            [
                "Neutral", "Soft Contrast", "Hard Contrast",
                "Brighten", "Darken", "Faded"
            ],
            self.apply_curve_preset
        )

    def set_curve_channel(self, channel):
        self.curve_channel = channel

    def apply_curve_preset(self, preset):
        if not self.require_image():
            return

        arr = np.array(self.current).astype(np.float32)
        x = np.arange(256, dtype=np.float32)

        if preset == "Neutral":
            curve = x
        elif preset == "Soft Contrast":
            curve = np.clip((x - 128) * 1.12 + 128, 0, 255)
        elif preset == "Hard Contrast":
            curve = np.clip((x - 128) * 1.35 + 128, 0, 255)
        elif preset == "Brighten":
            curve = np.clip(x * 1.18 + 10, 0, 255)
        elif preset == "Darken":
            curve = np.clip(x * 0.82, 0, 255)
        else:
            curve = np.clip(x * 0.82 + 35, 0, 255)

        if self.curve_channel == "RGB":
            for c in range(3):
                arr[:, :, c] = np.interp(arr[:, :, c], x, curve)

        elif self.curve_channel == "Luminance":
            lum = arr.mean(axis=2)
            new_lum = np.interp(lum, x, curve)
            factor = new_lum / np.maximum(lum, 1)
            arr *= factor[:, :, None]

        else:
            channel_index = {
                "Red": 0, "Green": 1, "Blue": 2
            }[self.curve_channel]

            arr[:, :, channel_index] = np.interp(
                arr[:, :, channel_index], x, curve
            )

        self.commit(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)))

    # -----------------------------------------------------
    # WHITE BALANCE
    # -----------------------------------------------------

    def white_balance_panel(self):
        self.add_slider(
            "Temperature (Cold ↔ Warm)",
            -100, 100, 0,
            self.change_temperature
        )

    def change_temperature(self, value):
        if self.current is None:
            return

        if not hasattr(self, "temperature_base") or self.temperature_base is None:
            self.temperature_base = self.current.copy()

        arr = np.array(self.temperature_base).astype(np.float32)
        amount = value / 100

        arr[:, :, 0] += amount * 35
        arr[:, :, 2] -= amount * 35

        self.current = Image.fromarray(
            np.clip(arr, 0, 255).astype(np.uint8)
        )
        self.view.set_image(self.current)

    # -----------------------------------------------------
    # COLOUR GRADING
    # -----------------------------------------------------

    def colour_grade_panel(self):
        self.grade_colors = {
            "Global": None,
            "Shadow": None,
            "Midtones": None,
            "Highlights": None
        }

        for part in self.grade_colors:
            self.add_button(
                f"🎨 {part} — Choose Color",
                lambda checked=False, p=part: self.choose_grade_color(p)
            )

        self.add_slider(
            "Luminance",
            -100, 100, 0,
            self.grade_luminance
        )

        self.add_button("Apply Colour Grade", self.apply_grade)

    def choose_grade_color(self, part):
        color = QColorDialog.getColor()
        if color.isValid():
            self.grade_colors[part] = color.name()

    def grade_luminance(self, value):
        self.grade_light = value

    def apply_grade(self):
        if not self.require_image():
            return

        arr = np.array(self.current).astype(np.float32)
        lum = arr.mean(axis=2)

        for part, color in self.grade_colors.items():
            if not color:
                continue

            rgb = np.array(QColor(color).getRgb()[:3])
            tint = (rgb - 128) / 128

            if part == "Global":
                mask = np.ones_like(lum)
            elif part == "Shadow":
                mask = np.clip((128 - lum) / 128, 0, 1)
            elif part == "Highlights":
                mask = np.clip((lum - 128) / 127, 0, 1)
            else:
                mask = 1 - np.abs(lum - 128) / 128
                mask = np.clip(mask, 0, 1)

            arr += tint[None, None, :] * mask[:, :, None] * 25

        light = getattr(self, "grade_light", 0)
        arr += light * 0.5

        self.commit(
            Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        )

    # -----------------------------------------------------
    # SELECTIVE
    # -----------------------------------------------------

    def selective_panel(self):
        self.add_label("Draw rectangles on the photo to select areas.")
        self.add_label("You can add multiple selections.")

        self.add_button("Add Selection Mode", self.start_selection)
        self.add_button("Clear Selections", self.clear_selections)

        self.add_slider(
            "Brightness", -100, 100, 0,
            lambda v: self.set_selective_value("brightness", v)
        )
        self.add_slider(
            "Contrast", -100, 100, 0,
            lambda v: self.set_selective_value("contrast", v)
        )
        self.add_slider(
            "Saturation", -100, 100, 0,
            lambda v: self.set_selective_value("saturation", v)
        )

        self.add_button("Apply to Selected Areas", self.apply_selective)

    def start_selection(self):
        self.view.set_mode("selective")

    def on_selection(self, rect):
        self.preview_label.setText(
            f"Selected areas: {len(self.view.selections)}"
        )

    def clear_selections(self):
        self.view.selections.clear()
        self.view.update()

    def set_selective_value(self, name, value):
        self.selective_values[name] = value

    def apply_selective(self):
        if not self.require_image():
            return

        if not self.view.selections:
            QMessageBox.information(
                self, "Selection", "First select an area on the photo."
            )
            return

        arr = np.array(self.current).astype(np.float32)

        for x1, y1, x2, y2 in self.view.selections:
            area = arr[y1:y2, x1:x2]

            area += self.selective_values.get("brightness", 0) * 1.2

            contrast = self.selective_values.get("contrast", 0)
            area[:] = (area - 128) * (1 + contrast / 100) + 128

            saturation = self.selective_values.get("saturation", 0)
            gray = area.mean(axis=2, keepdims=True)
            area[:] = gray + (area - gray) * (1 + saturation / 100)

        self.commit(
            Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        )

        self.selective_values.clear()

    # -----------------------------------------------------
    # HEALING
    # -----------------------------------------------------

    def healing_panel(self):
        if cv2 is None:
            self.add_label(
                "Install OpenCV to use Healing:\n"
                "python -m pip install opencv-python"
            )
            return

        self.add_label("Paint over unwanted objects in the photo.")
        self.add_label("Release the mouse to remove the marked area.")
        self.add_slider(
            "Brush Size", 5, 100, 25,
            self.set_brush_size
        )
        self.brush_size = 25

    def set_brush_size(self, value):
        self.brush_size = value

    def on_healing(self, points):
        if not self.require_image() or cv2 is None:
            return

        if not points:
            return

        mask = np.zeros(
            (self.current.height, self.current.width),
            dtype=np.uint8
        )

        for point in points:
            if point:
                cv2.circle(
                    mask, point, self.brush_size // 2,
                    255, -1
                )

        for i in range(1, len(points)):
            if points[i - 1] and points[i]:
                cv2.line(
                    mask, points[i - 1], points[i],
                    255, self.brush_size
                )

        arr = np.array(self.current)
        result = cv2.inpaint(
            cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
            mask, 3, cv2.INPAINT_TELEA
        )

        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        self.commit(Image.fromarray(result))

    # -----------------------------------------------------
    # CROP / ROTATE / FLIP
    # -----------------------------------------------------

    def crop_panel(self):
        self.add_combo(
            "Crop Ratio",
            [
                "Original", "DIN 3.8 cm", "Square 5 cm",
                "4:5", "5:4", "3:4", "2:3", "16:9"
            ],
            self.set_crop_ratio
        )

        self.crop_ratio = "Original"

        self.add_slider(
            "Straighten", -45, 45, 0,
            self.straighten
        )

        self.add_button("Apply Crop", self.apply_crop)
        self.add_button("Rotate Left", lambda: self.rotate(-90))
        self.add_button("Rotate Right", lambda: self.rotate(90))
        self.add_button("Flip Horizontal", self.flip_horizontal)
        self.add_button("Flip Vertical", self.flip_vertical)

    def set_crop_ratio(self, value):
        self.crop_ratio = value

    def straighten(self, value):
        if self.current is None:
            return

        if not hasattr(self, "straighten_base") or self.straighten_base is None:
            self.straighten_base = self.current.copy()

        self.current = self.straighten_base.rotate(
            value, resample=Image.Resampling.BICUBIC,
            expand=True
        )
        self.view.set_image(self.current)

    def apply_crop(self):
        if not self.require_image():
            return

        w, h = self.current.size

        ratios = {
            "4:5": 4 / 5,
            "5:4": 5 / 4,
            "3:4": 3 / 4,
            "2:3": 2 / 3,
            "16:9": 16 / 9,
            "Square 5 cm": 1,
            "DIN 3.8 cm": 3.8 / 5
        }

        if self.crop_ratio == "Original":
            return

        ratio = ratios[self.crop_ratio]
        current_ratio = w / h

        if current_ratio > ratio:
            new_w = int(h * ratio)
            left = (w - new_w) // 2
            box = (left, 0, left + new_w, h)
        else:
            new_h = int(w / ratio)
            top = (h - new_h) // 2
            box = (0, top, w, top + new_h)

        self.commit(self.current.crop(box))
        self.straighten_base = None

    def rotate(self, angle):
        if self.require_image():
            self.commit(self.current.rotate(angle, expand=True))

    def flip_horizontal(self):
        if self.require_image():
            self.commit(ImageOps.mirror(self.current))

    def flip_vertical(self):
        if self.require_image():
            self.commit(ImageOps.flip(self.current))

    # -----------------------------------------------------
    # VINTAGE
    # -----------------------------------------------------

    def vintage_panel(self):
        self.add_button("Apply Vintage Effect", self.apply_vintage)

    def apply_vintage(self):
        if not self.require_image():
            return

        img = ImageEnhance.Color(self.current).enhance(0.65)
        img = ImageEnhance.Contrast(img).enhance(0.9)

        arr = np.array(img).astype(np.float32)
        arr[:, :, 0] += 15
        arr[:, :, 1] += 5
        arr[:, :, 2] -= 12

        self.commit(
            Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        )

    # -----------------------------------------------------
    # BLACK AND WHITE
    # -----------------------------------------------------

    def bw_panel(self):
        self.add_combo(
            "Preset",
            ["Neutral", "Contrast", "Bright", "Dark", "Film"],
            self.apply_bw_preset
        )

        self.add_slider(
            "Brightness", -100, 100, 0,
            self.bw_brightness
        )
        self.add_slider(
            "Contrast", -100, 100, 0,
            self.bw_contrast
        )
        self.add_slider(
            "Grain", 0, 100, 0,
            self.bw_grain
        )

        self.bw_values = {
            "brightness": 0,
            "contrast": 0,
            "grain": 0
        }

    def apply_bw_preset(self, preset):
        if not self.require_image():
            return

        img = ImageOps.grayscale(self.current).convert("RGB")

        if preset == "Contrast":
            img = ImageEnhance.Contrast(img).enhance(1.6)
        elif preset == "Bright":
            img = ImageEnhance.Brightness(img).enhance(1.25)
        elif preset == "Dark":
            img = ImageEnhance.Brightness(img).enhance(0.7)
        elif preset == "Film":
            img = ImageEnhance.Contrast(img).enhance(0.8)
            img = ImageEnhance.Brightness(img).enhance(1.08)

        self.commit(img)

    def bw_brightness(self, value):
        self.bw_values["brightness"] = value

    def bw_contrast(self, value):
        self.bw_values["contrast"] = value

    def bw_grain(self, value):
        self.bw_values["grain"] = value

    # -----------------------------------------------------
    # TEXT
    # -----------------------------------------------------

    def text_panel(self):
        self.add_button("Add Text", self.add_text)
        self.add_button("Choose Text Color", self.choose_text_color)

        self.add_label("After entering text, click on the photo to place it.")

    def add_text(self):
        if not self.require_image():
            return

        text, ok = QInputDialog.getText(
            self, "Add Text", "Enter your text:"
        )

        if ok and text:
            self.pending_text = text
            self.view.set_mode("text")
            self.preview_label.setText("Click the photo to place your text.")

    def choose_text_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.text_color = color.name()

    def on_image_click(self, position):
        if self.active_tool == "Text" and hasattr(self, "pending_text"):
            img = self.current.copy()
            draw = ImageDraw.Draw(img)

            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except Exception:
                font = ImageFont.load_default()

            draw.text(
                position,
                self.pending_text,
                fill=self.text_color,
                font=font
            )

            self.commit(img)
            del self.pending_text
            self.view.set_mode("normal")

    # -----------------------------------------------------
    # FILTER
    # -----------------------------------------------------

    def filter_panel(self):
        self.add_combo(
            "Filter",
            ["Original", "Vivid", "Dramatic", "Black & White"],
            self.apply_filter
        )

    def apply_filter(self, preset):
        if not self.require_image():
            return

        if preset == "Original":
            self.commit(self.original.copy())
        elif preset == "Vivid":
            img = ImageEnhance.Color(self.current).enhance(1.2)
            img = ImageEnhance.Brightness(img).enhance(1.05)
            self.commit(img)
        elif preset == "Dramatic":
            img = ImageEnhance.Contrast(self.current).enhance(1.3)
            img = ImageEnhance.Color(img).enhance(0.85)
            self.commit(img)
        elif preset == "Black & White":
            self.commit(
                ImageOps.grayscale(self.current).convert("RGB")
            )

    # -----------------------------------------------------
    # THEME
    # -----------------------------------------------------

    def toggle_theme(self):
        self.theme = "white" if self.theme == "black" else "black"
        self.apply_theme()

    def apply_theme(self):
        if self.theme == "black":
            bg = "#171717"
            panel = "#242424"
            fg = "#ffffff"
            button = "#333333"
            border = "#444444"
        else:
            bg = "#f3f3f3"
            panel = "#ffffff"
            fg = "#111111"
            button = "#e3e3e3"
            border = "#cccccc"

        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {bg};
                color: {fg};
                font-size: 13px;
            }}

            QFrame {{
                background-color: {panel};
                border: 1px solid {border};
                border-radius: 7px;
            }}

            QPushButton {{
                background-color: {button};
                color: {fg};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 7px;
            }}

            QPushButton:hover {{
                border: 1px solid #36a9ff;
            }}

            QLabel {{
                color: {fg};
                border: none;
            }}

            QComboBox {{
                background-color: {button};
                color: {fg};
                border: 1px solid {border};
                padding: 6px;
                border-radius: 5px;
            }}

            QSlider::groove:horizontal {{
                height: 5px;
                background: {border};
                border-radius: 2px;
            }}

            QSlider::handle:horizontal {{
                background: #299cff;
                width: 13px;
                margin: -5px 0;
                border-radius: 6px;
            }}

            QScrollArea {{
                border: none;
            }}
        """)


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = AIPixelStudio()
    window.show()

    sys.exit(app.exec())