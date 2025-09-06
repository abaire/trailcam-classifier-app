from __future__ import annotations

# ruff: noqa: N802 Function name should be lowercase
import asyncio
import os
import sys
import threading
import traceback
import typing
from pathlib import Path

import qtinter
from PySide6.QtCore import Q_ARG, QDataStream, QMetaObject, QSettings, Qt, Signal, Slot
from PySide6.QtGui import QIntValidator
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from trailcam_classifier.main import ClassificationConfig, run_classification
from trailcam_classifier.util import MODEL_SAVE_FILENAME

_DEFAULT_OUTPUT_DIRECTORY = "images_with_objects_detected"


def run_coroutine_in_thread(
    window: MainWindow,
    coroutine: typing.Coroutine,
    *args,
    **kwargs,
) -> threading.Thread:
    """Runs a coroutine in a new thread."""

    def thread_target():
        def thread_safe_logger(message: str):
            QMetaObject.invokeMethod(window, "log", Qt.QueuedConnection, Q_ARG(str, message))

        def thread_safe_progress_update(item_name: str, total_count: int):
            QMetaObject.invokeMethod(
                window, "log_progress", Qt.QueuedConnection, Q_ARG(str, item_name), Q_ARG(int, total_count)
            )

        kwargs["logger"] = thread_safe_logger
        kwargs["progress_update"] = thread_safe_progress_update
        try:
            asyncio.run(coroutine(*args, **kwargs))
        except Exception:  # noqa: BLE001 Do not catch blind exception: `Exception`
            tb = traceback.format_exc()
            QMetaObject.invokeMethod(window, "log", Qt.QueuedConnection, Q_ARG(str, f"An error occurred:\n{tb}"))

    thread = threading.Thread(target=thread_target)
    thread.start()
    return thread


def get_resource_path(relative_path: str) -> Path:
    """Get the absolute path to a resource, works for dev and for PyInstaller"""

    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = Path(sys._MEIPASS)  # noqa: SLF001 Private member accessed: `_MEIPASS`
    except AttributeError:
        base_path = Path(".").absolute()

    return base_path / relative_path


def is_bundle() -> bool:
    """Returns True if the application is running from a PyInstaller bundle."""
    return hasattr(sys, "_MEIPASS")


class Application(QApplication):
    """Application class to handle app-level events."""

    new_file_open = Signal(str)


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.settings = QSettings()

        layout = QVBoxLayout(self)

        output_dir_layout = QHBoxLayout()
        output_dir_label = QLabel("Output Directory:")
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(self.settings.value("output_directory", "classified_output"))
        browse_output_button = QPushButton("Browse...")
        browse_output_button.clicked.connect(self.browse_output_directory)
        output_dir_layout.addWidget(output_dir_label)
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(browse_output_button)
        layout.addLayout(output_dir_layout)

        model_path_layout = QHBoxLayout()
        model_path_label = QLabel("Model Path:")
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setText(self.settings.value("model_path", MODEL_SAVE_FILENAME))
        browse_model_button = QPushButton("Browse...")
        browse_model_button.clicked.connect(self.browse_model_path)
        model_path_layout.addWidget(model_path_label)
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(browse_model_button)
        layout.addLayout(model_path_layout)

        confidence_layout = QHBoxLayout()
        confidence_label = QLabel("Confidence Threshold:")
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_edit = QLineEdit()
        self.confidence_edit.setValidator(QIntValidator(0, 100))
        self.confidence_slider.valueChanged.connect(self.update_confidence_edit)
        self.confidence_edit.textChanged.connect(self.update_confidence_slider)
        confidence_layout.addWidget(confidence_label)
        confidence_layout.addWidget(self.confidence_slider)
        confidence_layout.addWidget(self.confidence_edit)
        layout.addLayout(confidence_layout)

        self.confidence_slider.setValue(self.settings.value("confidence_threshold", 65, type=int))
        self.update_confidence_edit(self.confidence_slider.value())

        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def browse_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)

    def browse_model_path(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Model File", filter="PyTorch Models (*.pth)")
        if filepath:
            self.model_path_edit.setText(filepath)

    def update_confidence_edit(self, value):
        self.confidence_edit.setText(str(value))

    def update_confidence_slider(self, text):
        if text:
            self.confidence_slider.setValue(int(text))

    def accept(self):
        self.settings.setValue("output_directory", self.output_dir_edit.text())
        self.settings.setValue("model_path", self.model_path_edit.text())
        self.settings.setValue("confidence_threshold", self.confidence_slider.value())
        super().accept()


class DropLabel(QLabel):
    def __init__(self, text, parent: MainWindow):
        super().__init__(text, parent)
        self.parent_widget = parent
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            """
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
                padding: 20px;
                font-size: 16px;
            }
        """
        )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return

        # For now, we only support dropping a single folder.
        url = urls[0]
        if url.isLocalFile():
            folder_path = url.toLocalFile()
            if Path(folder_path).is_dir():
                self.parent_widget.start_classification(folder_path)
            else:
                self.parent_widget.log("Please drop a folder, not a file.")


class MainWindow(QMainWindow):
    log_updated = Signal(str)
    progress_updated = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trailcam Classifier")
        self.setGeometry(100, 100, 600, 400)
        self.settings = QSettings()
        self._classification_task = None
        self._progress_counter = 0

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        self.drop_label = DropLabel("Drop a folder here", self)
        layout.addWidget(self.drop_label)
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        layout.addWidget(self.log_widget)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self._create_menus()
        self.log_updated.connect(self.log)
        self.progress_updated.connect(self._on_progress_updated)
        QApplication.instance().new_file_open.connect(self.start_classification)

    def _create_menus(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        settings_action = file_menu.addAction("&Settings")
        settings_action.triggered.connect(self.open_settings)
        exit_action = file_menu.addAction("E&xit")
        exit_action.triggered.connect(self.close)

    def open_settings(self):
        dialog = SettingsDialog(self)
        dialog.exec()

    @Slot(str)
    def log(self, message: str):
        self.log_widget.append(message)

    @Slot(str, int)
    def log_progress(self, _item_name: str, total_count: int):
        self._progress_counter += 1
        self.progress_updated.emit(self._progress_counter, total_count)

    def _on_progress_updated(self, current_value: int, total_count: int):
        if self.progress_bar.maximum() != total_count:
            self.progress_bar.setMaximum(total_count)
        self.progress_bar.setValue(current_value)

    def get_output_directory(self) -> str:
        """Determines the output directory based on settings and execution context."""
        output_dir_setting = self.settings.value("output_directory")

        if output_dir_setting:
            output_path = Path(output_dir_setting)
            if not output_path.is_absolute() and is_bundle():
                # For relative paths when bundled, resolve them relative to the
                # directory containing the .app bundle.
                # sys.executable is .../Blah.app/Contents/MacOS/Blah in a bundle
                # .parents[3] is the directory containing the .app
                app_bundle_container_dir = Path(sys.executable).parents[3]
                return str(app_bundle_container_dir / output_path)
            return str(output_path)

        if is_bundle():
            return str(Path.home() / "Desktop" / _DEFAULT_OUTPUT_DIRECTORY)
        return _DEFAULT_OUTPUT_DIRECTORY

    def start_classification(self, folder_path: str):
        if self._classification_task and self._classification_task.is_alive():
            self.log("A classification process is already running.")
            return

        self.log_widget.clear()
        abs_folder_path = os.path.abspath(folder_path)
        self.log(f"Starting classification for folder: {abs_folder_path}")
        self.progress_bar.setValue(0)
        self._progress_counter = 0

        output_directory = os.path.abspath(self.get_output_directory())
        default_model_path = str(get_resource_path("model/trailcam_classifier_model.pt"))
        model_path: str = str(self.settings.value("model_path", default_model_path))
        confidence_threshold_int = self.settings.value("confidence_threshold", 65, type=int)
        confidence_threshold = confidence_threshold_int / 100.0

        config = ClassificationConfig(
            dirs=[abs_folder_path],
            output=output_directory,
            model=model_path,
            confidence_threshold=confidence_threshold,
        )

        self._classification_task = run_coroutine_in_thread(
            self,
            run_classification,
            config=config,
        )


def run_gui():
    QApplication.setOrganizationName("BearBrains")
    QApplication.setApplicationName("Trailcam Classifier")
    server_name = "TrailcamClassifierApp"

    socket = QLocalSocket()
    socket.connectToServer(server_name)

    if socket.waitForConnected(500):
        args = sys.argv[1:]
        if args:
            # We only process the first argument, which should be the folder path.
            # The OS can sometimes pass the folder's contents as subsequent arguments.
            out = QDataStream(socket)
            out.writeQString(args[0])
            socket.flush()
            socket.waitForBytesWritten(1000)
        socket.disconnectFromServer()
        return

    server = QLocalServer()

    def cleanup():
        server.close()
        # QLocalServer may not remove the socket file on all platforms
        # especially on unclean shutdowns. We do it manually.
        QLocalServer.removeServer(server_name)

    app = Application(sys.argv)
    app.aboutToQuit.connect(cleanup)

    server.listen(server_name)

    def handle_new_connection():
        socket = server.nextPendingConnection()
        if socket:
            socket.waitForReadyRead(1000)
            stream = QDataStream(socket)
            message = stream.readQString()
            if message:
                app.new_file_open.emit(message)
            socket.disconnectFromServer()

    server.newConnection.connect(handle_new_connection)

    with qtinter.using_asyncio_from_qt():
        window = MainWindow()
        window.show()

        if len(sys.argv) > 1:
            for arg in sys.argv[1:]:
                # On some systems, the OS might pass other flags. We only want
                # to open things that look like paths.
                if Path(arg).exists():
                    window.start_classification(arg)

        sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
