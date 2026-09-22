"""
pcb_scanner_gui.py

PyQt5 + PyQtGraph GUI for the Portable PCB Health Scanner.

Software-only diagnostic pipeline:

    Synthetic / Hardware Signal
              |
              v
        FFT Signature
              |
              v
       Healthy Baseline
              |
              v
     Signature Comparison
              |
              v
      Fault Classification
              |
              v
       Novelty / Abnormality
              |
              v
         GUI Diagnosis
              |
              v
        SQLite Logging
              |
              v
       CSV / Excel Export

The GUI does NOT perform fault classification itself.

All classification decisions are obtained from fault_classifier.py.
This keeps the GUI and diagnostic engine separated.

Supported profiles:

    HEALTHY BOARD
    OPEN TRACE
    SHORT CIRCUIT
    REGULATOR FAULT
    CAPACITOR DRIFT
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

import pyqtgraph as pg

from baseline_capture import (
    build_average_baseline,
)

from fault_classifier import (
    classify_fault,
)

from signature_comparison import (
    compare_signatures,
)

from signal_simulator import (
    AcquisitionConfig,
    BoardProfile,
    generate_known_good,
    generate_open_trace_fault,
    generate_short_circuit_fault,
    generate_regulator_fault,
    generate_capacitor_drift_fault,
)

from spectral_analysis import (
    compute_signature,
)

from signature_database import (
    SignatureDatabase,
)


# =====================================================================
# APPLICATION CONFIGURATION
# =====================================================================

APP_TITLE = "Portable PCB Health Scanner"

DATABASE_FILE = "pcb_scanner.db"

BASELINE_FILE = (
    "baseline_library/gui_baseline.npz"
)

BASELINE_COUNT = 10

# Deterministic seeds for repeatable testing.
TEST_SEEDS = {
    "HEALTHY BOARD": 1000,
    "OPEN TRACE": 1001,
    "SHORT CIRCUIT": 1002,
    "REGULATOR FAULT": 1003,
    "CAPACITOR DRIFT": 1004,
}


# =====================================================================
# MAIN WINDOW
# =====================================================================

class PCBHealthScannerGUI(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            APP_TITLE
        )

        self.resize(
            1450,
            950,
        )

        # -------------------------------------------------------------
        # Scanner configuration
        # -------------------------------------------------------------

        self.config = AcquisitionConfig()

        self.board = BoardProfile()

        # -------------------------------------------------------------
        # Runtime objects
        # -------------------------------------------------------------

        self.baseline = None

        self.baseline_id = None

        self.last_live_signature = None

        self.last_comparison = None

        self.last_result = None

        # -------------------------------------------------------------
        # Database
        # -------------------------------------------------------------

        self.database = SignatureDatabase(
            DATABASE_FILE
        )

        self.board_id = (
            self.database.add_board(

                board_name=(
                    "Synthetic Demo PCB"
                ),

                board_serial=(
                    "DEMO-001"
                ),

                description=(
                    "Portable PCB Health Scanner "
                    "software demonstration board"
                ),
            )
        )

        # -------------------------------------------------------------
        # UI
        # -------------------------------------------------------------

        self.build_ui()

        # -------------------------------------------------------------
        # Initial baseline
        # -------------------------------------------------------------

        self.build_baseline(
            show_message=False
        )

        self.refresh_history()

        self.update_status(
            "SYSTEM READY"
        )


    # =================================================================
    # BUILD UI
    # =================================================================

    def build_ui(self):

        central = QWidget()

        self.setCentralWidget(
            central
        )

        main_layout = QVBoxLayout(
            central
        )

        main_layout.setSpacing(10)

        # =============================================================
        # HEADER
        # =============================================================

        header = QLabel(
            "PORTABLE PCB HEALTH SCANNER"
        )

        header.setAlignment(
            Qt.AlignCenter
        )

        header.setStyleSheet(
            """
            QLabel {
                font-size: 26px;
                font-weight: bold;
                padding: 14px;
                background-color: #20242b;
                color: white;
                border-radius: 6px;
            }
            """
        )

        main_layout.addWidget(
            header
        )

        # =============================================================
        # CONTROL PANEL
        # =============================================================

        control_group = QGroupBox(
            "SCAN CONTROL"
        )

        control_layout = QGridLayout(
            control_group
        )

        # -------------------------------------------------------------
        # Profile
        # -------------------------------------------------------------

        profile_label = QLabel(
            "Test Profile:"
        )

        profile_label.setFont(
            QFont(
                "Arial",
                10,
                QFont.Bold,
            )
        )

        self.fault_combo = QComboBox()

        self.fault_combo.addItems(
            [
                "HEALTHY BOARD",
                "OPEN TRACE",
                "SHORT CIRCUIT",
                "REGULATOR FAULT",
                "CAPACITOR DRIFT",
            ]
        )

        self.fault_combo.setMinimumHeight(
            38
        )

        control_layout.addWidget(
            profile_label,
            0,
            0,
        )

        control_layout.addWidget(
            self.fault_combo,
            0,
            1,
        )

        # -------------------------------------------------------------
        # Scan button
        # -------------------------------------------------------------

        self.scan_button = QPushButton(
            "SIMULATE SCAN"
        )

        self.scan_button.setMinimumHeight(
            38
        )

        self.scan_button.clicked.connect(
            self.simulate_scan
        )

        control_layout.addWidget(
            self.scan_button,
            0,
            2,
        )

        # -------------------------------------------------------------
        # Baseline
        # -------------------------------------------------------------

        self.baseline_button = QPushButton(
            "REBUILD BASELINE"
        )

        self.baseline_button.setMinimumHeight(
            38
        )

        self.baseline_button.clicked.connect(
            self.build_baseline
        )

        control_layout.addWidget(
            self.baseline_button,
            0,
            3,
        )

        # -------------------------------------------------------------
        # CSV
        # -------------------------------------------------------------

        self.csv_button = QPushButton(
            "EXPORT CSV"
        )

        self.csv_button.setMinimumHeight(
            38
        )

        self.csv_button.clicked.connect(
            self.export_csv
        )

        control_layout.addWidget(
            self.csv_button,
            0,
            4,
        )

        # -------------------------------------------------------------
        # Excel
        # -------------------------------------------------------------

        self.excel_button = QPushButton(
            "EXPORT EXCEL"
        )

        self.excel_button.setMinimumHeight(
            38
        )

        self.excel_button.clicked.connect(
            self.export_excel
        )

        control_layout.addWidget(
            self.excel_button,
            0,
            5,
        )

        main_layout.addWidget(
            control_group
        )

        # =============================================================
        # RESULT PANEL
        # =============================================================

        result_group = QGroupBox(
            "DIAGNOSTIC RESULT"
        )

        result_layout = QGridLayout(
            result_group
        )

        # -------------------------------------------------------------
        # PASS / FAIL
        # -------------------------------------------------------------

        self.result_label = QLabel(
            "READY"
        )

        self.result_label.setAlignment(
            Qt.AlignCenter
        )

        self.result_label.setMinimumHeight(
            65
        )

        result_layout.addWidget(
            self.result_label,
            0,
            0,
        )

        # -------------------------------------------------------------
        # Fault
        # -------------------------------------------------------------

        self.fault_result_label = QLabel(
            "Fault: ---"
        )

        self.fault_result_label.setAlignment(
            Qt.AlignCenter
        )

        self.fault_result_label.setMinimumHeight(
            65
        )

        result_layout.addWidget(
            self.fault_result_label,
            0,
            1,
        )

        # -------------------------------------------------------------
        # Confidence
        # -------------------------------------------------------------

        self.confidence_label = QLabel(
            "Confidence: ---"
        )

        self.confidence_label.setAlignment(
            Qt.AlignCenter
        )

        self.confidence_label.setMinimumHeight(
            65
        )

        result_layout.addWidget(
            self.confidence_label,
            0,
            2,
        )

        # -------------------------------------------------------------
        # Deviation
        # -------------------------------------------------------------

        self.deviation_label = QLabel(
            "Deviation: ---"
        )

        self.deviation_label.setAlignment(
            Qt.AlignCenter
        )

        self.deviation_label.setMinimumHeight(
            65
        )

        result_layout.addWidget(
            self.deviation_label,
            0,
            3,
        )

        # -------------------------------------------------------------
        # Novelty
        # -------------------------------------------------------------

        self.novelty_label = QLabel(
            "Novelty: ---"
        )

        self.novelty_label.setAlignment(
            Qt.AlignCenter
        )

        self.novelty_label.setMinimumHeight(
            65
        )

        result_layout.addWidget(
            self.novelty_label,
            0,
            4,
        )

        # -------------------------------------------------------------
        # Common card style
        # -------------------------------------------------------------

        card_style = """
        QLabel {
            font-size: 17px;
            font-weight: bold;
            border: 2px solid #777;
            border-radius: 5px;
            padding: 10px;
            background-color: #f5f5f5;
            color: #222;
        }
        """

        self.fault_result_label.setStyleSheet(
            card_style
        )

        self.confidence_label.setStyleSheet(
            card_style
        )

        self.deviation_label.setStyleSheet(
            card_style
        )

        self.novelty_label.setStyleSheet(
            card_style
        )

        self.set_ready_result_style()

        main_layout.addWidget(
            result_group
        )

        # =============================================================
        # SPECTRUM
        # =============================================================

        plot_group = QGroupBox(
            "SPECTRAL SIGNATURE"
        )

        plot_layout = QVBoxLayout(
            plot_group
        )

        self.plot = pg.PlotWidget()

        self.plot.setBackground(
            "w"
        )

        self.plot.showGrid(
            x=True,
            y=True,
            alpha=0.25,
        )

        self.plot.setLabel(
            "left",
            "Magnitude",
        )

        self.plot.setLabel(
            "bottom",
            "Frequency",
            units="Hz",
        )

        self.plot.setTitle(
            "Healthy Baseline vs Live Signature"
        )

        self.plot.addLegend()

        plot_layout.addWidget(
            self.plot
        )

        main_layout.addWidget(
            plot_group,
            stretch=3,
        )

        # =============================================================
        # EVIDENCE + BAND INFORMATION
        # =============================================================

        information_layout = QHBoxLayout()

        # -------------------------------------------------------------
        # Evidence
        # -------------------------------------------------------------

        evidence_group = QGroupBox(
            "DIAGNOSTIC EVIDENCE"
        )

        evidence_layout = QVBoxLayout(
            evidence_group
        )

        self.evidence_label = QLabel(
            "Diagnostic evidence will appear here."
        )

        self.evidence_label.setWordWrap(
            True
        )

        self.evidence_label.setAlignment(
            Qt.AlignTop | Qt.AlignLeft
        )

        self.evidence_label.setStyleSheet(
            """
            QLabel {
                border: 1px solid #888;
                padding: 10px;
                font-size: 14px;
                background-color: white;
            }
            """
        )

        evidence_layout.addWidget(
            self.evidence_label
        )

        information_layout.addWidget(
            evidence_group,
            stretch=3,
        )

        # -------------------------------------------------------------
        # Band information
        # -------------------------------------------------------------

        band_group = QGroupBox(
            "SIGNATURE INFORMATION"
        )

        band_layout = QVBoxLayout(
            band_group
        )

        self.band_label = QLabel(
            "Dominant deviation band: ---"
        )

        self.band_label.setWordWrap(
            True
        )

        self.band_label.setAlignment(
            Qt.AlignTop | Qt.AlignLeft
        )

        self.band_label.setStyleSheet(
            """
            QLabel {
                border: 1px solid #888;
                padding: 10px;
                font-size: 14px;
                background-color: white;
            }
            """
        )

        band_layout.addWidget(
            self.band_label
        )

        information_layout.addWidget(
            band_group,
            stretch=1,
        )

        main_layout.addLayout(
            information_layout
        )

        # =============================================================
        # HISTORY
        # =============================================================

        history_group = QGroupBox(
            "SCAN HISTORY"
        )

        history_layout = QVBoxLayout(
            history_group
        )

        self.history_table = QTableWidget()

        self.history_table.setColumnCount(
            8
        )

        self.history_table.setHorizontalHeaderLabels(
            [
                "ID",
                "Time",
                "Result",
                "Fault Type",
                "Confidence",
                "Deviation",
                "Low Hz",
                "High Hz",
            ]
        )

        self.history_table.setEditTriggers(
            QTableWidget.NoEditTriggers
        )

        self.history_table.setSelectionBehavior(
            QTableWidget.SelectRows
        )

        self.history_table.setAlternatingRowColors(
            True
        )

        self.history_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )

        self.history_table.horizontalHeader().setStretchLastSection(
            True
        )

        history_layout.addWidget(
            self.history_table
        )

        main_layout.addWidget(
            history_group,
            stretch=2,
        )

        # =============================================================
        # STATUS
        # =============================================================

        self.status_label = QLabel(
            "SYSTEM STARTING..."
        )

        self.status_label.setAlignment(
            Qt.AlignCenter
        )

        self.status_label.setStyleSheet(
            """
            QLabel {
                padding: 8px;
                font-weight: bold;
                background-color: #20242b;
                color: white;
            }
            """
        )

        main_layout.addWidget(
            self.status_label
        )


    # =================================================================
    # RESULT STYLES
    # =================================================================

    def set_ready_result_style(self):

        self.result_label.setText(
            "READY"
        )

        self.result_label.setStyleSheet(
            """
            QLabel {
                font-size: 28px;
                font-weight: bold;
                border: 3px solid #777;
                border-radius: 5px;
                padding: 10px;
                background-color: #eeeeee;
                color: #222;
            }
            """
        )


    def set_healthy_result_style(self):

        self.result_label.setText(
            "PASS"
        )

        self.result_label.setStyleSheet(
            """
            QLabel {
                font-size: 30px;
                font-weight: bold;
                border: 3px solid #16803c;
                border-radius: 5px;
                padding: 10px;
                background-color: #dff5e5;
                color: #12652f;
            }
            """
        )


    def set_fault_result_style(self):

        self.result_label.setText(
            "FAIL"
        )

        self.result_label.setStyleSheet(
            """
            QLabel {
                font-size: 30px;
                font-weight: bold;
                border: 3px solid #c62828;
                border-radius: 5px;
                padding: 10px;
                background-color: #fde2e2;
                color: #9b1c1c;
            }
            """
        )


    def set_unknown_result_style(self):

        self.result_label.setText(
            "ABNORMAL"
        )

        self.result_label.setStyleSheet(
            """
            QLabel {
                font-size: 26px;
                font-weight: bold;
                border: 3px solid #e68a00;
                border-radius: 5px;
                padding: 10px;
                background-color: #fff0d6;
                color: #995c00;
            }
            """
        )


    # =================================================================
    # BUILD HEALTHY BASELINE
    # =================================================================

    def build_baseline(
        self,
        show_message: bool = True,
    ):

        try:

            self.scan_button.setEnabled(
                False
            )

            self.baseline_button.setEnabled(
                False
            )

            self.update_status(
                "BUILDING HEALTHY BASELINE..."
            )

            QApplication.processEvents()

            baseline_signatures = []

            # ---------------------------------------------------------
            # Generate healthy acquisition samples
            # ---------------------------------------------------------

            for i in range(
                BASELINE_COUNT
            ):

                signal = generate_known_good(

                    config=self.config,

                    board=self.board,

                    unit_variation=True,

                    seed=100 + i,
                )

                signature = compute_signature(

                    signal,

                    self.config.sample_rate_hz,

                    window_type="hanning",
                )

                baseline_signatures.append(
                    signature
                )

            # ---------------------------------------------------------
            # Create directory
            # ---------------------------------------------------------

            Path(
                BASELINE_FILE
            ).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ---------------------------------------------------------
            # Build average baseline
            # ---------------------------------------------------------

            self.baseline = build_average_baseline(

                baseline_signatures,

                BASELINE_FILE,
            )

            # ---------------------------------------------------------
            # Save baseline
            # ---------------------------------------------------------

            self.baseline_id = (
                self.database.save_baseline(

                    board_id=self.board_id,

                    name=(
                        "Healthy GUI Baseline"
                    ),

                    signature=self.baseline,

                    sample_count=BASELINE_COUNT,
                )
            )

            # ---------------------------------------------------------
            # Display baseline
            # ---------------------------------------------------------

            self.plot.clear()

            self.plot.addLegend()

            self.plot.plot(

                self.baseline.frequencies,

                self.baseline.magnitude,

                pen=pg.mkPen(
                    color=(0, 100, 220),
                    width=2,
                ),

                name="Healthy Baseline",
            )

            self.set_ready_result_style()

            self.fault_result_label.setText(
                "Fault: ---"
            )

            self.confidence_label.setText(
                "Confidence: ---"
            )

            self.deviation_label.setText(
                "Deviation: ---"
            )

            self.novelty_label.setText(
                "Novelty: ---"
            )

            self.band_label.setText(
                "Dominant deviation band: ---"
            )

            self.evidence_label.setText(
                "Healthy baseline successfully created."
            )

            self.update_status(
                "HEALTHY BASELINE READY"
            )

            if show_message:

                QMessageBox.information(

                    self,

                    "Baseline Complete",

                    (
                        "Healthy baseline rebuilt successfully.\n\n"
                        f"Samples: {BASELINE_COUNT}\n"
                        f"Database ID: {self.baseline_id}"
                    ),
                )

        except Exception as error:

            self.update_status(
                "BASELINE ERROR"
            )

            QMessageBox.critical(

                self,

                "Baseline Error",

                (
                    "Unable to build the healthy baseline.\n\n"
                    f"{error}"
                ),
            )

        finally:

            self.scan_button.setEnabled(
                True
            )

            self.baseline_button.setEnabled(
                True
            )


    # =================================================================
    # GENERATE TEST SIGNAL
    # =================================================================

    def generate_test_signal(
        self,
    ):

        selection = (
            self.fault_combo.currentText()
        )

        # -------------------------------------------------------------
        # Deterministic seed
        # -------------------------------------------------------------

        seed = TEST_SEEDS.get(
            selection,
            12345,
        )

        # -------------------------------------------------------------
        # Healthy
        # -------------------------------------------------------------

        if selection == "HEALTHY BOARD":

            return generate_known_good(

                config=self.config,

                board=self.board,

                unit_variation=True,

                seed=seed,
            )

        # -------------------------------------------------------------
        # Open trace
        # -------------------------------------------------------------

        if selection == "OPEN TRACE":

            return generate_open_trace_fault(

                config=self.config,

                board=self.board,

                band_index=1,

                seed=seed,
            )

        # -------------------------------------------------------------
        # Short circuit
        # -------------------------------------------------------------

        if selection == "SHORT CIRCUIT":

            return generate_short_circuit_fault(

                config=self.config,

                board=self.board,

                band_indices=(1, 2),

                seed=seed,
            )

        # -------------------------------------------------------------
        # Regulator fault
        # -------------------------------------------------------------

        if selection == "REGULATOR FAULT":

            return generate_regulator_fault(

                config=self.config,

                board=self.board,

                seed=seed,
            )

        # -------------------------------------------------------------
        # Capacitor drift
        # -------------------------------------------------------------

        if selection == "CAPACITOR DRIFT":

            return generate_capacitor_drift_fault(

                config=self.config,

                board=self.board,

                band_index=2,

                seed=seed,
            )

        raise ValueError(
            f"Unknown test profile: {selection}"
        )


    # =================================================================
    # RESET DISPLAY BEFORE SCAN
    # =================================================================

    def reset_result_display(self):

        self.set_ready_result_style()

        self.fault_result_label.setText(
            "Fault: ANALYZING..."
        )

        self.confidence_label.setText(
            "Confidence: ---"
        )

        self.deviation_label.setText(
            "Deviation: ---"
        )

        self.novelty_label.setText(
            "Novelty: ANALYZING..."
        )

        self.band_label.setText(
            "Dominant deviation band: ANALYZING..."
        )

        self.evidence_label.setText(
            "Analyzing spectral signature..."
        )

        QApplication.processEvents()


    # =================================================================
    # SIMULATE SCAN
    # =================================================================

    def simulate_scan(self):

        # -------------------------------------------------------------
        # Baseline check
        # -------------------------------------------------------------

        if self.baseline is None:

            QMessageBox.warning(

                self,

                "No Baseline",

                (
                    "No healthy baseline exists.\n\n"
                    "Build the baseline first."
                ),
            )

            return

        try:

            self.scan_button.setEnabled(
                False
            )

            self.baseline_button.setEnabled(
                False
            )

            self.reset_result_display()

            self.update_status(
                "RUNNING SIMULATED SCAN..."
            )

            # =========================================================
            # 1. Generate signal
            # =========================================================

            signal = (
                self.generate_test_signal()
            )

            self.update_status(
                "SIGNAL ACQUIRED"
            )

            QApplication.processEvents()

            # =========================================================
            # 2. Compute FFT signature
            # =========================================================

            live_signature = compute_signature(

                signal,

                self.config.sample_rate_hz,

                window_type="hanning",
            )

            self.update_status(
                "FFT SIGNATURE COMPUTED"
            )

            QApplication.processEvents()

            # =========================================================
            # 3. Compare against healthy baseline
            # =========================================================

            comparison = compare_signatures(

                self.baseline,

                live_signature,
            )

            self.update_status(
                "SIGNATURE COMPARISON COMPLETE"
            )

            QApplication.processEvents()

            # =========================================================
            # 4. Fault classification
            # =========================================================

            result = classify_fault(

                self.baseline,

                live_signature,

                comparison,
            )

            # =========================================================
            # 5. Save runtime objects
            # =========================================================

            self.last_live_signature = (
                live_signature
            )

            self.last_comparison = (
                comparison
            )

            self.last_result = (
                result
            )

            # =========================================================
            # 6. Log scan
            # =========================================================

            scan_id = self.database.log_scan(

                board_id=self.board_id,

                baseline_id=self.baseline_id,

                result=result,

                live_signature=live_signature,
            )

            # =========================================================
            # 7. Plot
            # =========================================================

            self.update_plot(
                live_signature
            )

            # =========================================================
            # 8. Display result
            # =========================================================

            self.update_result_display(
                result
            )

            # =========================================================
            # 9. History
            # =========================================================

            self.refresh_history()

            # =========================================================
            # 10. Final status
            # =========================================================

            self.update_status(

                f"SCAN COMPLETE | "
                f"SCAN ID: {scan_id} | "
                f"DIAGNOSIS: {result.label}"
            )

        except Exception as error:

            self.update_status(
                "SCAN ERROR"
            )

            QMessageBox.critical(

                self,

                "Scan Error",

                (
                    "The scan could not be completed.\n\n"
                    f"{error}"
                ),
            )

        finally:

            self.scan_button.setEnabled(
                True
            )

            self.baseline_button.setEnabled(
                True
            )


    # =================================================================
    # UPDATE PLOT
    # =================================================================

    def update_plot(
        self,
        live_signature,
    ):

        self.plot.clear()

        self.plot.addLegend()

        # -------------------------------------------------------------
        # Healthy baseline
        # -------------------------------------------------------------

        self.plot.plot(

            self.baseline.frequencies,

            self.baseline.magnitude,

            pen=pg.mkPen(
                color=(0, 100, 220),
                width=2,
            ),

            name="Healthy Baseline",
        )

        # -------------------------------------------------------------
        # Live signature
        # -------------------------------------------------------------

        self.plot.plot(

            live_signature.frequencies,

            live_signature.magnitude,

            pen=pg.mkPen(
                color=(220, 60, 60),
                width=2,
            ),

            name="Live Signature",
        )

        self.plot.setTitle(
            "Healthy Baseline vs Live Signature"
        )


    # =================================================================
    # UPDATE RESULT
    # =================================================================

    def update_result_display(
        self,
        result,
    ):

        label = str(
            result.label
        ).upper()

        confidence = float(
            result.confidence
        )

        deviation = float(
            result.overall_deviation
        )

        # -------------------------------------------------------------
        # Overall status
        # -------------------------------------------------------------

        if label == "HEALTHY":

            self.set_healthy_result_style()

        elif label == "UNKNOWN / ABNORMAL SIGNATURE":

            self.set_unknown_result_style()

        else:

            self.set_fault_result_style()

        # -------------------------------------------------------------
        # Fault label
        # -------------------------------------------------------------

        self.fault_result_label.setText(

            f"Fault: {result.label}"
        )

        # -------------------------------------------------------------
        # Confidence
        # -------------------------------------------------------------

        self.confidence_label.setText(

            f"Confidence: "
            f"{confidence * 100:.1f}%"
        )

        # -------------------------------------------------------------
        # Deviation
        # -------------------------------------------------------------

        self.deviation_label.setText(

            f"Deviation: "
            f"{deviation:.5f}"
        )

        # -------------------------------------------------------------
        # Novelty / abnormality
        # -------------------------------------------------------------

        if result.abnormal:

            self.novelty_label.setText(
                "Novelty: ABNORMAL"
            )

            self.novelty_label.setStyleSheet(
                """
                QLabel {
                    font-size: 17px;
                    font-weight: bold;
                    border: 2px solid #c62828;
                    border-radius: 5px;
                    padding: 10px;
                    background-color: #fde2e2;
                    color: #9b1c1c;
                }
                """
            )

        else:

            self.novelty_label.setText(
                "Novelty: NORMAL"
            )

            self.novelty_label.setStyleSheet(
                """
                QLabel {
                    font-size: 17px;
                    font-weight: bold;
                    border: 2px solid #16803c;
                    border-radius: 5px;
                    padding: 10px;
                    background-color: #dff5e5;
                    color: #12652f;
                }
                """
            )

        # -------------------------------------------------------------
        # Evidence
        # -------------------------------------------------------------

        evidence_text = (
            "<b>Diagnostic Evidence</b><br><br>"
        )

        if result.evidence:

            for index, item in enumerate(
                result.evidence,
                start=1,
            ):

                evidence_text += (
                    f"{index}. {item}<br>"
                )

        else:

            evidence_text += (
                "No additional evidence available."
            )

        self.evidence_label.setText(
            evidence_text
        )

        # -------------------------------------------------------------
        # Dominant band
        # -------------------------------------------------------------

        if result.abnormal:

            low = float(
                result.dominant_band_low_hz
            )

            high = float(
                result.dominant_band_high_hz
            )

            self.band_label.setText(

                "<b>Dominant deviation band</b><br><br>"
                f"Low frequency : {low:.0f} Hz<br>"
                f"High frequency: {high:.0f} Hz<br><br>"
                f"Bandwidth     : {max(0.0, high-low):.0f} Hz"
            )

        else:

            self.band_label.setText(
                "<b>Dominant deviation band</b><br><br>"
                "No abnormal deviation band detected."
            )


    # =================================================================
    # REFRESH HISTORY
    # =================================================================

    def refresh_history(self):

        try:

            df = (
                self.database.get_scan_history(
                    self.board_id
                )
            )

            self.history_table.setRowCount(
                len(df)
            )

            for row_index, row in df.iterrows():

                values = [

                    str(
                        row["scan_id"]
                    ),

                    str(
                        row["scan_time"]
                    ),

                    str(
                        row["result"]
                    ),

                    str(
                        row["fault_type"]
                    ),

                    (
                        f"{float(row['confidence']) * 100:.1f}%"
                    ),

                    (
                        f"{float(row['overall_deviation']):.5f}"
                    ),

                    (
                        f"{float(row['dominant_band_low_hz']):.0f}"
                    ),

                    (
                        f"{float(row['dominant_band_high_hz']):.0f}"
                    ),
                ]

                for column_index, value in enumerate(
                    values
                ):

                    item = QTableWidgetItem(
                        value
                    )

                    item.setTextAlignment(
                        Qt.AlignCenter
                    )

                    self.history_table.setItem(

                        row_index,

                        column_index,

                        item,
                    )

            self.history_table.resizeColumnsToContents()

        except Exception as error:

            print(
                "History refresh error:",
                error,
            )


    # =================================================================
    # EXPORT CSV
    # =================================================================

    def export_csv(self):

        try:

            output = (
                self.database.export_csv(

                    "exports/scan_history.csv",

                    self.board_id,
                )
            )

            QMessageBox.information(

                self,

                "CSV Export Complete",

                (
                    "Scan history exported successfully.\n\n"
                    f"{output}"
                ),
            )

            self.update_status(
                "CSV EXPORT COMPLETE"
            )

        except Exception as error:

            QMessageBox.critical(

                self,

                "CSV Export Error",

                str(error),
            )


    # =================================================================
    # EXPORT EXCEL
    # =================================================================

    def export_excel(self):

        try:

            output = (
                self.database.export_excel(

                    "exports/scan_history.xlsx",

                    self.board_id,
                )
            )

            QMessageBox.information(

                self,

                "Excel Export Complete",

                (
                    "Scan history exported successfully.\n\n"
                    f"{output}"
                ),
            )

            self.update_status(
                "EXCEL EXPORT COMPLETE"
            )

        except Exception as error:

            QMessageBox.critical(

                self,

                "Excel Export Error",

                str(error),
            )


    # =================================================================
    # STATUS
    # =================================================================

    def update_status(
        self,
        message: str,
    ):

        self.status_label.setText(
            message
        )


    # =================================================================
    # CLOSE EVENT
    # =================================================================

    def closeEvent(
        self,
        event,
    ):

        try:

            self.database.close()

        except Exception:

            pass

        event.accept()


# =====================================================================
# APPLICATION ENTRY POINT
# =====================================================================

def main():

    app = QApplication(
        sys.argv
    )

    app.setApplicationName(
        APP_TITLE
    )

    window = PCBHealthScannerGUI()

    window.show()

    sys.exit(
        app.exec_()
    )


if __name__ == "__main__":

    main()