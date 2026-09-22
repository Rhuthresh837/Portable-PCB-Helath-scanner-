"""
signature_database.py

SQLite persistence layer for the Portable PCB Health Scanner.

Stores:

    - PCB board metadata
    - Healthy baseline signatures
    - Scan history
    - Live FFT signatures
    - Classification results

Provides:

    - SQLite database
    - Pandas DataFrame scan history
    - CSV export
    - Excel export

Database file:

    pcb_scanner.db
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# CONFIGURATION
# =====================================================================

DEFAULT_DATABASE = "pcb_scanner.db"


# =====================================================================
# DATABASE CLASS
# =====================================================================

class SignatureDatabase:

    def __init__(
        self,
        database_path: str = DEFAULT_DATABASE,
    ):

        self.database_path = Path(
            database_path
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            str(self.database_path)
        )

        self.connection.row_factory = (
            sqlite3.Row
        )

        self.create_tables()


    # =================================================================
    # CREATE TABLES
    # =================================================================

    def create_tables(self):

        cursor = self.connection.cursor()

        # -------------------------------------------------------------
        # Boards
        # -------------------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS boards
            (
                board_id INTEGER PRIMARY KEY AUTOINCREMENT,

                board_name TEXT NOT NULL,

                board_serial TEXT UNIQUE,

                description TEXT,

                created_at TEXT NOT NULL
            )
            """
        )

        # -------------------------------------------------------------
        # Baselines
        # -------------------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS baselines
            (
                baseline_id INTEGER PRIMARY KEY AUTOINCREMENT,

                board_id INTEGER NOT NULL,

                name TEXT NOT NULL,

                sample_count INTEGER,

                frequencies TEXT NOT NULL,

                magnitude TEXT NOT NULL,

                phase TEXT NOT NULL,

                created_at TEXT NOT NULL,

                FOREIGN KEY(board_id)
                    REFERENCES boards(board_id)
            )
            """
        )

        # -------------------------------------------------------------
        # Scans
        # -------------------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scans
            (
                scan_id INTEGER PRIMARY KEY AUTOINCREMENT,

                board_id INTEGER NOT NULL,

                baseline_id INTEGER,

                scan_time TEXT NOT NULL,

                result TEXT NOT NULL,

                fault_type TEXT NOT NULL,

                confidence REAL,

                overall_deviation REAL,

                dominant_band_low_hz REAL,

                dominant_band_high_hz REAL,

                evidence TEXT,

                FOREIGN KEY(board_id)
                    REFERENCES boards(board_id),

                FOREIGN KEY(baseline_id)
                    REFERENCES baselines(baseline_id)
            )
            """
        )

        # -------------------------------------------------------------
        # Scan signatures
        # -------------------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_signatures
            (
                signature_id INTEGER PRIMARY KEY AUTOINCREMENT,

                scan_id INTEGER NOT NULL,

                frequencies TEXT NOT NULL,

                magnitude TEXT NOT NULL,

                phase TEXT NOT NULL,

                FOREIGN KEY(scan_id)
                    REFERENCES scans(scan_id)
            )
            """
        )

        self.connection.commit()


    # =================================================================
    # CLOSE DATABASE
    # =================================================================

    def close(self):

        if self.connection:

            self.connection.close()


    # =================================================================
    # BOARD MANAGEMENT
    # =================================================================

    def add_board(
        self,
        board_name: str,
        board_serial: str | None = None,
        description: str = "",
    ) -> int:

        cursor = self.connection.cursor()

        created_at = datetime.now().isoformat(
            timespec="seconds"
        )

        cursor.execute(
            """
            INSERT OR IGNORE INTO boards
            (
                board_name,
                board_serial,
                description,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                board_name,
                board_serial,
                description,
                created_at,
            ),
        )

        self.connection.commit()

        if board_serial is None:

            cursor.execute(
                """
                SELECT board_id
                FROM boards
                WHERE board_name = ?
                AND board_serial IS NULL
                """,
                (
                    board_name,
                ),
            )

        else:

            cursor.execute(
                """
                SELECT board_id
                FROM boards
                WHERE board_serial = ?
                """,
                (
                    board_serial,
                ),
            )

        row = cursor.fetchone()

        if row is None:

            raise RuntimeError(
                "Unable to create/retrieve board."
            )

        return int(
            row["board_id"]
        )


    def get_boards(self):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT
                board_id,
                board_name,
                board_serial,
                description,
                created_at
            FROM boards
            ORDER BY board_id
            """
        )

        return [
            dict(row)
            for row in cursor.fetchall()
        ]


    # =================================================================
    # BASELINE STORAGE
    # =================================================================

    def save_baseline(
        self,
        board_id: int,
        name: str,
        signature,
        sample_count: int,
    ) -> int:

        cursor = self.connection.cursor()

        created_at = datetime.now().isoformat(
            timespec="seconds"
        )

        frequencies = json.dumps(
            np.asarray(
                signature.frequencies
            ).tolist()
        )

        magnitude = json.dumps(
            np.asarray(
                signature.magnitude
            ).tolist()
        )

        phase = json.dumps(
            np.asarray(
                signature.phase
            ).tolist()
        )

        cursor.execute(
            """
            INSERT INTO baselines
            (
                board_id,
                name,
                sample_count,
                frequencies,
                magnitude,
                phase,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                board_id,
                name,
                sample_count,
                frequencies,
                magnitude,
                phase,
                created_at,
            ),
        )

        self.connection.commit()

        return int(
            cursor.lastrowid
        )


    def get_latest_baseline(
        self,
        board_id: int,
    ):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM baselines
            WHERE board_id = ?
            ORDER BY baseline_id DESC
            LIMIT 1
            """,
            (
                board_id,
            ),
        )

        row = cursor.fetchone()

        if row is None:

            return None

        return dict(row)


    # =================================================================
    # LOG SCAN
    # =================================================================

    def log_scan(
        self,
        board_id: int,
        baseline_id: int | None,
        result,
        live_signature,
    ) -> int:

        cursor = self.connection.cursor()

        scan_time = datetime.now().isoformat(
            timespec="seconds"
        )

        status = (
            "PASS"
            if result.label == "HEALTHY"
            else "FAIL"
        )

        evidence = json.dumps(
            result.evidence
        )

        cursor.execute(
            """
            INSERT INTO scans
            (
                board_id,
                baseline_id,
                scan_time,
                result,
                fault_type,
                confidence,
                overall_deviation,
                dominant_band_low_hz,
                dominant_band_high_hz,
                evidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                board_id,
                baseline_id,
                scan_time,
                status,
                result.label,
                float(
                    result.confidence
                ),
                float(
                    result.overall_deviation
                ),
                float(
                    result.dominant_band_low_hz
                ),
                float(
                    result.dominant_band_high_hz
                ),
                evidence,
            ),
        )

        scan_id = int(
            cursor.lastrowid
        )

        # -------------------------------------------------------------
        # Store FFT signature
        # -------------------------------------------------------------

        frequencies = json.dumps(
            np.asarray(
                live_signature.frequencies
            ).tolist()
        )

        magnitude = json.dumps(
            np.asarray(
                live_signature.magnitude
            ).tolist()
        )

        phase = json.dumps(
            np.asarray(
                live_signature.phase
            ).tolist()
        )

        cursor.execute(
            """
            INSERT INTO scan_signatures
            (
                scan_id,
                frequencies,
                magnitude,
                phase
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                scan_id,
                frequencies,
                magnitude,
                phase,
            ),
        )

        self.connection.commit()

        return scan_id


    # =================================================================
    # SCAN HISTORY
    # =================================================================

    def get_scan_history(
        self,
        board_id: int | None = None,
    ) -> pd.DataFrame:

        if board_id is None:

            query = """
                SELECT
                    s.scan_id,
                    b.board_name,
                    b.board_serial,
                    s.scan_time,
                    s.result,
                    s.fault_type,
                    s.confidence,
                    s.overall_deviation,
                    s.dominant_band_low_hz,
                    s.dominant_band_high_hz,
                    s.evidence
                FROM scans s
                JOIN boards b
                    ON s.board_id = b.board_id
                ORDER BY s.scan_id DESC
            """

            return pd.read_sql_query(
                query,
                self.connection,
            )

        query = """
            SELECT
                s.scan_id,
                b.board_name,
                b.board_serial,
                s.scan_time,
                s.result,
                s.fault_type,
                s.confidence,
                s.overall_deviation,
                s.dominant_band_low_hz,
                s.dominant_band_high_hz,
                s.evidence
            FROM scans s
            JOIN boards b
                ON s.board_id = b.board_id
            WHERE s.board_id = ?
            ORDER BY s.scan_id DESC
        """

        return pd.read_sql_query(
            query,
            self.connection,
            params=(board_id,),
        )


    # =================================================================
    # CSV EXPORT
    # =================================================================

    def export_csv(
        self,
        output_path: str,
        board_id: int | None = None,
    ) -> Path:

        df = self.get_scan_history(
            board_id
        )

        output = Path(
            output_path
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            output,
            index=False,
        )

        return output


    # =================================================================
    # EXCEL EXPORT
    # =================================================================

    def export_excel(
        self,
        output_path: str,
        board_id: int | None = None,
    ) -> Path:

        df = self.get_scan_history(
            board_id
        )

        output = Path(
            output_path
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with pd.ExcelWriter(
            output,
            engine="openpyxl",
        ) as writer:

            df.to_excel(
                writer,
                sheet_name="Scan History",
                index=False,
            )

            # ---------------------------------------------------------
            # Summary
            # ---------------------------------------------------------

            if not df.empty:

                summary = pd.DataFrame(
                    {
                        "Metric": [
                            "Total scans",
                            "PASS scans",
                            "FAIL scans",
                            "Average confidence",
                            "Average deviation",
                        ],

                        "Value": [

                            len(df),

                            int(
                                (
                                    df["result"]
                                    ==
                                    "PASS"
                                ).sum()
                            ),

                            int(
                                (
                                    df["result"]
                                    ==
                                    "FAIL"
                                ).sum()
                            ),

                            float(
                                df[
                                    "confidence"
                                ].mean()
                            ),

                            float(
                                df[
                                    "overall_deviation"
                                ].mean()
                            ),
                        ],
                    }
                )

            else:

                summary = pd.DataFrame(
                    {
                        "Metric": [
                            "Total scans",
                            "PASS scans",
                            "FAIL scans",
                        ],

                        "Value": [
                            0,
                            0,
                            0,
                        ],
                    }
                )

            summary.to_excel(
                writer,
                sheet_name="Summary",
                index=False,
            )

        return output


    # =================================================================
    # STATISTICS
    # =================================================================

    def get_statistics(
        self,
        board_id: int | None = None,
    ) -> dict:

        df = self.get_scan_history(
            board_id
        )

        if df.empty:

            return {
                "total": 0,
                "pass": 0,
                "fail": 0,
                "average_confidence": 0.0,
                "average_deviation": 0.0,
            }

        return {

            "total":
                len(df),

            "pass":
                int(
                    (
                        df["result"]
                        ==
                        "PASS"
                    ).sum()
                ),

            "fail":
                int(
                    (
                        df["result"]
                        ==
                        "FAIL"
                    ).sum()
                ),

            "average_confidence":
                float(
                    df[
                        "confidence"
                    ].mean()
                ),

            "average_deviation":
                float(
                    df[
                        "overall_deviation"
                    ].mean()
                ),
        }