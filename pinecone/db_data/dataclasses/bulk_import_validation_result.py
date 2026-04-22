from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BulkImportValidationResult:
    """Result of a bulk import parquet validation check.

    Attributes:
        is_valid: True if no errors were found.
        uri: The URI that was validated. Pass directly to ``index.bulk_import.start()``.
        errors: Blocking issues that would cause the import to fail.
        warnings: Non-blocking observations (e.g. detected dimension).
        files_checked: Number of parquet files whose schema was inspected.
        rows_sampled: Number of data rows checked (0 if schema-only validation).
    """

    is_valid: bool
    uri: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files_checked: int = 0
    rows_sampled: int = 0

    def __repr__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        lines = [f"BulkImportValidationResult({status})"]
        if self.uri:
            lines.append(f"  uri={self.uri!r}")
        if self.errors:
            lines.append(f"  errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append(f"  warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    - {w}")
        lines.append(
            f"  files_checked={self.files_checked}, rows_sampled={self.rows_sampled}"
        )
        return "\n".join(lines)
