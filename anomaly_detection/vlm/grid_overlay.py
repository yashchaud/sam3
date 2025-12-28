"""Grid overlay system for VLM input images."""

from dataclasses import dataclass
import numpy as np
import cv2

from .models import GridConfig


@dataclass(frozen=True)
class GridCell:
    """Represents a single grid cell."""
    col: int
    row: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    center_x: int
    center_y: int
    label: str

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    @property
    def center(self) -> tuple[int, int]:
        return (self.center_x, self.center_y)

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min


class GridOverlay:
    """Manages grid overlay on images for VLM processing."""

    def __init__(self, config: GridConfig | None = None):
        self.config = config or GridConfig()
        self._cells: list[GridCell] = []
        self._image_width = 0
        self._image_height = 0
        self._cell_width = 0
        self._cell_height = 0

    def compute_grid(self, image_width: int, image_height: int) -> list[GridCell]:
        """Compute grid cells for given image dimensions."""
        self._image_width = image_width
        self._image_height = image_height
        self._cell_width = image_width // self.config.cols
        self._cell_height = image_height // self.config.rows

        self._cells = []
        for row in range(self.config.rows):
            for col in range(self.config.cols):
                x_min = col * self._cell_width
                y_min = row * self._cell_height

                # Handle last column/row to capture remaining pixels
                if col == self.config.cols - 1:
                    x_max = image_width
                else:
                    x_max = (col + 1) * self._cell_width

                if row == self.config.rows - 1:
                    y_max = image_height
                else:
                    y_max = (row + 1) * self._cell_height

                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2

                # Label format: A1, A2, B1, B2, etc.
                label = f"{chr(65 + row)}{col + 1}"

                cell = GridCell(
                    col=col,
                    row=row,
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    center_x=center_x,
                    center_y=center_y,
                    label=label,
                )
                self._cells.append(cell)

        return self._cells

    def get_cell(self, col: int, row: int) -> GridCell | None:
        """Get specific grid cell by column and row."""
        for cell in self._cells:
            if cell.col == col and cell.row == row:
                return cell
        return None

    def get_cell_by_label(self, label: str) -> GridCell | None:
        """Get grid cell by label (e.g., 'A1', 'B2')."""
        for cell in self._cells:
            if cell.label.upper() == label.upper():
                return cell
        return None

    def point_to_cell(self, x: int, y: int) -> GridCell | None:
        """Find which cell contains a given point."""
        for cell in self._cells:
            if cell.x_min <= x < cell.x_max and cell.y_min <= y < cell.y_max:
                return cell
        return None

    def cell_label_to_center_point(self, label: str) -> tuple[int, int] | None:
        """Convert grid cell label to center point coordinates."""
        cell = self.get_cell_by_label(label)
        if cell:
            return cell.center
        return None

    def cell_label_to_box(self, label: str) -> tuple[int, int, int, int] | None:
        """Convert grid cell label to bounding box."""
        cell = self.get_cell_by_label(label)
        if cell:
            return cell.bounds
        return None

    def parse_cell_reference(self, ref: str) -> tuple[int, int] | None:
        """
        Parse cell reference like 'A1', 'B2' to (col, row).
        Returns None if invalid.
        """
        ref = ref.strip().upper()
        if len(ref) < 2:
            return None

        row_char = ref[0]
        col_str = ref[1:]

        if not row_char.isalpha() or not col_str.isdigit():
            return None

        row = ord(row_char) - ord('A')
        col = int(col_str) - 1

        if 0 <= row < self.config.rows and 0 <= col < self.config.cols:
            return (col, row)
        return None

    def draw_grid(self, image: np.ndarray) -> np.ndarray:
        """Draw grid overlay on image."""
        if len(self._cells) == 0:
            self.compute_grid(image.shape[1], image.shape[0])

        output = image.copy()

        # Draw vertical lines
        for col in range(1, self.config.cols):
            x = col * self._cell_width
            cv2.line(
                output,
                (x, 0),
                (x, self._image_height),
                self.config.line_color,
                self.config.line_thickness,
            )

        # Draw horizontal lines
        for row in range(1, self.config.rows):
            y = row * self._cell_height
            cv2.line(
                output,
                (0, y),
                (self._image_width, y),
                self.config.line_color,
                self.config.line_thickness,
            )

        # Draw labels if enabled
        if self.config.show_labels:
            for cell in self._cells:
                label_x = cell.x_min + 10
                label_y = cell.y_min + 30

                # Draw background for better visibility
                (text_w, text_h), _ = cv2.getTextSize(
                    cell.label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.label_font_scale,
                    2,
                )
                cv2.rectangle(
                    output,
                    (label_x - 2, label_y - text_h - 4),
                    (label_x + text_w + 2, label_y + 4),
                    (0, 0, 0),
                    -1,
                )

                cv2.putText(
                    output,
                    cell.label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.label_font_scale,
                    self.config.label_color,
                    2,
                )

        return output

    def get_grid_description(self) -> str:
        """Get text description of grid for VLM prompt."""
        lines = [
            f"The image has a {self.config.cols}x{self.config.rows} grid overlay.",
            "Grid cells are labeled with letters (rows) and numbers (columns):",
        ]

        for row in range(self.config.rows):
            row_labels = []
            for col in range(self.config.cols):
                cell = self.get_cell(col, row)
                if cell:
                    row_labels.append(cell.label)
            lines.append(f"  Row {chr(65 + row)}: {', '.join(row_labels)}")

        return "\n".join(lines)

    def get_sam_input_points(self, cell_labels: list[str]) -> tuple[list[list[list[int]]], list[list[list[int]]]]:
        """
        Convert cell labels to SAM input format.
        Returns (input_points, input_labels) for SAM.
        """
        points = []
        labels = []

        for label in cell_labels:
            center = self.cell_label_to_center_point(label)
            if center:
                points.append([[center[0], center[1]]])
                labels.append([[1]])  # Positive point

        return points, labels

    @property
    def cells(self) -> list[GridCell]:
        return self._cells

    @property
    def cell_width(self) -> int:
        return self._cell_width

    @property
    def cell_height(self) -> int:
        return self._cell_height
