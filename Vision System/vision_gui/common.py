"""Shared widgets and small helpers for the vision GUI."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


class StatusBadge(QLabel):
	COLORS = {
		"good": (34, 139, 94),
		"warning": (184, 119, 18),
		"error": (180, 55, 55),
		"neutral": (90, 99, 110),
	}

	def __init__(self, text: str = "—", parent=None) -> None:
		super().__init__(text, parent)
		self.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.setMinimumWidth(100)
		self.set_status(text, "neutral")

	def set_status(self, text: str, kind: str = "neutral") -> None:
		red, green, blue = self.COLORS.get(kind, self.COLORS["neutral"])
		self.setText(text)
		self.setStyleSheet(
			f"QLabel {{ color: white; background: rgb({red},{green},{blue}); "
			"border-radius: 5px; padding: 4px 8px; font-weight: 600; }"
		)


class VideoLabel(QLabel):
	"""Aspect-fit video label that maps clicks back to source pixels."""

	frame_clicked = Signal(int, int)

	def __init__(self, parent=None) -> None:
		super().__init__(parent)
		self._image: QImage | None = None
		self._target_rect = QRect()
		self.setMinimumSize(480, 360)
		self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
		self.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.setText("Camera stopped")
		self.setStyleSheet("background: #15181d; color: #aab2bd; border: 1px solid #343a43;")

	def set_bgr_frame(self, frame) -> None:
		if frame is None or getattr(frame, "ndim", 0) != 3 or frame.shape[2] != 3:
			return
		height, width = frame.shape[:2]
		image = QImage(
			frame.data, width, height, int(frame.strides[0]), QImage.Format.Format_BGR888
		)
		self._image = image.copy()
		self._render()

	def clear_frame(self, text: str = "Camera stopped") -> None:
		self._image = None
		self._target_rect = QRect()
		self.clear()
		self.setText(text)

	def _render(self) -> None:
		if self._image is None:
			return
		available = self.contentsRect().size()
		scaled_size = self._image.size().scaled(
			available, Qt.AspectRatioMode.KeepAspectRatio
		)
		left = self.contentsRect().left() + (available.width() - scaled_size.width()) // 2
		top = self.contentsRect().top() + (available.height() - scaled_size.height()) // 2
		self._target_rect = QRect(QPoint(left, top), scaled_size)
		pixmap = QPixmap.fromImage(self._image).scaled(
			scaled_size,
			Qt.AspectRatioMode.KeepAspectRatio,
			Qt.TransformationMode.SmoothTransformation,
		)
		self.setPixmap(pixmap)

	def resizeEvent(self, event) -> None:
		super().resizeEvent(event)
		self._render()

	def mousePressEvent(self, event: QMouseEvent) -> None:
		if (
			event.button() == Qt.MouseButton.LeftButton
			and self._image is not None
			and self._target_rect.contains(event.position().toPoint())
		):
			position = event.position()
			x = int(
				(position.x() - self._target_rect.left())
				* self._image.width() / max(1, self._target_rect.width())
			)
			y = int(
				(position.y() - self._target_rect.top())
				* self._image.height() / max(1, self._target_rect.height())
			)
			self.frame_clicked.emit(
				min(max(x, 0), self._image.width() - 1),
				min(max(y, 0), self._image.height() - 1),
			)
		super().mousePressEvent(event)

	def sizeHint(self) -> QSize:
		return QSize(720, 540)


def set_badge_boolean(badge: StatusBadge, valid: bool, good: str, bad: str) -> None:
	badge.set_status(good if valid else bad, "good" if valid else "error")
