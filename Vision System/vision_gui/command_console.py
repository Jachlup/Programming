"""Compact command registry console and structured runtime log."""

from __future__ import annotations

import html

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
	QCompleter,
	QHBoxLayout,
	QLineEdit,
	QPushButton,
	QTextEdit,
	QVBoxLayout,
	QWidget,
)

from commands import COMMANDS


class CommandConsole(QWidget):
	COLORS = {
		"command": "#8ec5ff",
		"success": "#79d69d",
		"info": "#d7dde5",
		"warning": "#ffc66d",
		"error": "#ff7b7b",
		"debug": "#89919d",
	}

	def __init__(self, controller, parent=None) -> None:
		super().__init__(parent)
		self.controller = controller
		self.output = QTextEdit()
		self.output.setReadOnly(True)
		self.output.document().setMaximumBlockCount(1000)
		self.output.setPlaceholderText("Commands, warnings, and errors appear here.")
		self.output.setMinimumHeight(120)

		self.command = QLineEdit()
		self.command.setPlaceholderText("Enter any registered command; type help for a list")
		completer = QCompleter(sorted(COMMANDS), self.command)
		completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
		completer.setFilterMode(completer.filterMode())
		self.command.setCompleter(completer)
		self.run = QPushButton("Run")
		self.clear = QPushButton("Clear log")
		self.run.clicked.connect(self._execute)
		self.command.returnPressed.connect(self._execute)
		self.clear.clicked.connect(self.output.clear)

		entry = QHBoxLayout()
		entry.addWidget(self.command, 1)
		entry.addWidget(self.run)
		entry.addWidget(self.clear)
		layout = QVBoxLayout(self)
		layout.setContentsMargins(4, 4, 4, 4)
		layout.addWidget(self.output, 1)
		layout.addLayout(entry)

		controller.log_emitted.connect(self.append_log)
		controller.command_completed.connect(self.command_completed)

	def _execute(self) -> None:
		line = self.command.text().strip()
		if not line:
			return
		self.controller.execute(line)
		self.command.clear()

	def append_log(self, level: str, message: str) -> None:
		color = self.COLORS.get(level, self.COLORS["info"])
		label = level.upper() if level not in {"info", "command"} else ""
		prefix = f"[{label}] " if label else ""
		self.output.append(
			f'<span style="color:{color}">{html.escape(prefix + str(message))}</span>'
		)
		self.output.moveCursor(QTextCursor.MoveOperation.End)

	def command_completed(self, command_line: str, success: bool) -> None:
		if command_line.strip():
			self.append_log(
				"success" if success else "error",
				"Command completed successfully." if success else "Command failed.",
			)
