"""
Console utilities with ANSI color codes

Usage:
    from console import console, Colors

    console.info("Loading model...")
    console.success("Model loaded")
    console.warning("Low memory")
    console.error("Failed to connect")
    console.header("EVA Pipeline")
    console.step("Processing audio", 1, 3)
"""

import sys


class Colors:
    """ANSI color codes"""
    # Reset
    RESET = "\033[0m"

    # Regular colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    @classmethod
    def disable(cls):
        """Disable colors for non-TTY output"""
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                setattr(cls, attr, "")


# Disable colors if not a TTY
if not sys.stdout.isatty():
    Colors.disable()


class Console:
    """Formatted console output without emojis"""

    def __init__(self):
        self.c = Colors

    def _print(self, prefix: str, color: str, message: str, indent: int = 0):
        """Print formatted message"""
        spaces = "  " * indent
        print(f"{spaces}{color}{prefix}{self.c.RESET} {message}")

    # Status messages
    def info(self, message: str, indent: int = 0):
        """Info message - blue [*]"""
        self._print("[*]", self.c.BLUE, message, indent)

    def success(self, message: str, indent: int = 0):
        """Success message - green [+]"""
        self._print("[+]", self.c.GREEN, message, indent)

    def warning(self, message: str, indent: int = 0):
        """Warning message - yellow [!]"""
        self._print("[!]", self.c.YELLOW, message, indent)

    def error(self, message: str, indent: int = 0):
        """Error message - red [-]"""
        self._print("[-]", self.c.RED, message, indent)

    def debug(self, message: str, indent: int = 0):
        """Debug message - dim [.]"""
        self._print("[.]", self.c.DIM, message, indent)

    # Structural elements
    def header(self, title: str, width: int = 60):
        """Print a header box"""
        print()
        print(f"{self.c.CYAN}{'=' * width}{self.c.RESET}")
        print(f"{self.c.BOLD}{self.c.CYAN}{title.center(width)}{self.c.RESET}")
        print(f"{self.c.CYAN}{'=' * width}{self.c.RESET}")

    def subheader(self, title: str, width: int = 60):
        """Print a subheader"""
        print()
        print(f"{self.c.CYAN}{'-' * width}{self.c.RESET}")
        print(f"{self.c.CYAN}{title}{self.c.RESET}")
        print(f"{self.c.CYAN}{'-' * width}{self.c.RESET}")

    def divider(self, width: int = 60, char: str = "-"):
        """Print a divider line"""
        print(f"{self.c.DIM}{char * width}{self.c.RESET}")

    # Progress indicators
    def step(self, message: str, current: int = None, total: int = None):
        """Print a step indicator"""
        if current is not None and total is not None:
            prefix = f"[{current}/{total}]"
        else:
            prefix = "[>]"
        self._print(prefix, self.c.MAGENTA, message)

    def item(self, label: str, value: str, indent: int = 1):
        """Print a labeled item"""
        spaces = "  " * indent
        print(f"{spaces}{self.c.DIM}{label}:{self.c.RESET} {value}")

    def list_item(self, message: str, indent: int = 1):
        """Print a list item"""
        spaces = "  " * indent
        print(f"{spaces}{self.c.DIM}-{self.c.RESET} {message}")

    # Special formatting
    def result(self, label: str, value: str, success: bool = True):
        """Print a result with status"""
        color = self.c.GREEN if success else self.c.RED
        status = "[+]" if success else "[-]"
        print(f"{color}{status}{self.c.RESET} {label}: {self.c.BOLD}{value}{self.c.RESET}")

    def progress_bar(self, current: int, total: int, width: int = 30, label: str = ""):
        """Print a progress bar"""
        filled = int(width * current / total)
        bar = "█" * filled + "░" * (width - filled)
        percent = current / total * 100
        label_str = f" {label}" if label else ""
        print(f"\r{self.c.CYAN}[{bar}]{self.c.RESET} {percent:5.1f}%{label_str}", end="", flush=True)
        if current >= total:
            print()

    def table_row(self, cols: list, widths: list = None):
        """Print a table row"""
        if widths is None:
            widths = [15] * len(cols)
        row = ""
        for col, width in zip(cols, widths):
            row += f"{str(col):<{width}}"
        print(f"  {row}")

    # Emotion-specific (for EVA)
    def emotion(self, name: str, score: float, bar_width: int = 20):
        """Print emotion with bar"""
        filled = int(bar_width * score)
        bar = "█" * filled + "░" * (bar_width - filled)
        percent = score * 100
        print(f"  {name:12s} [{self.c.CYAN}{bar}{self.c.RESET}] {percent:5.1f}%")


# Global instance
console = Console()


# Convenience functions for direct import
def info(msg, indent=0): console.info(msg, indent)
def success(msg, indent=0): console.success(msg, indent)
def warning(msg, indent=0): console.warning(msg, indent)
def error(msg, indent=0): console.error(msg, indent)
def header(title, width=60): console.header(title, width)


if __name__ == "__main__":
    # Demo
    console.header("EVA Console Demo")

    console.info("This is an info message")
    console.success("This is a success message")
    console.warning("This is a warning message")
    console.error("This is an error message")
    console.debug("This is a debug message")

    print()
    console.step("Processing step", 1, 3)
    console.step("Another step", 2, 3)
    console.step("Final step", 3, 3)

    print()
    console.item("Model", "whisper-base")
    console.item("Language", "vi")
    console.item("Device", "cuda")

    console.subheader("Emotion Analysis")

    console.emotion("happy", 0.85)
    console.emotion("sad", 0.10)
    console.emotion("angry", 0.03)
    console.emotion("neutral", 0.02)

    print()
    console.result("Pipeline", "Ready", success=True)
    console.result("LLM", "Not available", success=False)

    print()
    for i in range(101):
        console.progress_bar(i, 100, label="Loading")
        import time
        time.sleep(0.01)