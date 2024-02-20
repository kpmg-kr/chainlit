import os

from chainlit.logger import logger

# Default chainlit.md file created if none exists
DEFAULT_MARKDOWN_STR = """# README 🤖
"""

# Default chainlit_lighthouse.md file created if none exists
DEFAULT_MARKDOWN_LIGHTHOUSE_STR = """# Lighthouse README 🤖
"""


def init_markdown(root: str):
    """Initialize the chainlit.md file if it doesn't exist."""
    chainlit_md_file = os.path.join(root, "chainlit.md")

    if not os.path.exists(chainlit_md_file):
        with open(chainlit_md_file, "w", encoding="utf-8") as f:
            f.write(DEFAULT_MARKDOWN_STR)
            logger.info(f"Created default chainlit markdown file at {chainlit_md_file}")


def init_markdown_lighthouse(root: str):
    """
    Lighthouse
    """
    """Initialize the chainlit_lighthouse.md file if it doesn't exist."""
    chainlit_md_file = os.path.join(root, "chainlit_lighthouse.md")

    if not os.path.exists(chainlit_md_file):
        with open(chainlit_md_file, "w", encoding="utf-8") as f:
            f.write(DEFAULT_MARKDOWN_LIGHTHOUSE_STR)
            logger.info(f"Created default chainlit markdown file at {chainlit_md_file}")


def get_markdown_str(root: str):
    """Get the chainlit.md file as a string."""
    chainlit_md_path = os.path.join(root, "chainlit.md")
    if os.path.exists(chainlit_md_path):
        with open(chainlit_md_path, "r", encoding="utf-8") as f:
            chainlit_md = f.read()
            return chainlit_md
    else:
        return None


def get_markdown_lighthouse_str(root: str):
    """Get the chainlit.md file as a string."""
    chainlit_md_path = os.path.join(root, "chainlit_lighthouse.md")
    if os.path.exists(chainlit_md_path):
        with open(chainlit_md_path, "r", encoding="utf-8") as f:
            chainlit_md = f.read()
            return chainlit_md
    else:
        return None
