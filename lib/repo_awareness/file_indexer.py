import os
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
import fnmatch

"""
Phase 1.1: File System Indexer

Purpose: Turn repository into a known object with incremental change detection
Agentic Impact: Enables incremental reasoning, prevents re-reading unchanged files, supports "what changed?" queries
"""


class FileIndexer:
    """
    Indexes repository files with metadata and change detection.

    Stores:
    - file paths (relative to repo root)
    - file types (language detection)
    - size (bytes)
    - last modified timestamp
    - content hash (SHA256)
    """

    # Language mappings based on file extensions
    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "jsx",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".txt": "text",
        ".sql": "sql",
        ".html": "html",
        ".css": "css",
        ".sh": "shell",
        ".xml": "xml",
        ".csv": "csv",
    }

    # Patterns to exclude from indexing
    EXCLUDE_PATTERNS = [
        "__pycache__",
        ".git",
        "node_modules",
        ".venv",
        "venv",
        ".env",
        "*.pyc",
        ".DS_Store",
        ".pytest_cache",
        "dist",
        "build",
    ]

    def __init__(self, repo_path: str):
        """
        Initialize FileIndexer for a repository.

        Args:
            repo_path: Root directory of the repository
        """
        self.repo_path = Path(repo_path).resolve()
        self.index: Dict[str, Dict] = {}
        self.last_scan_time: Optional[datetime] = None

    def _should_exclude(self, path: Path) -> bool:
        """
        Check if a path should be excluded from indexing.

        Args:
            path: Path to check

        Returns:
            True if path should be excluded
        """
        relative_path = str(path.relative_to(self.repo_path))

        for pattern in self.EXCLUDE_PATTERNS:
            if fnmatch.fnmatch(relative_path, pattern) or                fnmatch.fnmatch(relative_path, f"**/{pattern}") or                fnmatch.fnmatch(relative_path, f"**/{pattern}/**"):
                return True
        return False

    def _detect_language(self, path: Path) -> str:
        """
        Detect programming language from file extension.

        Args:
            path: File path

        Returns:
            Language name or "unknown"
        """
        suffix = path.suffix.lower()
        return self.LANGUAGE_MAP.get(suffix, "unknown")

    def _compute_hash(self, path: Path) -> str:
        """
        Compute SHA256 hash of file content.

        Args:
            path: File path

        Returns:
            SHA256 hash string
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
        except (IOError, PermissionError):
            return ""
        return sha256_hash.hexdigest()

    def scan(self, incremental: bool = False) -> Dict:
        """
        Scan repository and build file index.

        Args:
            incremental: If True, only scan changed files

        Returns:
            Index metadata with scan results
        """
        scan_start = datetime.now()
        new_index = {}
        changed_files = []
        added_files = []
        removed_files = list(self.index.keys())  # Assume all removed, then subtract found

        # Walk through repository
        for root, dirs, files in os.walk(self.repo_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not self._should_exclude(Path(root) / d)]

            for file in files:
                file_path = Path(root) / file

                # Skip excluded files
                if self._should_exclude(file_path):
                    continue

                relative_path = str(file_path.relative_to(self.repo_path))

                # Remove from removed_files list (it still exists)
                if relative_path in removed_files:
                    removed_files.remove(relative_path)

                try:
                    # Get file metadata
                    stat = file_path.stat()
                    file_hash = self._compute_hash(file_path)

                    file_info = {
                        "path": relative_path,
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "modified_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "hash": file_hash,
                        "language": self._detect_language(file_path),
                        "type": "file"
                    }

                    # Check if changed (incremental mode)
                    if incremental and relative_path in self.index:
                        old_info = self.index[relative_path]
                        if old_info["hash"] != file_hash:
                            changed_files.append(relative_path)
                    elif incremental:
                        # New file
                        added_files.append(relative_path)
                    else:
                        # First scan (non-incremental), treat as added
                        added_files.append(relative_path)

                    new_index[relative_path] = file_info

                except (IOError, PermissionError) as e:
                    # Skip files we cannot read
                    print(f"Warning: Cannot read {file_path}: {e}")
                    continue

        # Update internal index
        self.index = new_index
        self.last_scan_time = scan_start

        # Build scan result
        result = {
            "scan_time": scan_start.isoformat(),
            "scan_duration_seconds": (datetime.now() - scan_start).total_seconds(),
            "total_files": len(self.index),
            "incremental": incremental,
            "changed_files": changed_files if incremental else [],
            "added_files": added_files if incremental else [],
            "removed_files": removed_files if incremental else [],
            "files": self.index
        }

        return result

    def get_file_info(self, relative_path: str) -> Optional[Dict]:
        """
        Get metadata for a specific file.

        Args:
            relative_path: Relative path to file

        Returns:
            File info dict or None if not found
        """
        return self.index.get(relative_path)

    def get_files_by_language(self, language: str) -> List[str]:
        """
        Get all files of a specific language.

        Args:
            language: Programming language name

        Returns:
            List of relative file paths
        """
        return [path for path, info in self.index.items() if info["language"] == language]

    def get_changed_files(self, since_hash: Dict) -> List[str]:
        """
        Compare current index with a previous index to find changed files.

        Args:
            since_hash: Previous file index dict

        Returns:
            List of changed file paths
        """
        changed = []

        # Check for modified files
        for path, info in self.index.items():
            if path in since_hash:
                if since_hash[path]["hash"] != info["hash"]:
                    changed.append(path)
            else:
                # New file
                changed.append(path)

        # Check for removed files
        for path in since_hash:
            if path not in self.index:
                changed.append(f"REMOVED:{path}")

        return changed

    def save_index(self, output_path: str) -> None:
        """
        Save index to JSON file.

        Args:
            output_path: Path to save index JSON
        """
        output = {
            "repo_path": str(self.repo_path),
            "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "index": self.index
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

    def load_index(self, input_path: str) -> None:
        """
        Load index from JSON file.

        Args:
            input_path: Path to load index JSON from
        """
        with open(input_path, "r") as f:
            data = json.load(f)

        self.index = data["index"]
        self.last_scan_time = datetime.fromisoformat(data["last_scan_time"]) if data["last_scan_time"] else None

    def get_stats(self) -> Dict:
        """
        Get statistics about the indexed repository.

        Returns:
            Statistics dict
        """
        languages = {}
        for info in self.index.values():
            lang = info["language"]
            languages[lang] = languages.get(lang, 0) + 1

        total_size = sum(info["size"] for info in self.index.values())

        return {
            "total_files": len(self.index),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "languages": languages,
            "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None
        }
