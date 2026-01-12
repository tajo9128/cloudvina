import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, deque
import ast

"""
Phase 1.2: Dependency Graph Builder

Purpose: Give Agent Zero architectural understanding, not just file awareness
Agentic Impact: Enables safe refactors, blast radius reasoning, prevents breaking hidden dependencies
"""


class DependencyGraph:
    """
    Builds and maintains a directed dependency graph of repository code.

    Identifies:
    - Upstream dependencies (what a file depends on)
    - Downstream impact (what depends on a file)
    - Critical files (highly connected)
    - Circular dependencies
    """

    def __init__(self, repo_path: str):
        """
        Initialize DependencyGraph for a repository.

        Args:
            repo_path: Root directory of the repository
        """
        self.repo_path = Path(repo_path).resolve()
        self.graph: Dict[str, Set[str]] = defaultdict(set)  # file -> [depends on these]
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)  # file -> [dependents of this]
        self.circular_deps: List[List[str]] = []
        self.file_types: Dict[str, str] = {}  # file -> language
        self.external_deps: Dict[str, Set[str]] = defaultdict(set)  # file -> [external dependencies]

    def _parse_python_imports(self, file_path: Path, content: str) -> Set[str]:
        """
        Parse Python import statements using AST.

        Args:
            file_path: Path to the Python file
            content: File content

        Returns:
            Set of imported module paths
        """
        imports = set()

        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                # Import: import module
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)

                # ImportFrom: from module import name
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Handle relative imports
                        if node.level > 0:
                            # Calculate absolute path for relative import
                            file_dir = file_path.parent
                            for _ in range(node.level - 1):
                                file_dir = file_dir.parent
                            module_path = file_dir.relative_to(self.repo_path)
                            module_path = module_path / Path(node.module.replace(".", "/"))
                            imports.add(str(module_path).replace("/", "."))
                        else:
                            imports.add(node.module)

                            # Also import specific names
                            for alias in node.names:
                                if alias.name != "*":
                                    imports.add(f"{node.module}.{alias.name}")
        except (SyntaxError, UnicodeDecodeError):
            # If AST parsing fails, use regex fallback
            imports.update(self._parse_imports_regex(content, python=True))

        return imports

    def _parse_js_imports(self, content: str) -> Set[str]:
        """
        Parse JavaScript/JSX import statements using regex.

        Args:
            content: File content

        Returns:
            Set of imported module paths
        """
        return self._parse_imports_regex(content, python=False)

    def _parse_imports_regex(self, content: str, python: bool = True) -> Set[str]:
        """
        Fallback regex-based import parsing.

        Args:
            content: File content
            python: True for Python, False for JS

        Returns:
            Set of imported module paths
        """
        imports = set()

        if python:
            # Python: import X, from X import Y, from . import Y
            patterns = [
                r"^import\s+(.+?)(?:\s+as\s\w+)?\$",
                r"^from\s+(.+?)\s+import\s+(.+?)\$",
            ]

            for line in content.split("\n"):
                for pattern in patterns:
                    match = re.match(pattern, line.strip())
                    if match:
                        module = match.group(1).split(",")[0].strip()
                        imports.add(module)
        else:
            # JavaScript: import X from "Y", import {X} from "Y", require("Y")
            patterns = [
                r"import\s+.*?from\s+["']([^"']+)["']",
                r"require\(["']([^"']+)["']\)",
                r"import\(["']([^"']+)["']\)",
            ]

            for pattern in patterns:
                for match in re.finditer(pattern, content):
                    imports.add(match.group(1))

        return imports

    def _resolve_import_to_file(self, import_path: str, source_file: Path) -> Optional[str]:
        """
        Resolve an import statement to an actual file path.

        Args:
            import_path: Module path from import statement
            source_file: File containing the import

        Returns:
            Relative file path or None if external/not found
        """
        # Python: import module.submodule
        if "." in import_path:
            # Try to find as .py file
            py_path = import_path.replace(".", "/") + ".py"
            init_path = import_path.replace(".", "/") + "/__init__.py"

            for possible in [py_path, init_path]:
                if (self.repo_path / possible).exists():
                    return possible

        # JavaScript: relative imports "./module", "../module"
        if import_path.startswith("./") or import_path.startswith("../"):
            source_dir = source_file.parent
            resolved = (source_dir / import_path).resolve()

            # Try common extensions
            for ext in [".js", ".jsx", ".ts", ".tsx", "/index.js", "/index.jsx"]:
                possible = str(resolved) + ext
                if Path(possible).exists():
                    return str(Path(possible).relative_to(self.repo_path))

            # Check if it is a directory with index
            if Path(resolved).is_dir():
                for index in ["index.js", "index.jsx"]:
                    index_path = resolved / index
                    if index_path.exists():
                        return str(index_path.relative_to(self.repo_path))

        # JavaScript: absolute imports from src/
        if not import_path.startswith(".") and not import_path.startswith("/"):
            # Check common source directories
            for src_dir in ["src", "app", "lib", "components", "services"]:
                possible = self.repo_path / src_dir / import_path
                if possible.exists():
                    if possible.is_dir():
                        # Try index files
                        for index in ["index.js", "index.jsx", "index.ts"]:
                            index_path = possible / index
                            if index_path.exists():
                                return str(index_path.relative_to(self.repo_path))
                    elif possible.suffix in [".js", ".jsx", ".ts", ".tsx"]:
                        return str(possible.relative_to(self.repo_path))
                    else:
                        # Try adding extensions
                        for ext in [".js", ".jsx", ".ts", ".tsx"]:
                            if (str(possible) + ext).exists():
                                return str((str(possible) + ext).replace(str(self.repo_path) + "/", ""))

        return None  # External dependency

    def build_from_file_index(self, file_index: Dict) -> Dict:
        """
        Build dependency graph from a file index.

        Args:
            file_index: File index dict from FileIndexer

        Returns:
            Build result with statistics
        """
        files = file_index.get("files", {})
        self.file_types = {path: info["language"] for path, info in files.items()}

        stats = {
            "total_files": len(files),
            "files_with_deps": 0,
            "external_deps": set(),
            "internal_deps": 0
        }

        for file_path, file_info in files.items():
            full_path = self.repo_path / file_path
            language = file_info["language"]

            # Read file content
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except (IOError, UnicodeDecodeError):
                continue

            # Parse imports based on language
            imports = set()
            if language == "python":
                imports = self._parse_python_imports(full_path, content)
            elif language in ["javascript", "jsx", "typescript", "tsx"]:
                imports = self._parse_js_imports(content)

            if imports:
                stats["files_with_deps"] += 1

                # Resolve each import to a file path
                for imp in imports:
                    resolved = self._resolve_import_to_file(imp, full_path)

                    if resolved:
                        # Internal dependency
                        if resolved in self.file_types:
                            self.graph[file_path].add(resolved)
                            self.reverse_graph[resolved].add(file_path)
                            stats["internal_deps"] += 1
                    else:
                        # External dependency
                        self.external_deps[file_path].add(imp)
                        stats["external_deps"].add(imp)

        # Detect circular dependencies
        self._detect_circular_deps()

        stats["total_deps"] = sum(len(deps) for deps in self.graph.values())
        stats["external_deps_count"] = len(stats["external_deps"])
        stats["circular_deps_count"] = len(self.circular_deps)

        return stats

    def _detect_circular_deps(self) -> None:
        """
        Detect circular dependencies using DFS.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in self.graph}
        cycles = []
        path = []

        def dfs(node):
            color[node] = GRAY
            path.append(node)

            for neighbor in self.graph[node]:
                if color[neighbor] == GRAY:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                elif color[neighbor] == WHITE:
                    dfs(neighbor)

            path.pop()
            color[node] = BLACK

        for node in self.graph:
            if color[node] == WHITE:
                dfs(node)

        self.circular_deps = cycles

    def get_upstream_dependencies(self, file_path: str) -> Set[str]:
        """
        Get all upstream dependencies (what this file depends on).

        Args:
            file_path: File to query

        Returns:
            Set of dependency file paths
        """
        return self.graph.get(file_path, set())

    def get_downstream_impact(self, file_path: str) -> Set[str]:
        """
        Get all downstream impact (what depends on this file).

        Args:
            file_path: File to query

        Returns:
            Set of dependent file paths
        """
        return self.reverse_graph.get(file_path, set())

    def get_blast_radius(self, file_path: str) -> Dict[str, int]:
        """
        Calculate blast radius - impact of changing a file.

        Args:
            file_path: File to analyze

        Returns:
            Dict with impact metrics
        """
        # Direct impact (first level dependencies)
        direct = self.reverse_graph.get(file_path, set())

        # Transitive impact (all levels)
        transitive = set()
        queue = list(direct)
        visited = set(queue)

        while queue:
            node = queue.pop(0)
            for dep in self.reverse_graph.get(node, set()):
                if dep not in visited:
                    visited.add(dep)
                    queue.append(dep)

        return {
            "direct_impact_count": len(direct),
            "transitive_impact_count": len(transitive),
            "direct_impact": list(direct),
            "critical_score": len(transitive) + (len(direct) * 2)  # Direct impact weighted more
        }

    def get_critical_files(self, top_n: int = 10) -> List[Dict[str, any]]:
        """
        Get the most critical files (highly connected).

        Args:
            top_n: Number of top files to return

        Returns:
            List of critical files with metrics
        """
        criticality = []

        for file_path in self.graph:
            # Count both upstream and downstream connections
            upstream = len(self.graph[file_path])
            downstream = len(self.reverse_graph.get(file_path, set()))
            total = upstream + downstream

            criticality.append({
                "file": file_path,
                "upstream_deps": upstream,
                "downstream_impact": downstream,
                "total_connections": total,
                "language": self.file_types.get(file_path, "unknown")
            })

        # Sort by total connections descending
        criticality.sort(key=lambda x: x["total_connections"], reverse=True)

        return criticality[:top_n]

    def save_graph(self, output_path: str) -> None:
        """
        Save dependency graph to JSON file.

        Args:
            output_path: Path to save graph JSON
        """
        output = {
            "repo_path": str(self.repo_path),
            "graph": {k: list(v) for k, v in self.graph.items()},
            "reverse_graph": {k: list(v) for k, v in self.reverse_graph.items()},
            "circular_dependencies": self.circular_deps,
            "external_dependencies": {k: list(v) for k, v in self.external_deps.items()},
            "file_types": self.file_types
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

    def load_graph(self, input_path: str) -> None:
        """
        Load dependency graph from JSON file.

        Args:
            input_path: Path to load graph JSON from
        """
        with open(input_path, "r") as f:
            data = json.load(f)

        self.graph = defaultdict(set, {k: set(v) for k, v in data["graph"].items()})
        self.reverse_graph = defaultdict(set, {k: set(v) for k, v in data["reverse_graph"].items()})
        self.circular_deps = data.get("circular_dependencies", [])
        self.external_deps = defaultdict(set, {k: set(v) for k, v in data.get("external_dependencies", {}).items()})
        self.file_types = data.get("file_types", {})
