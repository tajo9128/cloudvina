"""
AutoDock Vina Output Parser
Extracts docking results from Vina log files for comprehensive reporting.
"""

import re
from typing import Dict, List, Optional
from datetime import datetime


class VinaParser:
    """Parse AutoDock Vina output logs to extract docking results."""
    
    def __init__(self, log_content: str):
        """
        Initialize parser with Vina log content.
        
        Args:
            log_content: Raw text content from Vina output log
        """
        self.log_content = log_content
        self.poses: List[Dict] = []
        self.best_affinity: Optional[float] = None
        self.num_poses: int = 0
        self.energy_range_min: Optional[float] = None
        self.energy_range_max: Optional[float] = None
    
    def parse(self) -> Dict:
        """
        Parse the Vina log and extract all relevant data.
        
        Returns:
            Dict containing structured docking results
        """
        self._extract_poses()
        self._calculate_statistics()
        
        return {
            "poses": self.poses,
            "best_affinity": self.best_affinity,
            "num_poses": self.num_poses,
            "energy_range_min": self.energy_range_min,
            "energy_range_max": self.energy_range_max,
            "parsed_at": datetime.utcnow().isoformat()
        }
    
    def _extract_poses(self):
        """
        Extract binding modes/poses from the results table.
        
        Vina output format:
        -----+------------+----------+----------
        mode |   affinity | dist from | best mode
             | (kcal/mol) | rmsd l.b.| rmsd u.b.
        -----+------------+----------+----------
           1       -8.5      0.000      0.000
           2       -8.1      1.523      2.104
        """
        # Find the results table section
        table_pattern = r'mode\s+\|\s+affinity.*?\n-+\n((?:\s+\d+.*?\n)+)'
        match = re.search(table_pattern, self.log_content, re.MULTILINE | re.DOTALL)
        
        if not match:
            return
        
        table_content = match.group(1)
        
        # Parse each row
        # Format: "   1       -8.5      0.000      0.000"
        row_pattern = r'\s+(\d+)\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)'
        
        for row_match in re.finditer(row_pattern, table_content):
            mode = int(row_match.group(1))
            affinity = float(row_match.group(2))
            rmsd_lb = float(row_match.group(3))
            rmsd_ub = float(row_match.group(4))
            
            self.poses.append({
                "mode": mode,
                "affinity": affinity,
                "rmsd_lb": rmsd_lb,
                "rmsd_ub": rmsd_ub
            })
    
    def _calculate_statistics(self):
        """Calculate summary statistics from parsed poses."""
        if not self.poses:
            return
        
        affinities = [pose["affinity"] for pose in self.poses]
        
        self.num_poses = len(self.poses)
        self.best_affinity = min(affinities)  # Most negative = best
        self.energy_range_min = min(affinities)
        self.energy_range_max = max(affinities)
    
    @staticmethod
    def extract_receptor_info(log_content: str) -> Optional[str]:
        """
        Extract receptor filename from log.
        
        Returns:
            Receptor filename or None
        """
        pattern = r'Reading input.*?receptor:\s+(.+)'
        match = re.search(pattern, log_content)
        return match.group(1).strip() if match else None
    
    @staticmethod
    def extract_ligand_info(log_content: str) -> Optional[str]:
        """
        Extract ligand filename from log.
        
        Returns:
            Ligand filename or None
        """
        pattern = r'Reading input.*?ligand:\s+(.+)'
        match = re.search(pattern, log_content)
        return match.group(1).strip() if match else None


def parse_vina_log(log_content: str) -> Dict:
    """
    Convenience function to parse Vina log.
    
    Args:
        log_content: Raw text content from Vina log file
    
    Returns:
        Dict with structured docking results
    
    Example:
        >>> log = open("vina_output.log").read()
        >>> results = parse_vina_log(log)
        >>> print(results["best_affinity"])
        -8.5
    """
    parser = VinaParser(log_content)
    return parser.parse()
