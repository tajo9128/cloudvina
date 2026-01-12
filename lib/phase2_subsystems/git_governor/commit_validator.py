"""
Git Governor - CommitValidator implementation.
Handles diff analysis, secret detection, and risk assessment.
"""
import re
import subprocess
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from .interfaces import (
    ICommitValidator, BranchValidation, SecretCheck, RiskAssessment,
    CommitValidation, CommitRequest, FileChange, RiskLevel, CommitType
)
from .schemas import validate_schema, add_checksum, GIT_GOVERNOR_SCHEMAS


# Secret detection patterns (common but not exhaustive)
SECRET_PATTERNS = [
    (r'password\s*[=:]\s*["\']?[\w\-]+', 'password'),
    (r'api[_-]?key\s*[=:]\s*["\']?[\w\-]+', 'api_key'),
    (r'secret\s*[=:]\s*["\']?[\w\-]+', 'secret'),
    (r'token\s*[=:]\s*["\']?[\w\-]+', 'token'),
    (r'private[_-]?key\s*[=:]', 'private_key'),
    (r'aws[_-]?(access[_-]?key|secret)', 'aws_credentials'),
    (r'BEGIN\s+(RSA\s+)?PRIVATE\s+KEY', 'private_key_block'),
    (r'sk-[a-zA-Z0-9]{48}', 'openai_api_key'),
    (r'xox[baprs]-[0-9]{12}-[0-9]{12}-[0-9]{12}-[a-z0-9]{32}', 'slack_token'),
]

# Protected branches (default)
PROTECTED_BRANCHES = ['main', 'master', 'production', 'prod', 'staging', 'release']

# High-risk file patterns
HIGH_RISK_PATTERNS = [
    r'\.env$|\.env\.',
    r'config\.prod',
    r'secrets',
    r'credentials',
    r'docker-compose\.prod',
    r'k8s.*prod',
]


@dataclass
class CommitValidatorConfig:
    """Configuration for CommitValidator."""
    protected_branches: List[str] = None
    secret_patterns: List[tuple] = None
    high_risk_patterns: List[str] = None
    approval_required_risk_levels: List[RiskLevel] = None
    
    def __post_init__(self):
        if self.protected_branches is None:
            self.protected_branches = PROTECTED_BRANCHES
        if self.secret_patterns is None:
            self.secret_patterns = SECRET_PATTERNS
        if self.high_risk_patterns is None:
            self.high_risk_patterns = HIGH_RISK_PATTERNS
        if self.approval_required_risk_levels is None:
            self.approval_required_risk_levels = [RiskLevel.HIGH, RiskLevel.CRITICAL]


class CommitValidator(ICommitValidator):
    """Implements commit validation with security checks."""
    
    def __init__(self, config: Optional[CommitValidatorConfig] = None,
                 repo_path: Optional[str] = None):
        self.config = config or CommitValidatorConfig()
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        
        # Compile regex patterns
        self.secret_regexes = [(re.compile(pattern, re.IGNORECASE), name)
                               for pattern, name in self.config.secret_patterns]
        self.risk_regexes = [re.compile(pattern, re.IGNORECASE)
                             for pattern in self.config.high_risk_patterns]
    
    def validate_branch(self, branch: str) -> BranchValidation:
        """Validate that commits to this branch are allowed."""
        is_valid = True
        is_protected = branch.lower() in [b.lower() for b in self.config.protected_branches]
        error_message = None
        can_commit = not is_protected
        
        if is_protected:
            error_message = f"Branch '{branch}' is protected. Use feature branches instead."
        
        return BranchValidation(
            is_valid=is_valid,
            branch_name=branch,
            is_protected=is_protected,
            error_message=error_message,
            can_commit=can_commit
        )
    
    def check_secrets(self, diff: str) -> SecretCheck:
        """Scan for potential secrets in diff."""
        secrets_found = []
        
        for pattern, name in self.secret_regexes:
            matches = pattern.finditer(diff)
            for match in matches:
                # Mask the actual secret
                matched_text = match.group()
                start = max(0, match.start() - 20)
                end = min(len(diff), match.end() + 20)
                context = diff[start:end].replace('\n', ' ')
                
                secrets_found.append({
                    'type': name,
                    'context': context,
                    'line_position': match.start(),
                    'severity': 'high' if name in ['password', 'private_key', 'api_key'] else 'medium'
                })
        
        has_secrets = len(secrets_found) > 0
        risk_level = RiskLevel.CRITICAL if has_secrets else RiskLevel.LOW
        
        return SecretCheck(
            has_secrets=has_secrets,
            secrets_found=secrets_found,
            risk_level=risk_level
        )
    
    def _get_file_changes(self, files: List[str]) -> List[FileChange]:
        """Get detailed file changes using git diff."""
        changes = []
        
        for file_path in files:
            try:
                # Get diff statistics
                result = subprocess.run(
                    ['git', 'diff', '--numstat', 'HEAD', '--', file_path],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    parts = result.stdout.strip().split()
                    lines_added = int(parts[0]) if parts[0] != '-' else 0
                    lines_removed = int(parts[1]) if parts[1] != '-' else 0
                    
                    # Get content preview
                    preview_result = subprocess.run(
                        ['git', 'diff', '-U5', 'HEAD', '--', file_path],
                        cwd=self.repo_path,
                        capture_output=True,
                        text=True
                    )
                    
                    preview = preview_result.stdout[:500] if preview_result.returncode == 0 else ""
                    
                    changes.append(FileChange(
                        path=file_path,
                        change_type='modified',
                        lines_added=lines_added,
                        lines_removed=lines_removed,
                        content_preview=preview
                    ))
                else:
                    # New file or untracked
                    changes.append(FileChange(
                        path=file_path,
                        change_type='added',
                        lines_added=0,
                        lines_removed=0,
                        content_preview="new file"
                    ))
            except Exception as e:
                changes.append(FileChange(
                    path=file_path,
                    change_type='unknown',
                    lines_added=0,
                    lines_removed=0,
                    content_preview=f"error: {e}"
                ))
        
        return changes
    
    def analyze_risk(self, changes: List[FileChange]) -> RiskAssessment:
        """Analyze risk level of proposed changes."""
        risk_factors = []
        total_lines = sum(c.lines_added + c.lines_removed for c in changes)
        
        # Check file patterns
        high_risk_files = []
        for change in changes:
            for pattern in self.risk_regexes:
                if pattern.search(change.path):
                    high_risk_files.append(change.path)
        
        if high_risk_files:
            risk_factors.append(f"Modifying high-risk files: {', '.join(high_risk_files)}")
        
        # Check deletion operations
        deleted_files = [c.path for c in changes if c.change_type == 'deleted']
        if deleted_files:
            risk_factors.append(f"Deleting files: {', '.join(deleted_files)}")
        
        # Check large changes
        if total_lines > 500:
            risk_factors.append(f"Large change: {total_lines} lines modified")
        
        # Determine overall risk
        overall_risk = RiskLevel.LOW
        requires_approval = False
        
        if high_risk_files or deleted_files:
            overall_risk = RiskLevel.CRITICAL
            requires_approval = True
        elif total_lines > 200:
            overall_risk = RiskLevel.HIGH
            requires_approval = True
        elif total_lines > 50:
            overall_risk = RiskLevel.MEDIUM
        
        recommendations = []
        if overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append("Create a feature branch for this change")
            recommendations.append("Request peer review before committing")
            if high_risk_files:
                recommendations.append("Verify no sensitive data in configuration files")
        
        confidence = 0.85 if overall_risk == RiskLevel.LOW else 0.75
        
        return RiskAssessment(
            overall_risk=overall_risk,
            risk_factors=risk_factors,
            requires_approval=requires_approval,
            recommendations=recommendations,
            confidence=confidence
        )
    
    def generate_semantic_message(self, request: CommitRequest) -> str:
        """Generate conventional commit message."""
        parts = [request.commit_type.value]
        
        if request.scope:
            parts.append(f"({request.scope})")
        
        parts.append(f": {request.change_description}")
        
        message = "".join(parts)
        
        if request.body:
            message += f"\n\n{request.body}"
        
        if request.footer:
            message += f"\n\n{request.footer}"
        
        return message
    
    def validate_commit(self, request: CommitRequest, changes: List[FileChange]) -> CommitValidation:
        """Perform complete validation of commit request."""
        errors = []
        warnings = []
        
        # Get current branch
        try:
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            current_branch = result.stdout.strip()
        except:
            current_branch = "unknown"
            errors.append("Could not determine current branch")
        
        # Validate branch
        branch_valid = self.validate_branch(current_branch)
        
        # Analyze risk
        risk_assessment = self.analyze_risk(changes)
        
        # Get diff for secret check
        try:
            result = subprocess.run(
                ['git', 'diff', '--cached'] + request.files,
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            diff = result.stdout
        except:
            diff = ""
        
        # Check secrets
        secret_check = self.check_secrets(diff)
        
        # Collect errors
        if not branch_valid.is_valid:
            errors.append(branch_valid.error_message)
        
        if secret_check.has_secrets:
            errors.append(f"Secrets detected in diff: {len(secret_check.secrets_found)} found")
        
        # Collect warnings
        if secret_check.risk_level == RiskLevel.MEDIUM:
            warnings.append("Medium risk patterns detected - review carefully")
        
        if risk_assessment.risk_factors:
            warnings.extend([f"Risk: {rf}" for rf in risk_assessment.risk_factors[:3]])
        
        is_valid = len(errors) == 0
        can_proceed = is_valid and not secret_check.has_secrets
        
        return CommitValidation(
            is_valid=is_valid,
            branch_valid=branch_valid,
            secret_check=secret_check,
            risk_assessment=risk_assessment,
            can_proceed=can_proceed,
            errors=errors,
            warnings=warnings
        )
