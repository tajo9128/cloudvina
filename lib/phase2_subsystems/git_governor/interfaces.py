"""
Git Governor - Interfaces for governed commit operations.
Implements strict governance for all Git operations.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(Enum):
    """Risk assessment levels for commits."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CommitType(Enum):
    """Conventional commit types."""
    FEAT = "feat"
    FIX = "fix"
    DOCS = "docs"
    STYLE = "style"
    REFACTOR = "refactor"
    PERF = "perf"
    TEST = "test"
    CHORE = "chore"


class ApprovalStatus(Enum):
    """Approval workflow status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"


@dataclass
class FileChange:
    """Represents a file change in a commit."""
    path: str
    change_type: str  # added, modified, deleted, renamed
    lines_added: int
    lines_removed: int
    content_preview: str


@dataclass
class SecretCheck:
    """Result of secret detection scan."""
    has_secrets: bool
    secrets_found: List[Dict[str, str]] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW


@dataclass
class RiskAssessment:
    """Risk assessment for a commit."""
    overall_risk: RiskLevel
    risk_factors: List[str] = field(default_factory=list)
    requires_approval: bool = False
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.9


@dataclass
class BranchValidation:
    """Branch validation result."""
    is_valid: bool
    branch_name: str
    is_protected: bool
    error_message: Optional[str] = None
    can_commit: bool = False


@dataclass
class CommitValidation:
    """Complete validation result for a commit."""
    is_valid: bool
    branch_valid: BranchValidation
    secret_check: SecretCheck
    risk_assessment: RiskAssessment
    can_proceed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class CommitRequest:
    """Request for creating a commit."""
    files: List[str]
    change_description: str
    commit_type: CommitType
    scope: Optional[str] = None
    body: Optional[str] = None
    footer: Optional[str] = None
    force: bool = False


@dataclass
class Commit:
    """Result of a successful commit."""
    commit_hash: str
    branch: str
    message: str
    timestamp: str
    author: str
    files: List[str] = field(default_factory=list)
    lines_changed: Dict[str, int] = field(default_factory=dict)


@dataclass
class ApprovalResponse:
    """Response to an approval request."""
    request_id: str
    status: ApprovalStatus
    approved_by: Optional[str] = None
    approval_comment: Optional[str] = None
    timestamp: str = ""


@dataclass
class PushResult:
    """Result of a push operation."""
    success: bool
    remote: str
    branch: str
    commit_hash: str = ""
    ci_status: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class Checkpoint:
    """A safe rollback checkpoint."""
    checkpoint_id: str
    label: str
    commit_hash: str
    branch: str
    timestamp: str
    files_snapshot: List[str] = field(default_factory=list)


@dataclass
class GitStatus:
    """Current Git repository status."""
    branch: str
    is_clean: bool
    modified_files: List[str] = field(default_factory=list)
    staged_files: List[str] = field(default_factory=list)
    untracked_files: List[str] = field(default_factory=list)
    ahead_commits: int = 0
    behind_commits: int = 0
    last_commit: str = ""


class ICommitValidator(ABC):
    """Commit validation interface."""
    
    @abstractmethod
    def validate_branch(self, branch: str) -> BranchValidation:
        """Validate that commits to this branch are allowed."""
        pass
    
    @abstractmethod
    def check_secrets(self, diff: str) -> SecretCheck:
        """Scan for potential secrets in diff."""
        pass
    
    @abstractmethod
    def analyze_risk(self, changes: List[FileChange]) -> RiskAssessment:
        """Analyze risk level of proposed changes."""
        pass
    
    @abstractmethod
    def validate_commit(self, request: CommitRequest, changes: List[FileChange]) -> CommitValidation:
        """Perform complete validation of commit request."""
        pass
    
    @abstractmethod
    def generate_semantic_message(self, request: CommitRequest) -> str:
        """Generate conventional commit message."""
        pass


class IApprovalWorkflow(ABC):
    """Approval workflow interface."""
    
    @abstractmethod
    def request_approval(self, commit: CommitValidation) -> ApprovalResponse:
        """Request approval for a commit."""
        pass
    
    @abstractmethod
    def check_approval_status(self, request_id: str) -> ApprovalResponse:
        """Check status of approval request."""
        pass
    
    @abstractmethod
    def is_approval_required(self, risk_level: RiskLevel) -> bool:
        """Determine if approval is required for risk level."""
        pass


class IGitGovernor(ABC):
    """Git Governor main interface."""
    
    @abstractmethod
    def get_git_status(self) -> GitStatus:
        """Get current repository status."""
        pass
    
    @abstractmethod
    def validate_and_request_approval(self, request: CommitRequest) -> tuple[CommitValidation, Optional[ApprovalResponse]]:
        """Validate commit and request approval if needed."""
        pass
    
    @abstractmethod
    def create_commit(self, message: str, files: Optional[List[str]] = None) -> Commit:
        """Create a commit with validation."""
        pass
    
    @abstractmethod
    def push_commit(self, commit: Commit) -> PushResult:
        """Push commit with CI check."""
        pass
    
    @abstractmethod
    def create_checkpoint(self, label: str) -> Checkpoint:
        """Create a safe checkpoint before risky operations."""
        pass
    
    @abstractmethod
    def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """Rollback to a safe checkpoint."""
        pass
    
    @abstractmethod
    def get_available_checkpoints(self) -> List[Checkpoint]:
        """List all available checkpoints."""
        pass
