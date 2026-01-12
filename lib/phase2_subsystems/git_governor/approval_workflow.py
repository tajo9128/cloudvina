"""
Git Governor - ApprovalWorkflow implementation.
Manages approval requests for risky operations.
"""
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
from .interfaces import (
    IApprovalWorkflow, ApprovalResponse, ApprovalStatus, RiskLevel, CommitValidation
)
from .schemas import validate_schema, add_checksum, GIT_GOVERNOR_SCHEMAS


@dataclass
class ApprovalConfig:
    """Configuration for ApprovalWorkflow."""
    approval_timeout_minutes: int = 30
    auto_approve_levels: List[RiskLevel] = None
    auto_reject_on_timeout: bool = True
    
    def __post_init__(self):
        if self.auto_approve_levels is None:
            self.auto_approve_levels = [RiskLevel.LOW]


class ApprovalRequest:
    """Internal representation of an approval request."""
    
    def __init__(self, request_id: str, commit_validation: CommitValidation,
                 requested_by: str, created_at: datetime):
        self.request_id = request_id
        self.commit_validation = commit_validation
        self.requested_by = requested_by
        self.created_at = created_at
        self.expires_at = created_at + timedelta(minutes=30)
        self.status = ApprovalStatus.PENDING
        self.approved_by: Optional[str] = None
        self.approval_comment: Optional[str] = None
        self.approved_at: Optional[datetime] = None
    
    def approve(self, approved_by: str, comment: Optional[str] = None):
        """Approve this request."""
        self.status = ApprovalStatus.APPROVED
        self.approved_by = approved_by
        self.approval_comment = comment
        self.approved_at = datetime.now()
    
    def reject(self, rejected_by: str, comment: Optional[str] = None):
        """Reject this request."""
        self.status = ApprovalStatus.REJECTED
        self.approved_by = rejected_by
        self.approval_comment = comment or "Request rejected"
        self.approved_at = datetime.now()
    
    def is_expired(self) -> bool:
        """Check if this request has expired."""
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "request_id": self.request_id,
            "commit_validation": {
                "is_valid": self.commit_validation.is_valid,
                "can_proceed": self.commit_validation.can_proceed,
                "risk_level": self.commit_validation.risk_assessment.overall_risk.value,
                "errors": self.commit_validation.errors,
                "warnings": self.commit_validation.warnings
            },
            "requested_by": self.requested_by,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "status": self.status.value,
            "approved_by": self.approved_by,
            "approval_comment": self.approval_comment,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None
        }


class ApprovalWorkflow(IApprovalWorkflow):
    """Implements approval workflow management."""
    
    def __init__(self, config: Optional[ApprovalConfig] = None,
                 storage_path: Optional[str] = None):
        self.config = config or ApprovalConfig()
        self.storage_path = Path(storage_path) if storage_path else Path("/a0/tmp/approvals")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory request cache
        self._requests: Dict[str, ApprovalRequest] = {}
        self._load_requests()
    
    def _load_requests(self):
        """Load pending requests from storage."""
        if not self.storage_path.exists():
            return
        
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Skip expired requests
                expires_at = datetime.fromisoformat(data["expires_at"])
                if datetime.now() > expires_at:
                    file_path.unlink()
                    continue
                
                # Create ApprovalRequest (simplified)
                self._requests[data["request_id"]] = ApprovalRequest(
                    request_id=data["request_id"],
                    commit_validation=None,  # Would need full deserialization
                    requested_by=data["requested_by"],
                    created_at=datetime.fromisoformat(data["created_at"])
                )
            except Exception:
                pass
    
    def _save_request(self, request: ApprovalRequest):
        """Save request to storage."""
        file_path = self.storage_path / f"{request.request_id}.json"
        with open(file_path, 'w') as f:
            json.dump(request.to_dict(), f, indent=2)
    
    def request_approval(self, commit_validation: CommitValidation) -> ApprovalResponse:
        """Request approval for a commit."""
        request_id = str(uuid.uuid4())
        requested_by = "agent_zero"  # Could be configurable
        
        # Create request
        request = ApprovalRequest(
            request_id=request_id,
            commit_validation=commit_validation,
            requested_by=requested_by,
            created_at=datetime.now()
        )
        
        # Store request
        self._requests[request_id] = request
        self._save_request(request)
        
        # Determine if auto-approval applies
        risk_level = commit_validation.risk_assessment.overall_risk
        if risk_level in self.config.auto_approve_levels:
            request.approve("auto_approved", "Low risk operation auto-approved")
            self._save_request(request)
        
        # Build response
        response = ApprovalResponse(
            request_id=request_id,
            status=request.status,
            approved_by=request.approved_by,
            approval_comment=request.approval_comment,
            timestamp=datetime.now().isoformat()
        )
        
        return response
    
    def check_approval_status(self, request_id: str) -> ApprovalResponse:
        """Check status of approval request."""
        if request_id not in self._requests:
            return ApprovalResponse(
                request_id=request_id,
                status=ApprovalStatus.REJECTED,
                approved_by=None,
                approval_comment="Request not found",
                timestamp=datetime.now().isoformat()
            )
        
        request = self._requests[request_id]
        
        # Check for timeout
        if request.status == ApprovalStatus.PENDING and request.is_expired():
            if self.config.auto_reject_on_timeout:
                request.reject("system", "Request timed out")
                self._save_request(request)
        
        return ApprovalResponse(
            request_id=request_id,
            status=request.status,
            approved_by=request.approved_by,
            approval_comment=request.approval_comment,
            timestamp=datetime.now().isoformat()
        )
    
    def is_approval_required(self, risk_level: RiskLevel) -> bool:
        """Determine if approval is required for risk level."""
        return risk_level not in self.config.auto_approve_levels
    
    def approve_request(self, request_id: str, approved_by: str,
                        comment: Optional[str] = None) -> ApprovalResponse:
        """Manually approve a request."""
        if request_id not in self._requests:
            return ApprovalResponse(
                request_id=request_id,
                status=ApprovalStatus.REJECTED,
                approved_by=None,
                approval_comment="Request not found",
                timestamp=datetime.now().isoformat()
            )
        
        request = self._requests[request_id]
        request.approve(approved_by, comment)
        self._save_request(request)
        
        return ApprovalResponse(
            request_id=request_id,
            status=request.status,
            approved_by=request.approved_by,
            approval_comment=request.approval_comment,
            timestamp=datetime.now().isoformat()
        )
    
    def reject_request(self, request_id: str, rejected_by: str,
                       comment: Optional[str] = None) -> ApprovalResponse:
        """Manually reject a request."""
        if request_id not in self._requests:
            return ApprovalResponse(
                request_id=request_id,
                status=ApprovalStatus.REJECTED,
                approved_by=None,
                approval_comment="Request not found",
                timestamp=datetime.now().isoformat()
            )
        
        request = self._requests[request_id]
        request.reject(rejected_by, comment)
        self._save_request(request)
        
        return ApprovalResponse(
            request_id=request_id,
            status=request.status,
            approved_by=request.approved_by,
            approval_comment=request.approval_comment,
            timestamp=datetime.now().isoformat()
        )
    
    def cleanup_expired(self):
        """Remove expired requests."""
        expired_ids = [
            rid for rid, req in self._requests.items()
            if req.is_expired()
        ]
        
        for rid in expired_ids:
            file_path = self.storage_path / f"{rid}.json"
            if file_path.exists():
                file_path.unlink()
            del self._requests[rid]
        
        return len(expired_ids)
