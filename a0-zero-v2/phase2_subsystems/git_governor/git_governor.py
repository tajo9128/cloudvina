"""
Git Governor - Main GitGovernor implementation.
Coordinates all Git operations with validation, approval, and safety checks.
"""
import subprocess
import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from .interfaces import (
    IGitGovernor, GitStatus, Commit, PushResult, Checkpoint,
    CommitRequest, FileChange, CommitValidation, ApprovalResponse,
    CommitType, RiskLevel
)
from .commit_validator import CommitValidator, CommitValidatorConfig
from .approval_workflow import ApprovalWorkflow, ApprovalConfig
from .schemas import add_checksum, validate_schema


class GitGovernor(IGitGovernor):
    """Main Git Governor for safe autonomous Git operations."""
    
    def __init__(self, repo_path: Optional[str] = None,
                 validator_config: Optional[CommitValidatorConfig] = None,
                 approval_config: Optional[ApprovalConfig] = None):
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        
        # Initialize subsystems
        self.validator = CommitValidator(validator_config, self.repo_path)
        self.approval_workflow = ApprovalWorkflow(approval_config)
        
        # Checkpoints storage
        self.checkpoints_path = self.repo_path / ".git" / "a0_checkpoints"
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)
        
        # Validate repo
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"Not a Git repository: {self.repo_path}")
    
    def get_git_status(self) -> GitStatus:
        """Get current repository status."""
        try:
            # Get branch
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            branch = result.stdout.strip()
            
            # Get status
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            status_lines = result.stdout.strip().split('\n')
            
            modified_files = []
            staged_files = []
            untracked_files = []
            
            for line in status_lines:
                if not line:
                    continue
                status = line[:2]
                path = line[3:]
                
                if '??' in status:
                    untracked_files.append(path)
                elif status[0] in ['M', 'A', 'D', 'R']:
                    staged_files.append(path)
                elif status[1] in ['M', 'A', 'D', 'R']:
                    modified_files.append(path)
            
            # Get ahead/behind
            result = subprocess.run(
                ['git', 'rev-list', '--count', '--left-right', '@{u}...HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                behind, ahead = result.stdout.strip().split()
                ahead_commits = int(ahead)
                behind_commits = int(behind)
            else:
                ahead_commits = 0
                behind_commits = 0
            
            # Get last commit
            result = subprocess.run(
                ['git', 'log', '-1', '--format=%h %s'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            last_commit = result.stdout.strip() if result.returncode == 0 else ""
            
            is_clean = len(modified_files) == 0 and len(staged_files) == 0
            
            return GitStatus(
                branch=branch,
                is_clean=is_clean,
                modified_files=modified_files,
                staged_files=staged_files,
                untracked_files=untracked_files,
                ahead_commits=ahead_commits,
                behind_commits=behind_commits,
                last_commit=last_commit
            )
            
        except Exception as e:
            return GitStatus(
                branch="unknown",
                is_clean=False,
                modified_files=[],
                staged_files=[],
                untracked_files=[],
                ahead_commits=0,
                behind_commits=0,
                last_commit=f"error: {e}"
            )
    
    def validate_and_request_approval(self, request: CommitRequest) -> tuple[CommitValidation, Optional[ApprovalResponse]]:
        """Validate commit and request approval if needed."""
        # Get file changes
        changes = self.validator._get_file_changes(request.files)
        
        # Validate commit
        validation = self.validator.validate_commit(request, changes)
        
        # Request approval if needed
        approval_response = None
        if validation.can_proceed:
            risk_level = validation.risk_assessment.overall_risk
            if self.approval_workflow.is_approval_required(risk_level):
                approval_response = self.approval_workflow.request_approval(validation)
        
        return validation, approval_response
    
    def create_commit(self, message: str, files: Optional[List[str]] = None) -> Commit:
        """Create a commit with validation."""
        # Stage files if provided
        if files:
            subprocess.run(['git', 'add'] + files, cwd=self.repo_path, check=True)
        
        # Create commit
        result = subprocess.run(
            ['git', 'commit', '-m', message],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create commit: {result.stderr}")
        
        # Get commit details
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%H|%s|%an|%ae|%ct'],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        
        commit_hash, subject, author, email, timestamp = result.stdout.strip().split('|')
        
        return Commit(
            commit_hash=commit_hash,
            branch=self.get_git_status().branch,
            message=message,
            timestamp=datetime.fromtimestamp(int(timestamp)).isoformat(),
            author=f"{author} <{email}>",
            files=files or [],
            lines_changed={}
        )
    
    def push_commit(self, commit: Commit) -> PushResult:
        """Push commit with CI check."""
        # Get remote
        result = subprocess.run(
            ['git', 'remote', '-v'],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        
        remote = "origin"
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if '(push)' in line:
                    remote = line.split()[0]
                    break
        
        # Push
        result = subprocess.run(
            ['git', 'push', remote, commit.branch],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return PushResult(
                success=False,
                remote=remote,
                branch=commit.branch,
                commit_hash=commit.commit_hash,
                ci_status=None,
                error_message=result.stderr
            )
        
        # Note: In a real implementation, you would check CI status here
        # This is a placeholder for CI integration
        
        return PushResult(
            success=True,
            remote=remote,
            branch=commit.branch,
            commit_hash=commit.commit_hash,
            ci_status="pending",  # Would be actual CI status
            error_message=None
        )
    
    def create_checkpoint(self, label: str) -> Checkpoint:
        """Create a safe checkpoint before risky operations."""
        checkpoint_id = str(uuid.uuid4())
        
        # Get current commit
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        commit_hash = result.stdout.strip()
        
        # Get current branch
        branch = self.get_git_status().branch
        
        # Get file snapshot
        result = subprocess.run(
            ['git', 'ls-tree', '-r', '--name-only', 'HEAD'],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        files_snapshot = result.stdout.strip().split('\n')
        
        # Save checkpoint
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            label=label,
            commit_hash=commit_hash,
            branch=branch,
            timestamp=datetime.now().isoformat(),
            files_snapshot=files_snapshot
        )
        
        # Store checkpoint
        checkpoint_path = self.checkpoints_path / f"{checkpoint_id}.json"
        checkpoint_data = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "label": checkpoint.label,
            "commit_hash": checkpoint.commit_hash,
            "branch": checkpoint.branch,
            "timestamp": checkpoint.timestamp,
            "files_snapshot": checkpoint.files_snapshot
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Create git tag as well
        tag_name = f"a0-checkpoint-{checkpoint_id[:8]}"
        subprocess.run(
            ['git', 'tag', '-a', tag_name, '-m', f"Agent Zero Checkpoint: {label}"],
            cwd=self.repo_path,
            capture_output=True
        )
        
        return checkpoint
    
    def rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """Rollback to a safe checkpoint."""
        # Load checkpoint
        checkpoint_path = self.checkpoints_path / f"{checkpoint_id}.json"
        if not checkpoint_path.exists():
            return False
        
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        commit_hash = checkpoint_data["commit_hash"]
        
        # Reset to checkpoint commit
        result = subprocess.run(
            ['git', 'reset', '--hard', commit_hash],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        
        return result.returncode == 0
    
    def get_available_checkpoints(self) -> List[Checkpoint]:
        """List all available checkpoints."""
        checkpoints = []
        
        for file_path in self.checkpoints_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                checkpoints.append(Checkpoint(
                    checkpoint_id=data["checkpoint_id"],
                    label=data["label"],
                    commit_hash=data["commit_hash"],
                    branch=data["branch"],
                    timestamp=data["timestamp"],
                    files_snapshot=data["files_snapshot"]
                ))
            except Exception:
                pass
        
        # Sort by timestamp descending
        checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
        return checkpoints
