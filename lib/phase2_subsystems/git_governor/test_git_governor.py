"""
Git Governor - Comprehensive tests.
Validates all Git Governor functionality.
"""
import unittest
import tempfile
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, '/a0/lib')

from phase2_subsystems.git_governor.interfaces import (
    RiskLevel, CommitType, ApprovalStatus, FileChange, CommitRequest
)
from phase2_subsystems.git_governor.commit_validator import (
    CommitValidator, CommitValidatorConfig
)
from phase2_subsystems.git_governor.approval_workflow import (
    ApprovalWorkflow, ApprovalConfig, ApprovalRequest
)
from phase2_subsystems.git_governor.git_governor import GitGovernor


class TestCommitValidator(unittest.TestCase):
    """Test CommitValidator functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)
        self._init_git_repo()
        self.validator = CommitValidator(repo_path=self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def _init_git_repo(self):
        """Initialize a test Git repository."""
        subprocess.run(['git', 'init'], cwd=self.temp_dir, check=True,
                      capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=self.temp_dir, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@test.com'], cwd=self.temp_dir, check=True)
        
        test_file = self.repo_path / "test.txt"
        test_file.write_text("initial content")
        subprocess.run(['git', 'add', 'test.txt'], cwd=self.temp_dir, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=self.temp_dir, check=True,
                      capture_output=True)
    
    def test_validate_branch_protected(self):
        """Test validation of protected branches."""
        result = self.validator.validate_branch('main')
        self.assertTrue(result.is_protected)
        self.assertFalse(result.can_commit)
        
        result = self.validator.validate_branch('feature/test-branch')
        self.assertFalse(result.is_protected)
        self.assertTrue(result.can_commit)
    
    def test_check_secrets(self):
        """Test secret detection."""
        result = self.validator.check_secrets("no secrets here")
        self.assertFalse(result.has_secrets)
        self.assertEqual(result.risk_level, RiskLevel.LOW)
        
        result = self.validator.check_secrets("password = 'secret123'")
        self.assertTrue(result.has_secrets)
        self.assertEqual(result.risk_level, RiskLevel.CRITICAL)
        
        result = self.validator.check_secrets("api_key=sk-abc123xyz456")
        self.assertTrue(result.has_secrets)
    
    def test_analyze_risk_low(self):
        """Test risk assessment for low-risk changes."""
        changes = [
            FileChange(
                path="docs/readme.md",
                change_type="modified",
                lines_added=10,
                lines_removed=5,
                content_preview="some changes"
            )
        ]
        
        result = self.validator.analyze_risk(changes)
        self.assertEqual(result.overall_risk, RiskLevel.LOW)
        self.assertFalse(result.requires_approval)
        self.assertGreater(result.confidence, 0.7)
    
    def test_analyze_risk_high(self):
        """Test risk assessment for high-risk changes."""
        changes = [
            FileChange(
                path=".env.production",
                change_type="modified",
                lines_added=50,
                lines_removed=10,
                content_preview="config changes"
            )
        ]
        
        result = self.validator.analyze_risk(changes)
        self.assertEqual(result.overall_risk, RiskLevel.CRITICAL)
        self.assertTrue(result.requires_approval)
        self.assertGreater(len(result.risk_factors), 0)
    
    def test_analyze_risk_deletion(self):
        """Test risk assessment for file deletion."""
        changes = [
            FileChange(
                path="important_file.py",
                change_type="deleted",
                lines_added=0,
                lines_removed=100,
                content_preview="deleted"
            )
        ]
        
        result = self.validator.analyze_risk(changes)
        self.assertEqual(result.overall_risk, RiskLevel.CRITICAL)
        self.assertTrue(result.requires_approval)
    
    def test_generate_semantic_message(self):
        """Test semantic commit message generation."""
        request = CommitRequest(
            files=["test.py"],
            change_description="add new feature",
            commit_type=CommitType.FEAT,
            scope="api",
            body=None,
            footer=None
        )
        
        message = self.validator.generate_semantic_message(request)
        self.assertIn("feat(api): add new feature", message)
        
        request = CommitRequest(
            files=["test.py"],
            change_description="fix bug",
            commit_type=CommitType.FIX,
            body="Fixed the issue with...",
            footer="Closes #123"
        )
        
        message = self.validator.generate_semantic_message(request)
        self.assertIn("fix: fix bug", message)
        self.assertIn("Fixed the issue with...", message)
        self.assertIn("Closes #123", message)


class TestApprovalWorkflow(unittest.TestCase):
    """Test ApprovalWorkflow functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.workflow = ApprovalWorkflow(
            config=ApprovalConfig(
                approval_timeout_minutes=30,
                auto_approve_levels=[RiskLevel.LOW]
            ),
            storage_path=self.temp_dir
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_request_approval_low_risk(self):
        """Test approval request for low risk (auto-approve)."""
        from phase2_subsystems.git_governor.interfaces import (
            BranchValidation, SecretCheck, RiskAssessment, CommitValidation
        )
        
        branch_valid = BranchValidation(
            is_valid=True,
            branch_name="feature/test",
            is_protected=False,
            can_commit=True,
            error_message=None
        )
        
        secret_check = SecretCheck(
            has_secrets=False,
            secrets_found=[],
            risk_level=RiskLevel.LOW
        )
        
        risk_assessment = RiskAssessment(
            overall_risk=RiskLevel.LOW,
            risk_factors=[],
            requires_approval=False,
            recommendations=[],
            confidence=0.9
        )
        
        validation = CommitValidation(
            is_valid=True,
            branch_valid=branch_valid,
            secret_check=secret_check,
            risk_assessment=risk_assessment,
            can_proceed=True,
            errors=[],
            warnings=[]
        )
        
        response = self.workflow.request_approval(validation)
        self.assertEqual(response.status, ApprovalStatus.APPROVED)
        self.assertEqual(response.approved_by, "auto_approved")
    
    def test_request_approval_high_risk(self):
        """Test approval request for high risk (requires approval)."""
        from phase2_subsystems.git_governor.interfaces import (
            BranchValidation, SecretCheck, RiskAssessment, CommitValidation
        )
        
        branch_valid = BranchValidation(
            is_valid=True,
            branch_name="feature/test",
            is_protected=False,
            can_commit=True,
            error_message=None
        )
        
        secret_check = SecretCheck(
            has_secrets=False,
            secrets_found=[],
            risk_level=RiskLevel.LOW
        )
        
        risk_assessment = RiskAssessment(
            overall_risk=RiskLevel.HIGH,
            risk_factors=["Large change"],
            requires_approval=True,
            recommendations=["Request review"],
            confidence=0.75
        )
        
        validation = CommitValidation(
            is_valid=True,
            branch_valid=branch_valid,
            secret_check=secret_check,
            risk_assessment=risk_assessment,
            can_proceed=True,
            errors=[],
            warnings=[]
        )
        
        response = self.workflow.request_approval(validation)
        self.assertEqual(response.status, ApprovalStatus.PENDING)
        self.assertIsNone(response.approved_by)
    
    def test_is_approval_required(self):
        """Test approval requirement logic."""
        self.assertFalse(self.workflow.is_approval_required(RiskLevel.LOW))
        self.assertTrue(self.workflow.is_approval_required(RiskLevel.MEDIUM))
        self.assertTrue(self.workflow.is_approval_required(RiskLevel.HIGH))
        self.assertTrue(self.workflow.is_approval_required(RiskLevel.CRITICAL))
    
    def test_manual_approve_reject(self):
        """Test manual approve and reject."""
        from phase2_subsystems.git_governor.interfaces import (
            BranchValidation, SecretCheck, RiskAssessment, CommitValidation
        )
        
        branch_valid = BranchValidation(
            is_valid=True,
            branch_name="feature/test",
            is_protected=False,
            can_commit=True,
            error_message=None
        )
        
        secret_check = SecretCheck(
            has_secrets=False,
            secrets_found=[],
            risk_level=RiskLevel.LOW
        )
        
        risk_assessment = RiskAssessment(
            overall_risk=RiskLevel.LOW,
            risk_factors=[],
            requires_approval=False,
            recommendations=[],
            confidence=0.9
        )
        
        commit_validation = CommitValidation(
            is_valid=True,
            branch_valid=branch_valid,
            secret_check=secret_check,
            risk_assessment=risk_assessment,
            can_proceed=True,
            errors=[],
            warnings=[]
        )
        
        request_id = "test-request-123"
        self.workflow._requests[request_id] = ApprovalRequest(
            request_id=request_id,
            commit_validation=commit_validation,
            requested_by="agent_zero",
            created_at=datetime.now()
        )
        
        response = self.workflow.approve_request(request_id, "admin", "Looks good")
        self.assertEqual(response.status, ApprovalStatus.APPROVED)
        self.assertEqual(response.approved_by, "admin")
        
        self.workflow._requests[request_id].status = ApprovalStatus.PENDING
        response = self.workflow.reject_request(request_id, "admin", "Needs changes")
        self.assertEqual(response.status, ApprovalStatus.REJECTED)
        self.assertEqual(response.approval_comment, "Needs changes")


class TestGitGovernor(unittest.TestCase):
    """Test GitGovernor main functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)
        self._init_git_repo()
        self.governor = GitGovernor(repo_path=self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def _init_git_repo(self):
        """Initialize a test Git repository."""
        subprocess.run(['git', 'init'], cwd=self.temp_dir, check=True,
                      capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=self.temp_dir, check=True)
        subprocess.run(['git', 'config', 'user.email', 'test@test.com'], cwd=self.temp_dir, check=True)
        
        test_file = self.repo_path / "test.txt"
        test_file.write_text("initial content")
        subprocess.run(['git', 'add', 'test.txt'], cwd=self.temp_dir, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], cwd=self.temp_dir, check=True,
                      capture_output=True)
    
    def test_get_git_status(self):
        """Test getting Git status."""
        status = self.governor.get_git_status()
        self.assertIn(status.branch, ['master', 'main'])
        self.assertTrue(status.is_clean)
        self.assertEqual(len(status.modified_files), 0)
    
    def test_get_git_status_dirty(self):
        """Test getting Git status with changes."""
        test_file = self.repo_path / "test.txt"
        test_file.write_text("modified content")
        
        status = self.governor.get_git_status()
        # Git status might show modified files, untracked, or staged
        # The key is that is_clean should be False
        self.assertFalse(status.is_clean, "Repository should not be clean")
        # Check we have some kind of change
        total_changes = len(status.modified_files) + len(status.staged_files) + len(status.untracked_files)
        self.assertGreater(total_changes, 0, "Should have some changes")
    
    def test_get_git_status_staged(self):
        """Test getting Git status with staged changes."""
        test_file = self.repo_path / "test2.txt"
        test_file.write_text("new file")
        subprocess.run(['git', 'add', 'test2.txt'], cwd=self.temp_dir, check=True,
                      capture_output=True)
        
        status = self.governor.get_git_status()
        self.assertFalse(status.is_clean)
        self.assertGreater(len(status.staged_files), 0)
    
    def test_create_commit(self):
        """Test creating a commit."""
        test_file = self.repo_path / "test.txt"
        test_file.write_text("modified content")
        
        commit = self.governor.create_commit("test commit", ["test.txt"])
        
        self.assertIsNotNone(commit.commit_hash)
        self.assertIn(commit.branch, ['master', 'main'])
        self.assertEqual(commit.message, "test commit")
        
        status = self.governor.get_git_status()
        self.assertTrue(status.is_clean)
    
    def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        checkpoint = self.governor.create_checkpoint("test checkpoint")
        
        self.assertIsNotNone(checkpoint.checkpoint_id)
        self.assertEqual(checkpoint.label, "test checkpoint")
        self.assertIn(checkpoint.branch, ['master', 'main'])
        
        checkpoints = self.governor.get_available_checkpoints()
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0].checkpoint_id, checkpoint.checkpoint_id)
    
    def test_rollback_to_checkpoint(self):
        """Test rollback to checkpoint."""
        checkpoint = self.governor.create_checkpoint("before changes")
        
        test_file = self.repo_path / "test.txt"
        test_file.write_text("modified content")
        new_file = self.repo_path / "new.txt"
        new_file.write_text("new file")
        subprocess.run(['git', 'add', '.'], cwd=self.temp_dir, check=True,
                      capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'new commit'], cwd=self.temp_dir, check=True,
                      capture_output=True)
        
        success = self.governor.rollback_to_checkpoint(checkpoint.checkpoint_id)
        self.assertTrue(success)
        
        status = self.governor.get_git_status()
        self.assertEqual(status.last_commit[:7], checkpoint.commit_hash[:7])


def run_tests():
    """Run all Git Governor tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestCommitValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestApprovalWorkflow))
    suite.addTests(loader.loadTestsFromTestCase(TestGitGovernor))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
