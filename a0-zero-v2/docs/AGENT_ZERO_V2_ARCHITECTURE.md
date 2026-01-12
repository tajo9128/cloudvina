# Agent Zero v2 Architecture
## Exceeding Google Antigravity - IDE-Grade Autonomous Agent System

---

## 📌 Executive Summary

Agent Zero v2 transforms from a conversational AI agent into a **full-fledged IDE-grade autonomous development system** that surpasses Google Antigravity in:

- **Transparency & Explainability** ✅ (Antigravity: Limited)
- **Control & Customization** ✅ (Antigravity: None)
- **Open Architecture** ✅ (Antigravity: Closed)
- **Cost-Optimized Model Routing** ✅ (Antigravity: Hidden)

---

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Zero v2 Core System                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   User UI    │  │   CLI/API    │  │   Webhooks   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │                                       │
│                    ┌─────▼─────┐                                 │
│                    │  Orchest- │                                 │
│                    │   rator   │                                 │
│                    └─────┬─────┘                                 │
│                          │                                       │
│  ┌───────────┬───────────┼───────────┬───────────┬───────────┐  │
│  │           │           │           │           │           │  │
│  ▼           ▼           ▼           ▼           ▼           │  │
│┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │  │
││ Repo   │ │   Git  │ │Policy  │ │ Model  │ │ Audit  │        │  │
││ Aware- │ │Governor│ │ Engine │ │ Router │ │ Layer  │        │  │
││  ness  │ │        │ │        │ │        │ │        │        │  │
│└───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘        │  │
│    │          │          │          │          │              │  │
│    └──────────┼──────────┼──────────┼──────────┘              │  │
│               │          │          │                         │  │
│         ┌─────▼─────┐ ┌──▼──────────▼───┐                     │  │
│         │ Execution │ │  Safety &       │                     │  │
│         │  Engine   │ │ Validation      │                     │  │
│         └─────┬─────┘ └──────┬───────────┘                     │  │
│               │              │                                 │  │
│         ┌─────▼──────────────▼─────┐                          │  │
│         │      Tool Registry        │                          │  │
│         └──────────────────────────┘                          │  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔷 MODULE 1: Repo Awareness
### Persistent Project State Engine

#### Purpose
Maintain comprehensive, persistent knowledge of the entire repository state across all sessions.

#### Components

##### 1.1 File System Indexer
```python
class FileSystemIndexer:
    """Maintains complete file tree snapshot"""
    
    def index_project(self, project_path: str) -> ProjectIndex:
        """
        Returns:
        - file_tree: Complete directory structure
        - file_hashes: SHA-256 for each file
        - file_metadata: Size, type, language, last_modified
        - ignored_files: .gitignore patterns matched
        """
        pass
    
    def detect_changes(self, old_index: ProjectIndex, new_index: ProjectIndex) -> Diff:
        """Detect added, modified, deleted files"""
        pass
```

##### 1.2 Dependency Graph Builder
```python
class DependencyGraphBuilder:
    """Builds import/reference relationships"""
    
    SUPPORTED_LANGUAGES = [
        'python', 'javascript', 'typescript', 
        'go', 'rust', 'java', 'cpp'
    ]
    
    def build_dependency_graph(self, project_index: ProjectIndex) -> DependencyGraph:
        """
        Analyzes:
        - Import statements
        - Module requirements
        - Build files (package.json, requirements.txt, go.mod, Cargo.toml)
        - Reference relationships
        
        Returns graph with nodes (files/modules) and edges (dependencies)
        """
        pass
    
    def find_affected_files(self, changed_file: str) -> List[str]:
        """Find all files that depend on changed file"""
        pass
```

##### 1.3 Git Metadata Reader
```python
class GitMetadataReader:
    """Reads and analyzes git history"""
    
    def get_recent_commits(self, limit: int = 20) -> List[CommitInfo]:
        """Get recent commits with diffs"""
        pass
    
    def get_branches(self) -> List[BranchInfo]:
        """Get all branches and their status"""
        pass
    
    def get_open_todos(self) -> List[TodoItem]:
        """Scan commits, issues, PR comments for TODO/FIXME"""
        pass
```

##### 1.4 Semantic Code Indexer
```python
class SemanticCodeIndexer:
    """LLM-powered semantic understanding of codebase"""
    
    def index_functions(self, files: List[str]) -> FunctionIndex:
        """Extract and index all functions/classes"""
        pass
    
    def index_documentation(self, files: List[str]) -> DocIndex:
        """Extract and index docstrings, comments"""
        pass
    
    def search_semantic(self, query: str) -> List[CodeMatch]:
        """Semantic search across entire codebase"""
        pass
```

#### Storage Schema
```python
@dataclass
class ProjectState:
    project_id: str
    project_path: str
    file_index: FileIndex
    dependency_graph: DependencyGraph
    git_metadata: GitMetadata
    semantic_index: SemanticIndex
    last_updated: datetime
    version_hash: str  # Hash of entire state

@dataclass
class FileIndex:
    files: Dict[str, FileInfo]  # relative_path -> FileInfo
    tree_structure: Dict  # nested directory tree

@dataclass
class FileInfo:
    path: str
    hash: str
    size: int
    language: str
    last_modified: datetime
    imports: List[str]
    functions: List[FunctionDef]
```

#### Data Flow
```
Git Repo → File System Indexer → Dependency Graph Builder
                                            ↓
                                    Semantic Code Indexer
                                            ↓
                                   Project State Database
```

#### API
```python
# Query API
class RepoAwarenessAPI:
    def get_project_state(self, project_id: str) -> ProjectState:
        """Get complete project state"""
        pass
    
    def find_files(self, pattern: str, language: str = None) -> List[str]:
        """Find files matching pattern"""
        pass
    
    def get_dependencies(self, file_path: str) -> DependencyList:
        """Get all dependencies for a file"""
        pass
    
    def get_dependents(self, file_path: str) -> List[str]:
        """Get all files that depend on this file"""
        pass
    
    def search_code(self, query: str, semantic: bool = True) -> List[CodeMatch]:
        """Search codebase (semantic or literal)"""
        pass
```

---

---

## 🔷 MODULE 2: Git Commits
### Git Governor

#### Purpose
Provide governed, safe, and automated Git operations with full traceability and approval workflows.

#### Components

##### 2.1 Diff Analyzer
```python
class DiffAnalyzer:
    """Analyzes git diffs for safety and quality"""
    
    def analyze_diff(self, diff_str: str) -> DiffAnalysis:
        """
        Returns:
        - affected_files: List of changed files
        - lines_added: Count of added lines
        - lines_removed: Count of removed lines
        - risk_level: LOW, MEDIUM, HIGH, CRITICAL
        - risk_factors: List of detected risks
        - review_points: Points requiring human review
        """
        pass
    
    def detect_risks(self, diff: str) -> List[Risk]:
        """
        Detects:
        - Hardcoded secrets
        - Production config changes
        - Database schema changes
        - Breaking API changes
        - Security vulnerabilities
        - Deleted critical files
        """
        pass
```

##### 2.2 Commit Synthesizer
```python
class CommitSynthesizer:
    """Generates semantic commit messages"""
    
    def generate_commit_message(self, changes: List[FileChange], context: ProjectState) -> CommitMessage:
        """
        Generates:
        - Conventional commit format (type: description)
        - Detailed body with change summary
        - Breaking change notes
        - Issue/PR references
        
        Types: feat, fix, docs, style, refactor, test, chore
        """
        pass
    
    def suggest_branch_name(self, changes: List[FileChange]) -> str:
        """Suggest branch name based on changes"""
        pass
```

##### 2.3 Branch Manager
```python
class BranchManager:
    """Manages Git branches with safety checks"""
    
    def create_feature_branch(self, base_branch: str, branch_name: str) -> BranchResult:
        """Create feature branch from base"""
        pass
    
    def validate_branch_safe_for_push(self, branch_name: str) -> Validation:
        """
        Validates:
        - Not pushing to protected branches (main, master, production)
        - CI checks passed
        - No merge conflicts
        """
        pass
    
    def can_safely_merge(self, branch: str, target: str) -> SafetyCheck:
        """Check if branch can be safely merged"""
        pass
```

##### 2.4 Push Approval Engine
```python
class PushApprovalEngine:
    """Manages approval workflows for git operations"""
    
    def request_commit_approval(self, commit: PendingCommit) -> ApprovalRequest:
        """Request human approval for commit"""
        pass
    
    def request_push_approval(self, branch: str, changes: DiffSummary) -> ApprovalRequest:
        """Request human approval for push"""
        pass
    
    def auto_approve(self, commit: PendingCommit, policy: Policy) -> bool:
        """Auto-approve if meets policy criteria"""
        pass
```

##### 2.5 Rollback Manager
```python
class RollbackManager:
    """Manages safe rollbacks"""
    
    def create_rollback_commit(self, commit_to_rollback: str) -> str:
        """Create revert commit"""
        pass
    
    def create_safe_branch_point(self) -> str:
        """Save current state for potential rollback"""
        pass
    
    def rollback_to_safe_point(self, safe_point: str) -> bool:
        """Rollback to saved safe point"""
        pass
```

#### State Machine
```python
class GitGovernanceState(Enum):
    IDLE = "idle"
    STAGING = "staging"
    COMMIT_PENDING_APPROVAL = "commit_pending_approval"
    COMMITTED = "committed"
    PUSH_PENDING_APPROVAL = "push_pending_approval"
    PUSHED = "pushed"
    ROLLBACK_PENDING = "rollback_pending"
```

#### API
```python
class GitGovernorAPI:
    def stage_changes(self, files: List[str]) -> StageResult:
        """Stage files with diff analysis"""
        pass
    
    def commit(self, message: str = None, auto_approve: bool = False) -> CommitResult:
        """Commit staged changes with optional approval"""
        pass
    
    def push(self, branch: str, force: bool = False, auto_approve: bool = False) -> PushResult:
        """Push with safety checks and approval"""
        pass
    
    def create_branch(self, name: str, base: str = None) -> BranchResult:
        """Create new branch"""
        pass
    
    def rollback(self) -> RollbackResult:
        """Rollback to safe point"""
        pass
```

---

## 🔷 MODULE 3: Safe Autonomy
### Hard Execution Pipeline

#### Purpose
Ensure all agent actions pass through rigorous validation and safety checks before execution.

#### Components

##### 3.1 Intent Parser
```python
class IntentParser:
    """Understands and validates user intent"""
    
    def parse_intent(self, user_input: str, context: ConversationContext) -> Intent:
        """
        Returns:
        - intent_type: CODE, FILE, GIT, SYSTEM, QUERY
        - target_affected: What will be affected
        - risk_level: LOW, MEDIUM, HIGH, CRITICAL
        - confidence: Confidence score
        """
        pass
    
    def detect_ambiguous_intent(self, input: str) -> List[Ambiguity]:
        """Detect ambiguous requests needing clarification"""
        pass
```

##### 3.2 Action Planner
```python
class ActionPlanner:
    """Creates safe execution plans"""
    
    def create_plan(self, intent: Intent, context: ProjectState) -> ExecutionPlan:
        """
        Creates step-by-step plan:
        1. Read/understand affected files
        2. Generate changes
        3. Validate changes
        4. Stage changes
        5. Commit (with approval)
        
        Each step has:
        - pre_conditions: Must be true before
        - action: What to do
        - post_conditions: Must be true after
        - rollback: How to undo
        """
        pass
    
    def estimate_impact(self, plan: ExecutionPlan) -> ImpactAnalysis:
        """Estimate impact of plan execution"""
        pass
```

##### 3.3 Execution Validator
```python
class ExecutionValidator:
    """Validates actions before execution"""
    
    def validate_action(self, action: Action, context: Context) -> ValidationResult:
        """
        Validates:
        - File operations: Not deleting critical files
        - Commands: No destructive commands without approval
        - Git ops: Protected branch protection
        - Dependencies: Not breaking existing code
        """
        pass
    
    def check_file_deletion_safety(self, file_path: str) -> SafetyCheck:
        """
        Checks:
        - Is file in use?
        - Are there dependents?
        - Is file critical to project?
        """
        pass
    
    def validate_command_safety(self, command: str) -> SafetyCheck:
        """
        Checks for:
        - rm -rf patterns
        - Database drops
        - Production deployments
        - Secret exposure
        """
        pass
```

##### 3.4 Environment Locks
```python
class EnvironmentLockManager:
    """Manages environment-level locks"""
    
    ENVIRONMENTS = ['development', 'staging', 'production']
    
    def get_lock_state(self, environment: str) -> LockState:
        """Get current lock state"""
        pass
    
    def lock_environment(self, environment: str, reason: str) -> bool:
        """Lock environment for changes"""
        pass
    
    def can_modify_environment(self, environment: str, action: Action) -> bool:
        """Check if action allowed in environment"""
        pass
```

##### 3.5 Command Allow/Deny List
```python
class CommandFilter:
    """Filter commands based on policy"""
    
    ALLOWED_COMMANDS = {
        'development': ['*', ],  # All commands in dev
        'staging': [
            'git *', 'npm *', 'pip *', 'pytest *',
            'mvn *', 'cargo *', 'go build *'
        ],
        'production': [
            'git log', 'git status', 'git diff',
            'kubectl logs', 'kubectl get pods'
        ]
    }
    
    DENIED_COMMANDS = [
        'rm -rf', 'drop database', 'truncate',
        'delete from users', 'shutdown', 'reboot'
    ]
    
    def check_command_allowed(self, command: str, environment: str) -> FilterResult:
        """Check if command is allowed"""
        pass
```

##### 3.6 Rollback Engine
```python
class RollbackEngine:
    """Handles automatic rollbacks on failure"""
    
    def create_rollback_checkpoint(self) -> str:
        """Create checkpoint before risky action"""
        pass
    
    def trigger_rollback(self, checkpoint: str, reason: str) -> bool:
        """Rollback to checkpoint"""
        pass
    
    def auto_rollback_on_failure(self, step: Action, result: ExecutionResult) -> bool:
        """Decide if rollback needed"""
        pass
```

#### Pipeline Stages
```python
class ExecutionPipeline:
    """Hard execution pipeline"""
    
    STAGES = [
        'INTENT_PARSING',
        'PLANNING',
        'VALIDATION',
        'APPROVAL',
        'EXECUTION',
        'VERIFICATION',
        'AUDIT'
    ]
    
    def execute(self, request: UserRequest) -> PipelineResult:
        """
        Flow:
        1. Parse intent
        2. Create plan
        3. Validate each action
        4. Request approval if needed
        5. Execute with rollback checkpoint
        6. Verify results
        7. Log to audit
        """
        pass
```

---

---

## 🔷 MODULE 4: Model Routing
### Intelligent Model Router

#### Purpose
Automatically route tasks to the most appropriate LLM model based on task type, complexity, cost, and quality requirements.

#### Components

##### 4.1 Task Classifier
```python
class TaskClassifier:
    """Classifies incoming tasks for routing"""
    
    TASK_TYPES = [
        'PLANNING',          # High-level strategy
        'CODE_GENERATION',   # Writing new code
        'REFACTORING',       # Restructuring existing code
        'DEBUGGING',         # Finding and fixing bugs
        'TEST_GENERATION',   # Writing tests
        'DOCUMENTATION',     # Writing docs
        'QUERY',             # Simple Q&A
        'ANALYSIS',          # Code analysis
        'SUMMARIZATION',      # Summarizing content
        'TRANSLATION',       # Code translation
        'OPTIMIZATION',      # Performance tuning
    ]
    
    def classify(self, user_input: str, context: Context) -> TaskClassification:
        """
        Returns:
        - task_type: Primary task type
        - subtasks: List of subtasks
        - complexity: LOW, MEDIUM, HIGH
        - estimated_tokens: Token estimate
        - urgency: NORMAL, HIGH, CRITICAL
        """
        pass
```

##### 4.2 Model Selector
```python
class ModelSelector:
    """Selects optimal model for each task"""
    
    # Available models with metadata
    MODELS = {
        'glm-4.7': {
            'provider': 'openai_compatible',
            'strengths': ['PLANNING', 'CODE_GENERATION', 'REFACTORING'],
            'weaknesses': [],
            'cost_per_1k_tokens': 0.01,
            'max_tokens': 128000,
            'speed': 'fast',
            'quality': 'high'
        },
        'gpt-4': {
            'provider': 'openai',
            'strengths': ['REFACTORING', 'DEBUGGING', 'ANALYSIS'],
            'weaknesses': ['COST'],
            'cost_per_1k_tokens': 0.03,
            'max_tokens': 8192,
            'speed': 'medium',
            'quality': 'very_high'
        },
        'claude-3': {
            'provider': 'anthropic',
            'strengths': ['CODE_GENERATION', 'DOCUMENTATION'],
            'weaknesses': ['COST'],
            'cost_per_1k_tokens': 0.015,
            'max_tokens': 100000,
            'speed': 'medium',
            'quality': 'high'
        },
        'mistral-7b': {
            'provider': 'ollama',
            'strengths': ['QUERY', 'SIMPLE_TASKS'],
            'weaknesses': ['COMPLEX_REASONING'],
            'cost_per_1k_tokens': 0,
            'max_tokens': 4096,
            'speed': 'very_fast',
            'quality': 'medium'
        },
    }
    
    def select_model(self, task: TaskClassification, budget: Budget = None) -> ModelSelection:
        """
        Selects model based on:
        - Task type match
        - Complexity level
        - Cost constraints
        - Quality requirements
        - Speed requirements
        """
        pass
    
    def calculate_cost(self, model: str, estimated_tokens: int) -> CostEstimate:
        """Estimate cost for task"""
        pass
```

##### 4.3 Fallback Manager
```python
class FallbackManager:
    """Manages fallback chain when models fail"""
    
    def get_fallback_chain(self, primary_model: str, task: TaskClassification) -> List[str]:
        """Returns ordered list of fallback models"""
        pass
    
    def should_fallback(self, error: Exception, attempt: int) -> bool:
        """Decide if fallback should trigger"""
        pass
    
    def execute_with_fallback(self, task: Task, models: List[str]) -> TaskResult:
        """Execute task with fallback chain"""
        pass
```

##### 4.4 Cost Optimizer
```python
class CostOptimizer:
    """Optimizes model selection for cost efficiency"""
    
    def optimize_routing(self, tasks: List[Task]) -> RoutingPlan:
        """
        Strategies:
        - Group similar tasks
        - Batch when possible
        - Use cheaper models for simple tasks
        - Cache results
        """
        pass
    
    def estimate_session_cost(self, tasks: List[Task]) -> CostEstimate:
        """Estimate total session cost"""
        pass
```

##### 4.5 Quality Tracker
```python
class QualityTracker:
    """Tracks model quality metrics"""
    
    def record_result(self, model: str, task: Task, result: TaskResult, quality: float):
        """Record model performance"""
        pass
    
    def get_model_stats(self, model: str, time_period: timedelta) -> ModelStats:
        """Get model performance statistics"""
        pass
    
    def recommend_model(self, task_type: str) -> str:
        """Recommend best model for task type based on history"""
        pass
```

#### Routing Rules
```python
class RoutingRules:
    """Define explicit routing rules"""
    
    RULES = [
        # Planning: Use best model for reasoning
        Rule(
            condition=TaskType.PLANNING,
            models=['glm-4.7', 'gpt-4'],
            priority='quality'
        ),
        
        # Code Generation: Balance quality and speed
        Rule(
            condition=TaskType.CODE_GENERATION,
            models=['glm-4.7', 'claude-3'],
            priority='balanced'
        ),
        
        # Tests: Use fast, cheap model
        Rule(
            condition=TaskType.TEST_GENERATION,
            models=['mistral-7b', 'glm-4.7'],
            priority='speed'
        ),
        
        # Documentation: Use model good at natural language
        Rule(
            condition=TaskType.DOCUMENTATION,
            models=['claude-3', 'glm-4.7'],
            priority='quality'
        ),
        
        # Simple Queries: Use cheapest
        Rule(
            condition=TaskType.QUERY & Complexity.LOW,
            models=['mistral-7b'],
            priority='cost'
        ),
    ]
```

#### API
```python
class ModelRouterAPI:
    def route_task(self, task: str, context: Context) -> RoutingDecision:
        """Route task to best model"""
        pass
    
    def execute_with_routing(self, prompt: str, context: Context) -> str:
        """Execute with automatic routing"""
        pass
    
    def get_routing_stats(self) -> RoutingStats:
        """Get routing statistics"""
        pass
    
    def set_custom_rule(self, rule: RoutingRule) -> bool:
        """Add custom routing rule"""
        pass
```

---

## 🔷 MODULE 5: Explainability
### Self-Audit Layer

#### Purpose
Provide complete transparency, traceability, and auditability of all agent decisions and actions.

#### Components

##### 5.1 Decision Logger
```python
class DecisionLogger:
    """Logs all decisions with reasoning"""
    
    def log_decision(self, decision: Decision):
        """
        Logs:
        - timestamp: When decision was made
        - decision_type: What type of decision
        - intent: User's original intent
        - reasoning: Why this decision was made
        - alternatives_considered: What alternatives were evaluated
        - rejected_alternatives: Why alternatives were rejected
        - model_used: Which LLM made the decision
        - confidence: Confidence in decision
        - risk_assessment: Risk level of decision
        """
        pass
    
    def get_decision_history(self, task_id: str) -> List[Decision]:
        """Get all decisions for a task"""
        pass
```

##### 5.2 Plan Auditor
```python
class PlanAuditor:
    """Audits execution plans"""
    
    def audit_plan(self, plan: ExecutionPlan) -> PlanAudit:
        """
        Returns:
        - plan_quality_score: 0-100
        - missing_steps: Steps that might be missing
        - risk_factors: Risks in the plan
        - estimated_time: Time estimate
        - estimated_cost: Cost estimate
        - prerequisites: Dependencies
        """
        pass
    
    def compare_plans(self, plan1: ExecutionPlan, plan2: ExecutionPlan) -> Comparison:
        """Compare two execution plans"""
        pass
```

##### 5.3 Execution Tracer
```python
class ExecutionTracer:
    """Traces execution in detail"""
    
    def trace_execution(self, execution_id: str) -> ExecutionTrace:
        """
        Returns:
        - steps: Each step executed
        - inputs: Inputs to each step
        - outputs: Outputs from each step
        - errors: Any errors encountered
        - time_taken: Time per step
        - resources_used: Resources consumed
        """
        pass
    
    def visualize_trace(self, execution_id: str) -> Visualization:
        """Generate visual representation of execution"""
        pass
```

##### 5.4 Diff Summarizer
```python
class DiffSummarizer:
    """Summarizes code changes"""
    
    def summarize_diff(self, diff: str) -> DiffSummary:
        """
        Returns:
        - files_changed: List of changed files
        - summary: High-level summary
        - key_changes: Key changes made
        - impact_analysis: Impact of changes
        - test_implications: What tests to run
        - potential_issues: Potential issues introduced
        """
        pass
```

##### 5.5 Risk Assessor
```python
class RiskAssessor:
    """Assesses risks of actions"""
    
    def assess_risk(self, action: Action, context: Context) -> RiskAssessment:
        """
        Returns:
        - risk_level: LOW, MEDIUM, HIGH, CRITICAL
        - risk_factors: List of risk factors
        - mitigation_strategies: How to mitigate risks
        - approval_required: If approval required
        """
        pass
    
    def get_risk_history(self, action_type: str) -> RiskHistory:
        """Get historical risk data"""
        pass
```

##### 5.6 Audit Report Generator
```python
class AuditReportGenerator:
    """Generates comprehensive audit reports"""
    
    def generate_task_report(self, task_id: str) -> AuditReport:
        """
        Report includes:
        - Task summary
        - Intent and goals
        - Plan created
        - Decisions made (with reasoning)
        - Actions taken
        - Changes made (with diffs)
        - Risks assessed
        - Results achieved
        - Errors encountered
        - Recommendations
        """
        pass
    
    def generate_session_report(self, session_id: str) -> AuditReport:
        """Generate report for entire session"""
        pass
```

#### Audit Storage
```python
@dataclass
class AuditEntry:
    entry_id: str
    timestamp: datetime
    task_id: str
    user_id: str
    
    # Decision info
    decision_type: DecisionType
    intent: str
    reasoning: str
    alternatives_considered: List[Alternative]
    rejected_alternatives: List[RejectionReason]
    
    # Execution info
    model_used: str
    confidence: float
    risk_assessment: RiskAssessment
    
    # Result info
    action_taken: Action
    result: ExecutionResult
    changes: DiffSummary
```

#### API
```python
class ExplainabilityAPI:
    def explain_decision(self, decision_id: str) -> DecisionExplanation:
        """Get detailed explanation of a decision"""
        pass
    
    def trace_execution(self, execution_id: str) -> ExecutionTrace:
        """Get execution trace"""
        pass
    
    def get_audit_report(self, task_id: str) -> AuditReport:
        """Get audit report for task"""
        pass
    
    def query_decisions(self, query: DecisionQuery) -> List[Decision]:
        """Query decision history"""
        pass
    
    def export_audit_logs(self, format: str = 'json') -> bytes:
        """Export audit logs"""
        pass
```

---

---

## 🔷 MODULE 6: Open & Customizable
### Plugin & Extension System

#### Purpose
Provide a fully open, extensible platform that users and organizations can customize for their specific needs.

#### Components

##### 6.1 Plugin Manager
```python
class PluginManager:
    """Manages plugins and extensions"""
    
    SUPPORTED_PLUGIN_TYPES = [
        'TOOL',           # Custom tools
        'MODEL_PROVIDER', # New LLM providers
        'POLICY',         # Custom policies
        'AUDIT_LOGGER',   # Custom audit logging
        'UI_EXTENSION',   # UI components
        'INSTRUMENT',     # Custom instruments
    ]
    
    def install_plugin(self, plugin_path: str) -> PluginInstallResult:
        """Install plugin from path"""
        pass
    
    def uninstall_plugin(self, plugin_id: str) -> bool:
        """Uninstall plugin"""
        pass
    
    def list_plugins(self) -> List[PluginInfo]:
        """List installed plugins"""
        pass
    
    def enable_plugin(self, plugin_id: str) -> bool:
        """Enable plugin"""
        pass
    
    def disable_plugin(self, plugin_id: str) -> bool:
        """Disable plugin"""
        pass
```

##### 6.2 Tool Adapter Framework
```python
class ToolAdapter:
    """Base class for custom tools"""
    
    @abstractmethod
    def execute(self, params: Dict) -> ToolResult:
        """Execute tool with parameters"""
        pass
    
    @abstractmethod
    def validate_params(self, params: Dict) -> ValidationResult:
        """Validate input parameters"""
        pass
    
    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Get tool schema for UI/documentation"""
        pass

# Example: Custom Docker tool
class DockerToolAdapter(ToolAdapter):
    def execute(self, params: Dict) -> ToolResult:
        """Execute docker command"""
        # Execute docker commands
        pass
```

##### 6.3 Model Provider Interface
```python
class ModelProvider(ABC):
    """Interface for adding new LLM providers"""
    
    @abstractmethod
    def initialize(self, config: ProviderConfig) -> bool:
        """Initialize provider with config"""
        pass
    
    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """List available models"""
        pass
    
    @abstractmethod
    def complete(self, prompt: str, model: str, **kwargs) -> CompletionResult:
        """Generate completion"""
        pass
    
    @abstractmethod
    def get_pricing(self, model: str) -> PricingInfo:
        """Get model pricing"""
        pass

# Example: Custom provider
class HuggingFaceProvider(ModelProvider):
    def complete(self, prompt: str, model: str, **kwargs) -> CompletionResult:
        """Use HuggingFace Inference API"""
        pass
```

##### 6.4 Policy Module System
```python
class PolicyModule(ABC):
    """Base class for custom policies"""
    
    @abstractmethod
    def evaluate_action(self, action: Action, context: Context) -> PolicyDecision:
        """Evaluate if action is allowed"""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """Get policy priority (higher = more important)"""
        pass

# Example: Pharma compliance policy
class PharmaCompliancePolicy(PolicyModule):
    def evaluate_action(self, action: Action, context: Context) -> PolicyDecision:
        """Enforce pharmaceutical industry compliance"""
        # Check for GxP compliance
        # Validate audit trail requirements
        pass
```

##### 6.5 Enterprise Auth Adapter
```python
class AuthAdapter(ABC):
    """Interface for enterprise authentication"""
    
    @abstractmethod
    def authenticate(self, credentials: Credentials) -> AuthResult:
        """Authenticate user"""
        pass
    
    @abstractmethod
    def authorize(self, user: User, action: Action) -> bool:
        """Authorize action for user"""
        pass
    
    @abstractmethod
    def get_roles(self, user: User) -> List[Role]:
        """Get user roles"""
        pass

# Examples:
class SAMLAuthAdapter(AuthAdapter):
    """SAML-based SSO authentication"""
    pass

class LDAPAuthAdapter(AuthAdapter):
    """LDAP authentication"""
    pass

class OAuth2AuthAdapter(AuthAdapter):
    """OAuth2 authentication"""
    pass
```

##### 6.6 Configuration Schema
```python
class ConfigurableComponent:
    """Component that can be configured"""
    
    @abstractmethod
    def get_config_schema(self) -> ConfigSchema:
        """Get configuration schema (JSON Schema)"""
        pass
    
    @abstractmethod
    def apply_config(self, config: Dict) -> bool:
        """Apply configuration"""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict) -> ValidationResult:
        """Validate configuration"""
        pass
```

#### Plugin API
```python
class PluginAPI:
    def register_tool(self, tool: ToolAdapter) -> str:
        """Register custom tool"""
        pass
    
    def register_model_provider(self, provider: ModelProvider) -> str:
        """Register model provider"""
        pass
    
    def register_policy(self, policy: PolicyModule) -> str:
        """Register policy module"""
        pass
    
    def register_auth_adapter(self, auth: AuthAdapter) -> str:
        """Register auth adapter"""
        pass
```

---

## 🔷 MODULE 7: Enterprise Policy
### Policy Engine

#### Purpose
Provide enterprise-grade governance, security, and compliance controls for regulated environments.

#### Components

##### 7.1 Policy Engine Core
```python
class PolicyEngine:
    """Core policy evaluation engine"""
    
    def evaluate(self, action: Action, context: Context) -> PolicyDecision:
        """
        Evaluate action against all policies:
        1. Load applicable policies (by type, environment)
        2. Sort by priority
        3. Evaluate each policy
        4. Aggregate results
        5. Return decision
        
        Returns:
        - allowed: True if allowed
        - denied: List of policies that denied
        - warnings: List of policy warnings
        - conditions: Conditions that must be met
        """
        pass
    
    def register_policy(self, policy: Policy) -> str:
        """Register new policy"""
        pass
    
    def remove_policy(self, policy_id: str) -> bool:
        """Remove policy"""
        pass
    
    def list_policies(self, filter: PolicyFilter = None) -> List[Policy]:
        """List policies"""
        pass
```

##### 7.2 Built-in Policy Library
```python
class BuiltInPolicies:
    """Pre-defined enterprise policies"""
    
    # Security Policies
    class NoSecretsInPrompts(Policy):
        """Prevent secrets in prompts"""
        def evaluate(self, action: Action, context: Context) -> PolicyDecision:
            # Scan for secrets, API keys, passwords
            pass
    
    class NoProductionChanges(Policy):
        """Block changes to production without approval"""
        def evaluate(self, action: Action, context: Context) -> PolicyDecision:
            # Check environment
            pass
    
    # Compliance Policies
    class AuditTrailRequired(Policy):
        """Ensure audit trail for all actions"""
        def evaluate(self, action: Action, context: Context) -> PolicyDecision:
            # Check if audit logging is enabled
            pass
    
    # Safety Policies
    class NoFileDeletion(Policy):
        """Prevent file deletion"""
        def evaluate(self, action: Action, context: Context) -> PolicyDecision:
            # Check if action is file deletion
            pass
    
    class ProtectedBranches(Policy):
        """Protect critical branches"""
        def evaluate(self, action: Action, context: Context) -> PolicyDecision:
            # Check if action affects protected branches
            pass
    
    # Approval Policies
    class ApprovalRequired(Policy):
        """Require approval for certain actions"""
        def evaluate(self, action: Action, context: Context) -> PolicyDecision:
            # Check if action requires approval
            pass
```

##### 7.3 Role-Based Access Control (RBAC)
```python
class RBACPolicy(Policy):
    """Role-based access control"""
    
    ROLES = {
        'admin': {'permissions': ['*']},
        'developer': {
            'permissions': [
                'read:*', 'write:code', 'run:tests',
                'commit:feature', 'push:feature'
            ]
        },
        'reviewer': {
            'permissions': [
                'read:*', 'review:*', 'approve:merge'
            ]
        },
        'viewer': {
            'permissions': ['read:*']
        },
    }
    
    def check_permission(self, user: User, action: str) -> bool:
        """Check if user has permission for action"""
        pass
```

##### 7.4 Compliance Policies
```python
class CompliancePolicy(Policy):
    """Base class for compliance policies"""
    
    COMPLIANCE_STANDARDS = {
        'SOC2': [
            'audit_trail',
            'access_control',
            'data_encryption',
            'change_management'
        ],
        'ISO27001': [
            'information_security',
            'risk_management',
            'access_control',
            'compliance_monitoring'
        ],
        'HIPAA': [
            'phi_protection',
            'audit_logging',
            'access_control',
            'data_disposal'
        ],
        'GDPR': [
            'data_protection',
            'consent_management',
            'data_portability',
            'privacy_by_design'
        ],
        'GxP': [
            'audit_trail',
            'electronic_signatures',
            'change_control',
            'validation'
        ]
    }
```

##### 7.5 Secret Detection
```python
class SecretDetector:
    """Detect secrets in prompts and code"""
    
    SECRET_PATTERNS = {
        'api_key': r'(?i)(api[_-]?key|apikey)["\'\s:=]+([a-z0-9]{20,})',
        'password': r'(?i)(password|pass|pwd)["\'\s:=]+([^\s]+)',
        'token': r'(?i)(token|bearer)["\'\s:=]+([a-z0-9\-._]{20,})',
        'aws_key': r'AKIA[0-9A-Z]{16}',
        'private_key': r'-----BEGIN (RSA )?PRIVATE KEY-----',
    }
    
    def scan_for_secrets(self, text: str) -> List[SecretFound]:
        """Scan text for secrets"""
        pass
    
    def check_prompt(self, prompt: str) -> SecretCheckResult:
        """Check if prompt contains secrets"""
        pass
```

##### 7.6 Environment Gatekeeper
```python
class EnvironmentGatekeeper:
    """Control actions based on environment"""
    
    ENVIRONMENTS = {
        'development': {
            'allowed_actions': ['*'],
            'require_approval': [],
        },
        'staging': {
            'allowed_actions': [
                'read:*', 'write:code', 'run:tests',
                'deploy:staging'
            ],
            'require_approval': ['deploy:*', 'delete:*'],
        },
        'production': {
            'allowed_actions': [
                'read:*', 'deploy:production'
            ],
            'require_approval': [
                'deploy:*', 'write:config', 'write:secrets'
            ],
        },
    }
    
    def check_environment_access(self, user: User, env: str, action: str) -> bool:
        """Check if action allowed in environment"""
        pass
```

#### Policy Definition Format
```python
@dataclass
class Policy:
    policy_id: str
    name: str
    description: str
    
    # Scope
    applies_to: List[ActionType]
    environments: List[str]
    roles: List[str]
    
    # Rules
    conditions: List[Condition]
    effect: PolicyEffect  # ALLOW, DENY, REQUIRE_APPROVAL
    
    # Metadata
    priority: int
    enabled: bool
    created_at: datetime
    created_by: str
```

#### API
```python
class PolicyEngineAPI:
    def evaluate_action(self, action: Action, context: Context) -> PolicyDecision:
        """Evaluate action against policies"""
        pass
    
    def add_policy(self, policy: Policy) -> str:
        """Add new policy"""
        pass
    
    def remove_policy(self, policy_id: str) -> bool:
        """Remove policy"""
        pass
    
    def list_policies(self) -> List[Policy]:
        """List all policies"""
        pass
    
    def test_policy(self, policy: Policy, action: Action) -> PolicyDecision:
        """Test policy against action without applying"""
        pass
    
    def get_compliance_report(self, standard: str) -> ComplianceReport:
        """Get compliance report for standard"""
        pass
```

---

---

## 🚀 Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
**Priority: CRITICAL**

| Module | Component | Effort | Dependencies |
|--------|-----------|--------|---------------|
| **Repo Awareness** | File System Indexer | 2 days | - |
| **Repo Awareness** | Dependency Graph Builder | 3 days | File System Indexer |
| **Explainability** | Decision Logger | 2 days | - |
| **Explainability** | Execution Tracer | 2 days | - |

**Deliverables:**
- Persistent project state storage
- Basic code dependency tracking
- Decision logging infrastructure
- Execution trace framework

---

### Phase 2: Governance (Weeks 3-4)
**Priority: HIGH**

| Module | Component | Effort | Dependencies |
|--------|-----------|--------|---------------|
| **Git Governor** | Diff Analyzer | 2 days | Repo Awareness |
| **Git Governor** | Commit Synthesizer | 2 days | Diff Analyzer |
| **Git Governor** | Push Approval Engine | 3 days | Commit Synthesizer |
| **Safe Autonomy** | Intent Parser | 2 days | - |
| **Safe Autonomy** | Execution Validator | 3 days | Intent Parser |

**Deliverables:**
- Safe git operations with approval workflow
- Diff-based risk analysis
- Intent-aware command validation

---

### Phase 3: Intelligence (Weeks 5-6)
**Priority: HIGH**

| Module | Component | Effort | Dependencies |
|--------|-----------|--------|---------------|
| **Model Routing** | Task Classifier | 2 days | - |
| **Model Routing** | Model Selector | 3 days | Task Classifier |
| **Model Routing** | Fallback Manager | 2 days | Model Selector |
| **Explainability** | Plan Auditor | 2 days | Repo Awareness |
| **Explainability** | Diff Summarizer | 2 days | Git Governor |

**Deliverables:**
- Intelligent model routing by task type
- Cost optimization
- Automatic fallback chains
- Plan quality assessment

---

### Phase 4: Security & Compliance (Weeks 7-8)
**Priority: MEDIUM**

| Module | Component | Effort | Dependencies |
|--------|-----------|--------|---------------|
| **Enterprise Policy** | Policy Engine Core | 3 days | - |
| **Enterprise Policy** | Built-in Policy Library | 3 days | Policy Engine |
| **Enterprise Policy** | RBAC Policy | 2 days | Policy Engine |
| **Enterprise Policy** | Secret Detector | 2 days | Policy Engine |
| **Safe Autonomy** | Environment Locks | 2 days | Policy Engine |
| **Safe Autonomy** | Command Filter | 2 days | Policy Engine |

**Deliverables:**
- Full policy engine with built-in policies
- Role-based access control
- Secret detection and prevention
- Environment-based access control

---

### Phase 5: Extensibility (Weeks 9-10)
**Priority: MEDIUM**

| Module | Component | Effort | Dependencies |
|--------|-----------|--------|---------------|
| **Open & Customizable** | Plugin Manager | 3 days | - |
| **Open & Customizable** | Tool Adapter Framework | 2 days | Plugin Manager |
| **Open & Customizable** | Model Provider Interface | 2 days | Plugin Manager |
| **Open & Customizable** | Policy Module System | 2 days | Policy Engine |
| **Open & Customizable** | Enterprise Auth Adapter | 3 days | Policy Engine |

**Deliverables:**
- Full plugin system
- Custom tool, model, and policy support
- Enterprise authentication adapters

---

### Phase 6: Polish (Weeks 11-12)
**Priority: LOW**

| Module | Component | Effort | Dependencies |
|--------|-----------|--------|---------------|
| **All Modules** | Integration Testing | 5 days | All previous |
| **Explainability** | Audit Report Generator | 2 days | All previous |
| **All Modules** | Documentation | 3 days | All previous |

**Deliverables:**
- Complete integration test suite
- Comprehensive audit reporting
- Full documentation

---

## 🔄 Data Flow Integration

### Complete Request Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER REQUEST                                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │    Intent Parser        │  ◄─── Safe Autonomy
              │    (classify task)      │
              └──────────┬──────────────┘
                         │
                         ▼
              ┌─────────────────────────┐
              │    Task Classifier      │  ◄─── Model Routing
              │    (determine type)      │
              └──────────┬──────────────┘
                         │
           ┌─────────────┴─────────────┐
           │                           │
           ▼                           ▼
┌──────────────────────┐   ┌─────────────────────────┐
│   Repo Awareness     │   │   Model Selector        │ ◄─── Model Routing
│   (get context)      │   │   (pick best model)     │
└──────────┬───────────┘   └──────────┬──────────────────┘
           │                          │
           └──────────┬───────────────┘
                      │
                      ▼
           ┌─────────────────────────┐
           │    Action Planner       │  ◄─── Safe Autonomy
           │    (create plan)        │
           └──────────┬──────────────┘
                      │
                      ▼
           ┌─────────────────────────┐
           │   Plan Auditor         │  ◄─── Explainability
n           │   (audit plan)         │
           └──────────┬──────────────┘
                      │
                      ▼
           ┌─────────────────────────┐
           │  Execution Validator    │  ◄─── Safe Autonomy
           │  (validate actions)     │
           └──────────┬──────────────┘
                      │
                      ▼
           ┌─────────────────────────┐
           │   Policy Engine         │  ◄─── Enterprise Policy
           │   (check permissions)   │
           └──────────┬──────────────┘
                      │
           ┌─────────┴─────────┐
           │                   │
           ▼ (approve)        ▼ (deny)
     ┌──────────┐         ┌──────────┐
     │ Execute  │         │  Reject  │
     └────┬─────┘         └──────────┘
          │
          ▼
   ┌─────────────────┐
   │ Execution Trace │  ◄─── Explainability
   │ (log each step) │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ Git Governor    │  ◄─── Git Governor
   │ (if code change)│
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ Diff Summarizer  │  ◄─── Explainability
   │ (summarize)      │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ Decision Logger  │  ◄─── Explainability
   │ (log decisions)  │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ Audit Report     │  ◄─── Explainability
   │ Generator        │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │   RESPONSE       │
   │   (to user)      │
   └─────────────────┘
```

---

## 📊 Module Interactions

### Dependency Graph

```
                 +-------------------+
                 |     Orchestator   |
                 +---------+---------+
                           |
       +-------------------+-------------------+
       |                   |                   |
+------v------+   +-------v-------+   +-------v-------+
| Repo        |   | Model         |   | Safe          |
| Awareness   |   | Router        |   | Autonomy      |
+------+------+-   +-------+-------+   +-------+-------+
       |                  |                   |
       |    +-------------+-------+             |
       |    |             |       |             |
+------v----v----+  +---v-------v---+   +-----v-------+
| Git Governor  |  | Explainability|   | Policy      |
+-------+--------+  +-------+-------+   +-----+-------+
        |                  |                  |
        +----------+-------+                  |
                   |                        |
            +------v------------------------v---+
            |    Open & Customizable          |
            +----------------------------------+
```

---

## ✅ Competitive Advantages vs Antigravity

| Capability | Antigravity | Agent Zero v2 | Winner |
|-------------|-------------|---------------|--------|
| **Repo Awareness** | Full | Full + Semantic Indexing | **Agent Zero** |
| **Git Operations** | Native | Governed + Auditable | **Agent Zero** |
| **Safety** | Strong | Strong + Policy Engine | **Agent Zero** |
| **Model Routing** | Hidden | Transparent + Customizable | **Agent Zero** |
| **Explainability** | Limited | Full Self-Audit | **Agent Zero** |
| **Open/Customizable** | Closed | Full Plugin System | **Agent Zero** |
| **Enterprise Policy** | Built-in | Built-in + Extensible | **Agent Zero** |
| **Cost Control** | Hidden | Full Transparency | **Agent Zero** |
| **Local Deployment** | None | Supported | **Agent Zero** |
| **Research Ready** | No | Yes | **Agent Zero** |

---

## 🎯 Use Case Fit

### Best For Agent Zero v2:

✅ **Research & Academia** - Full explainability and audit trails

✅ **Pharma & Healthcare** - GxP compliance, policy enforcement

✅ **Government** - Security, audit trails, RBAC

✅ **Enterprise** - Custom policies, enterprise auth, on-prem

✅ **Open Source Projects** - Transparent, customizable

✅ **Cost-Sensitive Organizations** - Explicit cost control

---

## 📦 Technical Stack Recommendations

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.11+ (FastAPI) |
| **Frontend** | React + TypeScript |
| **Database** | PostgreSQL + Redis (caching) |
| **Vector DB** | ChromaDB (semantic search) |
| **Message Queue** | Redis / RabbitMQ |
| **LLM Integration** | LiteLLM (multi-provider) |
| **Git Operations** | GitPython / pygit2 |
| **Plugin System** | Python entry_points |
| **Authentication** | Python Social Auth (SAML, LDAP, OAuth2) |
| **Containerization** | Docker + Docker Compose |
| **Monitoring** | Prometheus + Grafana |

---

## 🎓 Key Architectural Principles

1. **Transparency First** - Every decision is logged and explainable

2. **Safety by Default** - Unsafe operations require explicit approval

3. **Open Architecture** - Everything is pluggable and customizable

4. **Git-Native** - Git operations are first-class, governed, audited

5. **Policy-Driven** - All actions pass through policy engine

6. **Model-Agnostic** - Support for any LLM provider via plugins

7. **Context-Aware** - Persistent repository state enables smarter decisions

8. **Audit-Ready** - Complete traceability for compliance

---

## 📝 Next Steps

1. **Review Architecture** - Share with stakeholders for feedback

2. **Choose Starting Phase** - Begin with Phase 1 (Foundation)
3. **Setup Development Environment** - Initialize project structure
4. **Create Proof of Concept** - Implement Repo Awareness MVP
5. **Iterative Development** - Follow roadmap phases
6. **Continuous Integration** - Set up automated testing
7. **User Testing** - Gather feedback from early adopters
8. **Refine and Polish** - Based on feedback and metrics

---

## 🌟 Success Metrics

| Metric | Target |
|--------|--------|
| **Decision Transparency** | 100% of decisions logged |
| **Audit Trail Completeness** | 100% of actions traceable |
| **Policy Compliance** | 100% enforcement |
| **Model Cost Optimization** | 30% reduction vs single-model |
| **Git Safety** | 0% unintended pushes to production |
| **Response Time** | < 2s for common tasks |
| **Extensibility** | < 1 day to add new tool/model/policy |

---

## 📚 Document Information

- **Version:** 1.0
- **Date:** 2026-01-12
- **Author:** Agent Zero AI
- **Status:** Architecture Complete - Ready for Implementation

---

**End of Agent Zero v2 Architecture Document**
