# Phase 2: Git Governor & Model Router
# Building on Phase 1 Foundation

## Overview

Phase 2 implements two critical autonomous development subsystems:
1. **Git Governor** - Governed commits with approval workflows
2. **Model Router** - Intelligent LLM routing by task type

## Goals

### Git Governor Goals
- Enforce branch-based development (never commit to main)
- Require approval for risky operations (file deletion, production changes)
- Generate semantic commit messages (conventional format)
- Perform diff analysis and risk assessment before committing
- Validate no secrets or production config changes
- Check CI status before pushing
- Maintain safe checkpoints and rollback options

### Model Router Goals
- Classify tasks by type: PLANNING, CODE_GEN, REFACTOR, DEBUG, TEST, DOCS, QUERY
- Route tasks to appropriate models (high-quality vs fast/cheap)
- Track quality and cost metrics
- Optimize model selection over time
- Support fallback models when primary unavailable

## Success Criteria

### Git Governor
- All commits go through validation
- No commits to main/production without override
- All risky operations require approval
- Semantic commit messages generated
- Secret detection working
- Rollback capability tested

### Model Router
- All tasks classified correctly
- Appropriate models selected
- Cost/quality tracked
- Fallback models working
- Routing decisions explainable
- Integration with Phase 1 complete

## Integration with Phase 1

- Git Governor uses Explainability (decision logging)
- Model Router uses Safe Autonomy (execution boundaries)
- Both use Repo Awareness (dependency context)
- Both use Persistent State (checkpoints, metrics)
- Both enforce through Control Contract
