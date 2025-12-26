"""
BioDockify Sentinel - Self-Healing System
Active Monitoring & Remediation Engine
"""

import boto3
import os
from datetime import datetime, timedelta
from supabase import Client
import traceback
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentinel")

class BioDockifySentinel:
    def __init__(self, supabase_client: Client):
        self.db = supabase_client
        self.batch = boto3.client('batch', region_name=os.getenv("AWS_REGION", "us-east-1"))
        
    async def scan_and_heal(self):
        """
        Main Loop: Scans system for anomalies and triggers healing.
        Returns a report of actions taken.
        """
        report = {
            "scanned_at": datetime.utcnow().isoformat(),
            "anomalies_detected": 0,
            "actions_taken": []
        }
        
        print("🤖 Sentinel: Starting System Scan...")
        
        # 1. Check for Stuck PREPARATION jobs (Processing > 30 mins)
        await self._heal_stuck_preparation(report)
        
        # 2. Check for Stuck SUBMITTED jobs (Submitted > 2 hours)
        # This usually means they are queued in AWS but not getting picked up, or AWS failed silently.
        await self._monitor_aws_batch_state(report)
        
        # 3. Check for Zombie Jobs (Running > 24 hours)
        await self._flag_zombies(report)
        
        return report

    async def _heal_stuck_preparation(self, report):
        """
        Detects jobs stuck in 'PROCESSING' state (local PDBQT conversion)
        """
        threshold = datetime.utcnow() - timedelta(minutes=30)
        
        response = self.db.table("jobs") \
            .select("*") \
            .eq("status", "PROCESSING") \
            .lt("created_at", threshold.isoformat()) \
            .execute()
            
        stuck_jobs = response.data or []
        
        for job in stuck_jobs:
            report["anomalies_detected"] += 1
            action = f"Auto-Healing Job {job['id']}: Stuck in PREP. Resetting to FAILED to allow manual retry."
            
            # Action: For now, Mark as FAILED with a helpful message. 
            # In V2, we could auto-restart the conversion with 'lenient' mode.
            
            msg = "Sentinel: Job stuck in preparation for > 30mins. Marked as Failed."
            self.db.table("jobs").update({
                "status": "FAILED",
                "error_message": msg
            }).eq("id", job['id']).execute()
            
            self._log_sentinel_action(job['id'], "Stuck Prep", "Marked FAILED")
            report["actions_taken"].append(action)

    async def _monitor_aws_batch_state(self, report):
        """
        Checks AWS Batch connectivity and status of submitted jobs.
        Detects Spot Instance terminations.
        """
        # Get jobs that are SUBMITTED or RUNNING locally
        response = self.db.table("jobs") \
            .select("*") \
            .in_("status", ["SUBMITTED", "RUNNING"]) \
            .execute()
            
        active_jobs = response.data or []
        if not active_jobs: return

        # Group by Batch ID to minimize API calls
        # Note: AWS Batch DescribeJobs accepts max 100 IDs.
        
        # For prototype, we check subsets
        aws_ids = [j['batch_job_id'] for j in active_jobs if j.get('batch_job_id')]
        if not aws_ids: return
        
        # Split into chunks of 100
        chunks = [aws_ids[i:i + 100] for i in range(0, len(aws_ids), 100)]
        
        for chunk in chunks:
            try:
                aws_resp = self.batch.describe_jobs(jobs=chunk)
                aws_jobs = {j['jobId']: j for j in aws_resp.get('jobs', [])}
                
                for local_job in active_jobs:
                    aws_id = local_job.get('batch_job_id')
                    if aws_id in aws_jobs:
                        aws_status = aws_jobs[aws_id]['status']
                        status_reason = aws_jobs[aws_id].get('statusReason', '')
                        
                        # --- DETECT SPOT FAILURE ---
                        if aws_status == 'FAILED' and "Spot" in status_reason:
                            report["anomalies_detected"] += 1
                            
                            # ACTION: AUTO-RETRY (Logic to come)
                            # For now, we log it. Real healing needs re-submission logic.
                            action = f"Sentinel: Detected SPOT Instance Failure for {local_job['id']}. Requesting creation of new job..."
                            
                            # Update DB to reflect precise error
                            self.db.table("jobs").update({
                                "status": "FAILED", 
                                "error_message": "AWS Spot Instance Reclaimed. Please Retry."
                            }).eq("id", local_job['id']).execute()
                            
                            self._log_sentinel_action(local_job['id'], "Spot Failure", "Marked FAILED (Spot Reclaim)")
                            report["actions_taken"].append(action)

            except Exception as e:
                print(f"Sentinel AWS Poll Error: {e}")

    async def _flag_zombies(self, report):
        """Jobs running for > 24 hours are likely zombies."""
        threshold = datetime.utcnow() - timedelta(hours=24)
        
        response = self.db.table("jobs") \
            .select("*") \
            .eq("status", "RUNNING") \
            .lt("created_at", threshold.isoformat()) \
            .execute()
            
        zombies = response.data or []
        for job in zombies:
            report["anomalies_detected"] += 1
            # Just log for admin review
            self._log_sentinel_action(job['id'], "Zombie Job (>24h)", "Flagged for Admin")
            report["actions_taken"].append(f"Flagged Zombie Job {job['id']}")

    def _log_sentinel_action(self, target_id, anomaly, action):
        """Log to Admin Actions table"""
        try:
            self.db.table("admin_actions").insert({
                "action_type": "sentinel_heal",
                "target_type": "job",
                "target_id": target_id,
                "details": {"anomaly": anomaly, "action": action},
                # 'admin_id' might be null or a special 'system' UUID if enforced
            }).execute()
        except:
            pass # Best effort logging
