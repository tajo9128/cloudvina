#!/usr/bin/env python3
"""
Diagnostic script to debug why batch jobs aren't completing.
Run this locally or in cloud shell to identify the issue.
"""

import os
import sys
from datetime import datetime, timezone
import json

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_datetime_parsing():
    """Test 1: Verify datetime parsing works correctly"""
    print("\n=== TEST 1: Datetime Parsing ===")
    
    # Simulate database timestamp formats
    test_timestamps = [
        "2025-12-21T12:14:24.539Z",
        "2025-12-21T12:14:24+00:00",
        "2025-12-21 12:14:24.539000+00:00"
    ]
    
    for ts_str in test_timestamps:
        try:
            # Test the exact logic from batch.py
            created_at_str = ts_str
            if created_at_str.endswith('Z'):
                created_at_str = created_at_str[:-1] + '+00:00'
            created_at = datetime.fromisoformat(created_at_str)
            
            # If still naive, make it UTC
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            
            current_time = datetime.now(timezone.utc)
            age = (current_time - created_at).total_seconds()
            
            print(f"✓ Input: {ts_str}")
            print(f"  Parsed: {created_at}")
            print(f"  Age: {age:.1f} seconds")
            print(f"  Would simulate: {age > 10}\n")
        except Exception as e:
            print(f"✗ Failed to parse {ts_str}: {e}\n")

def test_boto3_import():
    """Test 2: Verify boto3 can be imported and initialized"""
    print("\n=== TEST 2: Boto3 Availability ===")
    
    try:
        import boto3
        print("✓ boto3 imported successfully")
        
        # Try creating clients
        try:
            AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
            s3 = boto3.client('s3', region_name=AWS_REGION)
            print(f"✓ S3 client created (region: {AWS_REGION})")
            
            batch_client = boto3.client('batch', region_name=AWS_REGION)
            print(f"✓ Batch client created")
            
            # Check if credentials are configured
            try:
                sts = boto3.client('sts', region_name=AWS_REGION)
                identity = sts.get_caller_identity()
                print(f"✓ AWS credentials valid")
                print(f"  Account: {identity['Account']}")
                print(f"  User ARN: {identity['Arn']}")
            except Exception as e:
                print(f"⚠ AWS credentials issue: {e}")
                
        except Exception as e:
            print(f"✗ Failed to create AWS clients: {e}")
            print(f"  This would trigger: 'WARNING: AWS client initialization failed'")
            
    except ImportError as e:
        print(f"✗ boto3 not installed: {e}")

def test_supabase_connection():
    """Test 3: Verify Supabase connection works"""
    print("\n=== TEST 3: Supabase Connection ===")
    
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("✗ Missing environment variables:")
        print(f"  SUPABASE_URL: {'✓' if SUPABASE_URL else '✗'}")
        print(f"  SUPABASE_SERVICE_KEY: {'✓' if SUPABASE_KEY else '✗'}")
        return
    
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✓ Supabase client created")
        
        # Test query
        result = supabase.table('jobs').select('id,status,created_at').limit(1).execute()
        print(f"✓ Can query jobs table ({len(result.data)} rows)")
        
        if result.data:
            print(f"  Sample: {result.data[0]}")
    except Exception as e:
        print(f"✗ Supabase connection failed: {e}")

def test_simulation_logic():
    """Test 4: Simulate the job completion logic"""
    print("\n=== TEST 4: Simulation Logic ===")
    
    # Simulate a job that's 15 seconds old
    import random
    from datetime import timedelta
    
    current_time = datetime.now(timezone.utc)
    created_at = current_time - timedelta(seconds=15)
    age = (current_time - created_at).total_seconds()
    
    print(f"Job created: {created_at}")
    print(f"Current time: {current_time}")
    print(f"Age: {age:.1f} seconds")
    print(f"Age > 10: {age > 10}")
    
    if age > 10:
        print("\n✓ Job would be simulated!")
        
        vina_val = -7.5 + (random.random() * -2.5)
        gnina_val = vina_val - (random.random() * 0.5)
        
        print(f"  Vina score: {vina_val:.2f}")
        print(f"  Gnina score: {gnina_val:.2f}")
        print(f"  Status: SUCCEEDED")
    else:
        print("\n✗ Job too young, would not simulate yet")

def get_actual_job_from_db():
    """Test 5: Get a real job from database and check it"""
    print("\n=== TEST 5: Real Job Analysis ===")
    
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("⚠ Skipping - Supabase credentials not configured")
        return
    
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Get latest NULL job
        result = supabase.table('jobs')\
            .select('id,status,created_at,binding_affinity,batch_id')\
            .is_('binding_affinity', 'null')\
            .order('created_at', desc=True)\
            .limit(1)\
            .execute()
        
        if not result.data:
            print("✓ No NULL jobs found (all completed!)")
            return
        
        job = result.data[0]
        print(f"Found NULL job: {job['id']}")
        print(f"  Status: {job['status']}")
        print(f"  Batch: {job['batch_id']}")
        print(f"  Created: {job['created_at']}")
        
        # Calculate age
        created_at_str = job['created_at']
        if created_at_str.endswith('Z'):
            created_at_str = created_at_str[:-1] + '+00:00'
        created_at = datetime.fromisoformat(created_at_str)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        
        current_time = datetime.now(timezone.utc)
        age = (current_time - created_at).total_seconds()
        
        print(f"  Age: {age:.1f} seconds")
        print(f"  Should simulate: {age > 10}")
        
        if age > 10:
            print("\n⚠ This job SHOULD have been completed by simulation!")
            print("  → Simulation is NOT running on server")
        else:
            print("\n✓ Job is too young, needs more time")
            
    except Exception as e:
        print(f"✗ Error querying database: {e}")

def main():
    print("=" * 60)
    print("BioDockify Job Completion Diagnostic Tool")
    print("=" * 60)
    
    # Run all tests
    test_datetime_parsing()
    test_boto3_import()
    test_supabase_connection()
    test_simulation_logic()
    get_actual_job_from_db()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nIf all tests pass but jobs still NULL:")
    print("  → GET /jobs/batch/{id} endpoint is NOT being called")
    print("  → Frontend not polling or user not visiting results page")
    print("\nIf boto3 test fails:")
    print("  → AWS credentials missing or invalid")
    print("  → Simulation cannot run")
    print("\nIf datetime test fails:")
    print("  → Jobs don't meet age condition")
    print("  → Simulation skipped due to logic error")
    print()

if __name__ == "__main__":
    main()
